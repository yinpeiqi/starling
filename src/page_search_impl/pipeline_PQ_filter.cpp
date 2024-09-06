#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include "logger.h"
#include "percentile_stats.h"
#include "pq_flash_index.h"
#include "timer.h"

namespace diskann {
  template<typename T>
  void PQFlashIndex<T>::page_search(
      const T *query1, const _u64 k_search, const _u32 mem_L, const _u64 l_search, _u64 *indices,
      float *distances, const _u64 beam_width, const _u32 io_limit,
      const bool use_reorder_data, const float use_ratio, QueryStats *stats) {
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }

    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    float        query_norm = 0;
    const T *    query = data.scratch.aligned_query_T;
    const float *query_float = data.scratch.aligned_query_float;

    uint32_t query_dim = metric == diskann::Metric::INNER_PRODUCT ? this-> data_dim - 1: this-> data_dim;

    for (uint32_t i = 0; i < query_dim; i++) {
      data.scratch.aligned_query_float[i] = query1[i];
      data.scratch.aligned_query_T[i] = query1[i];
      query_norm += query1[i] * query1[i];
    }

    // if inner product, we also normalize the query and set the last coordinate
    // to 0 (this is the extra coordindate used to convert MIPS to L2 search)
    if (metric == diskann::Metric::INNER_PRODUCT) {
      query_norm = std::sqrt(query_norm);
      data.scratch.aligned_query_T[this->data_dim - 1] = 0;
      data.scratch.aligned_query_float[this->data_dim - 1] = 0;
      for (uint32_t i = 0; i < this->data_dim - 1; i++) {
        data.scratch.aligned_query_T[i] /= query_norm;
        data.scratch.aligned_query_float[i] /= query_norm;
      }
    }

    IOContext &ctx = data.ctx;
    auto       query_scratch = &(data.scratch);

    // reset query
    query_scratch->reset();

    // pointers to buffers for data
    T *   data_buf = query_scratch->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    // query <-> PQ chunk centers distances
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query_float, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = query_scratch->aligned_dist_scratch;
    _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

    Timer                 query_timer, io_timer, cpu_timer, tmp_timer, part_timer, subpart_timer;
    std::vector<Neighbor> retset(l_search + 1);
    // std::vector<bool> visited(10000000, false);
    tsl::robin_set<_u64> &visited = *(query_scratch->visited);
    // tsl::robin_set<unsigned> &page_visited = *(query_scratch->page_visited);    // deperate
    tsl::robin_map<_u64, bool> exact_visited;
    unsigned cur_list_size = 0;

    std::vector<Neighbor> full_retset;
    full_retset.reserve(4096);

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
                                                            const _u64 n_ids,
                                                            float *dists_out) {
      pq_flash_index_utils::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                         pq_coord_scratch);
      pq_flash_index_utils::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                       dists_out);
    };

    auto compute_exact_dists_and_push = [&](const char* node_buf, const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;
      tmp_timer.reset();
      memcpy(node_fp_coords_copy, node_buf, disk_bytes_per_point);
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy,
                                            (unsigned) aligned_dim);
      if (stats != nullptr) {
        stats->n_ext_cmps++;
        stats->cmp_us += (double) tmp_timer.elapsed();
      }
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      exact_visited[id] = true;
      return cur_expanded_dist;
    };

    auto compute_and_push_nbrs = [&](const char *node_buf, unsigned& nk) {
      unsigned *node_nbrs = OFFSET_TO_NODE_NHOOD(node_buf);
      unsigned nnbrs = *(node_nbrs++);
      unsigned nbors_cand_size = 0;
      tmp_timer.reset();
      for (unsigned m = 0; m < nnbrs; ++m) {
        if (visited.find(node_nbrs[m]) != visited.end()) {
          continue;
        }
        else {
          visited.insert(node_nbrs[m]);
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
        }
      }
      if (stats != nullptr) {
        stats->insert_visited_us += (double) tmp_timer.elapsed();
        stats->insert_visited += nbors_cand_size;
        stats->check_visited += nnbrs;
      }
      if (nbors_cand_size) {
        tmp_timer.reset();
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += (double) nbors_cand_size;
          stats->cmp_us += (double) tmp_timer.elapsed();
        }
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node_nbrs[m];
          const float nbor_dist = dist_scratch[m];
          if (nbor_dist >= retset[cur_list_size - 1].distance &&
              (cur_list_size == l_search)) {
            continue;
          }
          Neighbor nn(nbor_id, nbor_dist, true);
          // Return position in sorted list where nn inserted
          auto     r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < l_search) ++cur_list_size;
          // nk logs the best position in the retset that was updated due to neighbors of n.
          if (r < nk) nk = r;
        }
      }
    };

    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
      compute_pq_dists(node_ids, n_ids, dist_scratch);
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        // visited[node_ids[i]] = true;
        visited.insert(node_ids[i]);
      }
    };

    tmp_timer.reset();
    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      std::vector<T*> res = std::vector<T*>();
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data(), nullptr, res);
      compute_and_add_to_retset(mem_tags.data(), std::min((unsigned)mem_L,(unsigned)l_search));
    } else {
      _u32   best_medoid = 0;
      float  best_dist = (std::numeric_limits<float>::max)();
      for (_u64 cur_m = 0; cur_m < num_medoids; cur_m++) {
        float cur_expanded_dist = dist_cmp_float->compare(
            query_float, centroid_data + aligned_dim * cur_m,
            (unsigned) aligned_dim);
        if (cur_expanded_dist < best_dist) {
          best_medoid = medoids[cur_m];
          best_dist = cur_expanded_dist;
        }
      }
      compute_and_add_to_retset(&best_medoid, 1);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    if (stats != nullptr) {
      stats->preprocess_us += (double) tmp_timer.elapsed();
    }
    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    tsl::robin_map<char*, _u32> frontier_nhoods;
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);

    // these data are count seperately
    int n_io_in_q = 0; // how many io left
    int n_proc_in_q = 0; // how many proc left
    int tot_proc = 0;

    // pre-allocated here
    std::vector<const char*> vis_cand;
    vis_cand.reserve(20);   // at most researve 12 is enough

    // these three data are in beam level
    unsigned nk = cur_list_size;
    sector_scratch_idx = 0;
    // this is a ring queue.
    std::vector<char*> sector_buffers(MAX_N_SECTOR_READS);
    int top_bufs_idx = 0;
    int botm_bufs_idx = 0;
    while (k < cur_list_size && num_ios < io_limit) {

      if (n_io_in_q > 0) {
        unsigned min_r = 0;
        if (n_proc_in_q == 0) min_r = 1;
        part_timer.reset();
        std::vector<char*> tmp_bufs(beam_width * 2);
        int n_read_blks = reader->get_events(ctx, min_r, n_io_in_q, tmp_bufs);
        for (int i = n_read_blks - 1; i >= 0; i--) {
          sector_buffers[top_bufs_idx] = tmp_bufs[i];
          top_bufs_idx = (top_bufs_idx + 1) % MAX_N_SECTOR_READS;
        }
        if (stats != nullptr) stats->read_disk_us += (double) part_timer.elapsed();
        n_io_in_q -= n_read_blks;
        n_proc_in_q += n_read_blks;
      }

      if (n_proc_in_q > 0) {
        part_timer.reset();
        auto sector_buf = sector_buffers[botm_bufs_idx];
        botm_bufs_idx = (botm_bufs_idx + 1) % MAX_N_SECTOR_READS;
        if (frontier_nhoods.find(sector_buf) == frontier_nhoods.end()) {
          std::cout << "read error: " << int((sector_buf - sector_scratch) / SECTOR_LEN) << " " << std::this_thread::get_id() << std::endl;
          std::cout << n_io_in_q << " " << n_proc_in_q << std::endl;
          exit(-1);
        }

        _u32 exact_id = frontier_nhoods[sector_buf];
        unsigned pid = id2page_[exact_id];
        const unsigned p_size = gp_layout_[pid].size();
        unsigned* node_in_page = gp_layout_[pid].data();
        compute_pq_dists(node_in_page, p_size, dist_scratch);

        unsigned cand_size = 0;
        for (unsigned j = 0; j < p_size; ++j) {
          unsigned id = gp_layout_[pid][j];
          if (id != exact_id && (dist_scratch[j] >= retset[cur_list_size - 1].distance * 1.2 
              || exact_visited[id])) {
            continue;
          }
          char *node_buf = sector_buf + j * max_node_len;
          compute_exact_dists_and_push(node_buf, id);
          vis_cand[cand_size++] = node_buf;
        }
        for (unsigned j = 0; j < cand_size; ++j) {
          compute_and_push_nbrs(vis_cand[j], nk);
        }
        if (stats != nullptr) stats->disk_proc_us += (double) part_timer.elapsed();

        frontier_nhoods.erase(sector_buf);
        n_proc_in_q--;
        tot_proc++;
      }

      if (n_io_in_q == 0 && n_proc_in_q < beam_width) {
        // clear iteration state
        frontier.clear();
        frontier_read_reqs.clear();
        // find new beam
        if (nk < k) k = nk;
        _u32 marker = k;
        _u32 num_seen = 0;

        // distribute cache and disk-read nodes
        part_timer.reset();
        while (marker < cur_list_size && num_seen < beam_width) {
          if (retset[marker].flag) {
            const unsigned pid = id2page_[retset[marker].id];
            if (exact_visited.find(retset[marker].id) == exact_visited.end()) {
            // if (page_visited.find(pid) == page_visited.end()) {
              num_seen++;
              frontier.push_back(retset[marker].id);
              for (unsigned j = 0; j < gp_layout_[pid].size(); ++j) {
                unsigned id = gp_layout_[pid][j];
                if (exact_visited.find(id) == exact_visited.end())
                  exact_visited.insert({id, false});
                else if (stats != nullptr) {
                  stats->repeat_read++;
                }
              }
              // page_visited.insert(pid);
            }
            retset[marker].flag = false;
          }
          marker++;
        }
        if (stats != nullptr) stats->dispatch_us += (double) part_timer.elapsed();

        // read nhoods of frontier ids
        if (!frontier.empty()) {
          part_timer.reset();
          if (stats != nullptr) stats->n_hops++;
          n_io_in_q += frontier.size();
          for (_u64 i = 0; i < frontier.size(); i++) {
            auto id = frontier[i];
            auto sector_buf = sector_scratch + sector_scratch_idx * SECTOR_LEN;
            sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
            auto offset = (static_cast<_u64>(id2page_[id]+1)) * SECTOR_LEN;
            // if (frontier_nhoods.find(sector_buf) != frontier_nhoods.end()) {
            //   std::cout << "insert false! " << int((sector_buf - sector_scratch) / SECTOR_LEN) << " " << id << std::endl;
            // }
            frontier_nhoods.insert({sector_buf, id});
            frontier_read_reqs.push_back(AlignedRead(offset, SECTOR_LEN, sector_buf));
            if (stats != nullptr) {
              stats->n_4k++;
              stats->n_ios++;
            }
            num_ios++;
          }
          reader->submit_reqs(frontier_read_reqs, ctx);
          if (stats != nullptr) stats->read_disk_us += (double) part_timer.elapsed();
        }
        nk = cur_list_size;
        if (n_io_in_q == 0 && n_proc_in_q == 0) k = cur_list_size;
      }
    }
    tmp_timer.reset();

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) {
                return left.distance < right.distance;
              });

    // copy k_search values
    _u64 t = 0;
    for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
      if(i > 0 && full_retset[i].id == full_retset[i-1].id){
        continue;
      }
      indices[t] = full_retset[i].id;
      if (distances != nullptr) {
        distances[t] = full_retset[i].distance;
        if (metric == diskann::Metric::INNER_PRODUCT) {
          // flip the sign to convert min to max
          distances[t] = (-distances[t]);
          // rescale to revert back to original norms (cancelling the effect of
          // base and query pre-processing)
          if (max_base_norm != 0)
            distances[t] *= (max_base_norm * query_norm);
        }
      }
      t++;
    }

    if (t < k_search) {
      diskann::cerr << "The number of unique ids is less than topk" << std::endl;
      exit(1);
    }

    this->thread_data.push(data);
    this->thread_data.push_notify_all();

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
      stats->postprocess_us = (double) tmp_timer.elapsed();
    }
  }

  template class PQFlashIndex<_u8>;
  template class PQFlashIndex<_s8>;
  template class PQFlashIndex<float>;

} // namespace diskann
