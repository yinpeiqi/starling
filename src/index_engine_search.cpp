#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include "logger.h"
#include "percentile_stats.h"
#include "index_engine.h"
#include "timer.h"

namespace diskann {
  // data could be parse streamingly from queue.
  template<typename T>
  void IndexEngine<T>::page_search(
      const T *query_ptr, const _u64 query_num, const _u64 query_aligned_dim, const _u64 k_search, const _u32 mem_L,
      const _u64 l_search, std::vector<_u64>& indices_vec, std::vector<float>& distances_vec,
      const _u64 beam_width, const _u32 io_limit, const bool use_reorder_data,
      const float pq_filter_ratio, QueryStats* stats_ptr) {
    // here are global checks / global data init.
    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);

    uint32_t query_dim = metric == diskann::Metric::INNER_PRODUCT ? this-> data_dim - 1: this-> data_dim;

    // atomic pointer to query.
    std::atomic_int cur_task = 0;
    // parallel for
    pool->runTask([&, this](int tid) {
      // thread local data init
      IOContext& ctx = ctxs[tid];
      auto scratch = scratchs[tid];
      auto query_scratch = &(scratchs[tid]);
      // these pointers can init earlier
      const T *    query = scratch.aligned_query_T;
      const float *query_float = scratch.aligned_query_float;
      // sector scratch
      _u64 &sector_scratch_idx = query_scratch->sector_idx;
      char *sector_scratch = query_scratch->sector_scratch;
      float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
      // query <-> neighbor list
      float *dist_scratch = query_scratch->aligned_dist_scratch;
      _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;
      // visited map/set
      tsl::robin_set<_u64> &visited = *(query_scratch->visited);
      tsl::robin_map<_u64, bool>& exact_visited = *(query_scratch->exact_visited);

      // pre-allocated data field for searching
      std::vector<const char*> vis_cand;
      vis_cand.reserve(20);   // at most researve 12 is enough
      // this is a ring queue for storing sector buffers ptr.
      std::vector<char*> sector_buffers(MAX_N_SECTOR_READS);
      // pre-allocated buffer
      std::vector<char*> tmp_bufs(beam_width * 2);
      std::vector<int> read_fids(beam_width * 2);
      std::vector<unsigned> nbr_buf(max_degree);
      while(true) {
        int i = cur_task++;
        if (i >= query_num) {
          break;
        }
        Timer query_timer, tmp_timer, part_timer;

        // get the current query pointers
        const T* query1 = query_ptr + (i * query_aligned_dim);
        _u64* indices = indices_vec.data() + (i * k_search);
        float* distances = distances_vec.data() + (i * k_search);
        QueryStats* stats = stats_ptr + i;

        _mm_prefetch((char *) query1, _MM_HINT_T1);
        // copy query to thread specific aligned and allocated memory (for distance
        // calculations we need aligned data)
        float query_norm = 0;
        for (uint32_t i = 0; i < query_dim; i++) {
          scratch.aligned_query_float[i] = query1[i];
          scratch.aligned_query_T[i] = query1[i];
          query_norm += query1[i] * query1[i];
        }
        // if inner product, we also normalize the query and set the last coordinate
        // to 0 (this is the extra coordindate used to convert MIPS to L2 search)
        if (metric == diskann::Metric::INNER_PRODUCT) {
          query_norm = std::sqrt(query_norm);
          scratch.aligned_query_T[this->data_dim - 1] = 0;
          scratch.aligned_query_float[this->data_dim - 1] = 0;
          for (uint32_t i = 0; i < this->data_dim - 1; i++) {
            scratch.aligned_query_T[i] /= query_norm;
            scratch.aligned_query_float[i] /= query_norm;
          }
        }
        // reset query
        query_scratch->reset();

        // query <-> PQ chunk centers distances
        pq_table.populate_chunk_distances(query_float, pq_dists);

        std::vector<Neighbor> retset(l_search + 1);
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
          tmp_timer.reset();
          float cur_expanded_dist = dist_cmp->compare(query, (T*)node_buf,
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
              nbr_buf[nbors_cand_size++] = node_nbrs[m];
            }
          }
          if (stats != nullptr) {
            stats->insert_visited_us += (double) tmp_timer.elapsed();
            stats->insert_visited += nbors_cand_size;
            stats->check_visited += nnbrs;
          }
          if (nbors_cand_size) {
            tmp_timer.reset();
            _mm_prefetch((char *) nbr_buf.data(), _MM_HINT_T1);
            compute_pq_dists(nbr_buf.data(), nbors_cand_size, dist_scratch);
            if (stats != nullptr) {
              stats->n_cmps += (double) nbors_cand_size;
              stats->cmp_us += (double) tmp_timer.elapsed();
            }
            for (unsigned m = 0; m < nbors_cand_size; ++m) {
              const int nbor_id = nbr_buf[m];
              const float nbor_dist = dist_scratch[m];
              if (nbor_dist >= retset[cur_list_size - 1].distance &&
                  (cur_list_size == l_search)) {
                continue;
              }
              Neighbor nn(nbor_id, nbor_dist, true);
              // Return position in sorted list where nn inserted
              // TODO (IO): notify this link
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
            visited.insert(node_ids[i]);
          }
        };

        tmp_timer.reset();
        if (mem_L) {
          std::vector<unsigned> mem_tags(mem_L);
          std::vector<float> mem_dists(mem_L);
          std::vector<T*> res = std::vector<T*>();
          mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data(), nullptr, res);
          compute_and_add_to_retset(mem_tags.data(), std::min((unsigned)mem_L, (unsigned)l_search));
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
        std::vector<FrontierData> frontier;
        frontier.reserve(2 * beam_width);
        tsl::robin_map<char*, _u32> frontier_nhoods;
        std::vector<AlignedRead> frontier_read_reqs;
        frontier_read_reqs.reserve(2 * beam_width);

        // these data are count seperately
        _u32 n_io_in_q = 0; // how many io left
        _u32 n_proc_in_q = 0; // how many proc left

        // these three data are in beam level
        unsigned nk = cur_list_size;
        // bottom and top for the ring queue.
        _u32 top_bufs_idx = 0;
        _u32 botm_bufs_idx = 0;
        while (k < cur_list_size && num_ios < io_limit) {

          if (n_io_in_q > 0) {
            unsigned min_r = 0;
            if (n_proc_in_q == 0) min_r = 1;
            part_timer.reset();
            int n_read_blks = io_manager->get_events(ctx, min_r, n_io_in_q, tmp_bufs);
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
              std::cout << "read error: " << int((sector_buf - sector_scratch) / SECTOR_LEN) \
                        << " " << std::this_thread::get_id() << " " \
                        << n_io_in_q << " " << n_proc_in_q << std::endl;
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
              if (id != exact_id) {
                if (dist_scratch[j] >= retset[cur_list_size - 1].distance * pq_filter_ratio
                  || exact_visited[id]) {
                    continue;
                } else {
                  // TODO (IO): notify this link (discovered)
                  // replace only the other nodes
                }
              }
              char *node_buf = sector_buf + j * max_node_len;
              _mm_prefetch((char *) node_buf, _MM_HINT_T0);
              compute_exact_dists_and_push(node_buf, id);
              _mm_prefetch((char *) OFFSET_TO_NODE_NHOOD(node_buf), _MM_HINT_T0);
              vis_cand[cand_size++] = node_buf;
            }
            for (unsigned j = 0; j < cand_size; ++j) {
              compute_and_push_nbrs(vis_cand[j], nk);
            }
            if (stats != nullptr) stats->disk_proc_us += (double) part_timer.elapsed();

            frontier_nhoods.erase(sector_buf);
            n_proc_in_q--;
          }

          if (n_io_in_q == 0 && n_proc_in_q < beam_width) {
            // clear iteration state
            frontier.clear();
            frontier_read_reqs.clear();
            read_fids.clear();
            // find new beam
            if (nk < k) k = nk;
            _u32 marker = k;
            _u32 num_seen = 0;

            // distribute cache and disk-read nodes
            part_timer.reset();
            while (marker < cur_list_size && num_seen < beam_width) {
              if (retset[marker].flag) {
                if (exact_visited.find(retset[marker].id) == exact_visited.end()) {
                  num_seen++;
                  // use the cached page.
                  if (id2cache_page_.find(retset[marker].id) != id2cache_page_.end()) {
                    const unsigned pid = id2cache_page_[retset[marker].id];
                    for (unsigned j = 0; j < cache_layout_[pid].size(); ++j) {
                      unsigned id = cache_layout_[pid][j];
                      if (exact_visited.find(id) == exact_visited.end())
                        exact_visited.insert({id, false});
                      else if (stats != nullptr) {
                        stats->repeat_read++;
                      }
                    }
                    frontier.emplace_back(retset[marker].id, pid, cache_fid);
                  }
                  else {
                    const unsigned pid = id2page_[retset[marker].id];
                    for (unsigned j = 0; j < gp_layout_[pid].size(); ++j) {
                      unsigned id = gp_layout_[pid][j];
                      if (exact_visited.find(id) == exact_visited.end())
                        exact_visited.insert({id, false});
                      else if (stats != nullptr) {
                        stats->repeat_read++;
                      }
                    }
                    frontier.emplace_back(retset[marker].id, pid, disk_fid);
                  }
                }
                retset[marker].flag = false;
              }
              marker++;
            }
            if (stats != nullptr) stats->dispatch_us += (double) part_timer.elapsed();

            // read nhoods of frontier ids
            // TODO (IO): let the IO manager to do it
            if (!frontier.empty()) {
              part_timer.reset();
              if (stats != nullptr) stats->n_hops++;
              n_io_in_q += frontier.size();
              for (_u64 i = 0; i < frontier.size(); i++) {
                auto sector_buf = sector_scratch + sector_scratch_idx * SECTOR_LEN;
                sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
                auto offset = (static_cast<_u64>(frontier[i].pid)) * SECTOR_LEN;
                if (frontier[i].fid == disk_fid) {
                  offset += SECTOR_LEN;
                }
                frontier_nhoods.insert({sector_buf, frontier[i].id});
                frontier_read_reqs.push_back(AlignedRead(offset, SECTOR_LEN, sector_buf));
                read_fids.push_back(frontier[i].fid);
                if (stats != nullptr) {
                  stats->n_4k++;
                  stats->n_ios++;
                }
                num_ios++;
              }
              io_manager->submit_reqs(frontier_read_reqs, read_fids, ctx);
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
          if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
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

        if (stats != nullptr) {
          stats->total_us = (double) query_timer.elapsed();
          stats->postprocess_us = (double) tmp_timer.elapsed();
        }
      }
    });
  }

  template class IndexEngine<_u8>;
  template class IndexEngine<_s8>;
  template class IndexEngine<float>;
} // namespace diskann
