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

    if (n_io_nthreads > 0) {
      start_io_threads();
    }
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
      tsl::robin_set<_u32> &visited = *(query_scratch->visited);
      tsl::robin_map<_u32, bool>& exact_visited = *(query_scratch->exact_visited);
      tsl::robin_map<_u32, std::shared_ptr<FrontierNode>> id2ftr; // id 2 frontier
      id2ftr.reserve(1024);

      // pre-allocated data field for searching
      std::vector<std::pair<_u32, const char*>> vis_cand;
      vis_cand.reserve(20);   // at most researve 12 is enough
      // this is a ring queue for storing sector buffers ptr.
      // when read done, push a sector_buf to here, wait for execute
      std::vector<char*> sector_buffers(MAX_N_SECTOR_READS);
      // pre-allocated buffer, will clear up each iter/step.
      std::vector<char*> tmp_bufs(beam_width * 2);
      std::vector<int> read_fids(beam_width * 2);
      std::vector<AlignedRead> frontier_read_reqs(beam_width * 2);
      std::vector<unsigned> nbr_buf(max_degree);
      while(true) {
        size_t i = cur_task++;
        if (i >= query_num) {
          break;
        }
        Timer query_timer, tmp_timer, part_timer;

        // record the frontier node, also used to record search path.
        std::vector<std::shared_ptr<FrontierNode>> frontier;
        size_t ftr_id = 0; // frontier id

        // get the current query pointers
        const T* query1 = query_ptr + (i * query_aligned_dim);
        _u64* indices = indices_vec.data() + (i * k_search);
        float* distances = distances_vec.data() + (i * k_search);
        QueryStats* stats = stats_ptr + i;

        _mm_prefetch((char *) query1, _MM_HINT_T1);
        // copy query to thread specific aligned and allocated memory (for distance
        // calculations we need aligned data)
        float query_norm = 0;
        for (_u32 i = 0; i < query_dim; i++) {
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
          for (_u32 i = 0; i < this->data_dim - 1; i++) {
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

        auto compute_and_push_nbrs = [&](const char *node_buf, const unsigned src_id, unsigned& nk) {
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
              // IO: notify the src node
              Neighbor nn(nbor_id, nbor_dist, true, src_id);
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

        // map unfinished sector_buf to the frontier node.
        tsl::robin_map<char*, std::shared_ptr<FrontierNode>> sec_buf2ftr;

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
            if (sec_buf2ftr.find(sector_buf) == sec_buf2ftr.end()) {
              std::cout << "read error: " << int((sector_buf - sector_scratch) / SECTOR_LEN) \
                        << " " << std::this_thread::get_id() << " " \
                        << n_io_in_q << " " << n_proc_in_q << std::endl;
              exit(-1);
            }

            auto fn = sec_buf2ftr[sector_buf];
            const _u32 exact_id = fn->id;
            const int fid = fn->fid;
            const int pid = fn->pid;
            unsigned id, p_size;
            unsigned* node_in_page;
            if (fid == disk_fid) {
              p_size = gp_layout_[pid].size();
              node_in_page = gp_layout_[pid].data();
            } else if (fid == cache_fid) {
              p_size = cache_layout_[pid].size();
              node_in_page = cache_layout_[pid].data();
            } else {
              std::cerr << "wrong fid: " << fid << std::endl;
              exit(0);
            }
            compute_pq_dists(node_in_page, p_size, dist_scratch);

            unsigned cand_size = 0;
            for (unsigned j = 0; j < p_size; ++j) {
              if (fid == disk_fid) {
                id = gp_layout_[pid][j];
              } else if (fid == cache_fid) {
                id = cache_layout_[pid][j];
              }
              if (id != exact_id) {
                if (dist_scratch[j] >= retset[cur_list_size - 1].distance * pq_filter_ratio
                  || exact_visited[id]) {
                    continue;
                } else {
                  // IO: notify this link (discovered from block)
                  // replace only the other nodes
                  auto fn = std::make_shared<FrontierNode>(id, pid, fid);
                  fn->sector_buf = sector_buf;
                  fn->node_buf = sector_buf + j * max_node_len;
                  // update the src id's in_blk (neighbors).
                  id2ftr[exact_id]->in_blk_.push_back(fn);
                  id2ftr.insert({id, fn});
                  frontier.push_back(fn);
                  ftr_id++;
                }
              } else {
                auto fn = id2ftr[id];
                fn->node_buf = sector_buf + j * max_node_len;
              }

              char *node_buf = sector_buf + j * max_node_len;
              _mm_prefetch((char *) node_buf, _MM_HINT_T0);
              compute_exact_dists_and_push(node_buf, id);
              _mm_prefetch((char *) OFFSET_TO_NODE_NHOOD(node_buf), _MM_HINT_T0);
              vis_cand[cand_size++] = std::make_pair(id, node_buf);
            }
            for (unsigned j = 0; j < cand_size; ++j) {
              compute_and_push_nbrs(vis_cand[j].second, vis_cand[j].first, nk);
            }
            if (stats != nullptr) stats->disk_proc_us += (double) part_timer.elapsed();

            sec_buf2ftr.erase(sector_buf);
            n_proc_in_q--;
          }

          if (n_io_in_q == 0 && n_proc_in_q < beam_width) {
            // clear iteration state
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
                  unsigned pid;
                  int fid;
                  // use the cached page.
                  if (id2cache_page_.find(retset[marker].id) != id2cache_page_.end()) {
                    pid = id2cache_page_[retset[marker].id];
                    fid = cache_fid;
                    for (unsigned j = 0; j < cache_layout_[pid].size(); ++j) {
                      unsigned id = cache_layout_[pid][j];
                      if (exact_visited.find(id) == exact_visited.end())
                        exact_visited.insert({id, false});
                      else if (stats != nullptr) {
                        stats->repeat_read++;
                      }
                    }
                  }
                  else {
                    pid = id2page_[retset[marker].id];
                    fid = disk_fid;
                    for (unsigned j = 0; j < gp_layout_[pid].size(); ++j) {
                      unsigned id = gp_layout_[pid][j];
                      if (exact_visited.find(id) == exact_visited.end())
                        exact_visited.insert({id, false});
                      else if (stats != nullptr) {
                        stats->repeat_read++;
                      }
                    }
                  }
                  auto fn = std::make_shared<FrontierNode>(retset[marker].id, pid, fid);
                  frontier.push_back(fn);
                  // update id2ftr: from last node's neighbors.
                  if (id2ftr.find(retset[marker].rev_id) != id2ftr.end()) {
                    id2ftr[retset[marker].rev_id]->nb_.push_back(fn);
                  }
                  id2ftr.insert({retset[marker].id, fn});
                }
                retset[marker].flag = false;
              }
              marker++;
            }
            if (stats != nullptr) stats->dispatch_us += (double) part_timer.elapsed();

            // read nhoods of frontier ids
            // TODO (IO): let the IO manager to do it (maybe don't need do it first)?
            if (ftr_id < frontier.size()) {
              part_timer.reset();
              if (stats != nullptr) stats->n_hops++;
              n_io_in_q += frontier.size() - ftr_id;
              while(ftr_id < frontier.size()) {
                auto sector_buf = sector_scratch + sector_scratch_idx * SECTOR_LEN;
                sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
                auto offset = (static_cast<_u64>(frontier[ftr_id]->pid)) * SECTOR_LEN;
                if (frontier[ftr_id]->fid == disk_fid) {
                  offset += SECTOR_LEN; // one page for metadata
                }
                sec_buf2ftr.insert({sector_buf, frontier[ftr_id]});
                frontier_read_reqs.push_back(AlignedRead(offset, SECTOR_LEN, sector_buf));
                read_fids.push_back(frontier[ftr_id]->fid);
                // update sector_buf for the current node.
                id2ftr[frontier[ftr_id]->id]->sector_buf = sector_buf;
                if (stats != nullptr) {
                  stats->n_4k++;
                  stats->n_ios++;
                }
                num_ios++;
                ftr_id++;
              }
              io_manager->submit_read_reqs(frontier_read_reqs, read_fids, ctx);
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
        // better clear here.
        id2ftr.clear();
        if (n_io_nthreads > 0) {
          path_queue_.push(frontier);
        } else {
          frontier.clear();
        }
      }
    });
    if (n_io_nthreads > 0) {
      stop_io_threads();
    }
  }

  template<typename T>
  void IndexEngine<T>::start_io_threads() {
    freq_->move();
    io_stop_.store(false);
    io_pool->runTaskAsync([&, this](int tid) {
      // IO context init.
      IOContext& ctx = w_ctxs[tid];
      char* write_sector_scratch = disk_write_buffer[tid];
      std::vector<std::shared_ptr<FrontierNode>> nodes;
      double save_ratio = 0.1;

      while (true) {
        if (path_queue_.try_pop(nodes)) {
          for (size_t i = 0; i < nodes.size(); i++) {
            freq_->add(nodes[i]->id);
          }
          // re-sort by neighbor used
          std::sort(nodes.begin(), nodes.end(),
            [&](const std::shared_ptr<FrontierNode> left, const std::shared_ptr<FrontierNode> right) {
            if (left->nb_.size() != right->nb_.size()) {
              return left->nb_.size() > right->nb_.size();
            }
            return freq_->get(left->id) > freq_->get(right->id);
          });

          int save_num = std::min((int)((double)nodes.size() * save_ratio), MAX_N_SECTOR_WRITE);
          // write offsets and buffer address, file id.
          std::vector<AlignedWrite> write_reqs;
          std::vector<int> write_fids;
          // ID to pid mapping for new layouts, and new layout to append.
          std::vector<std::pair<_u32, _u32>> new_id2pids;
          std::vector<std::vector<_u32>> new_layouts;
          int saved_cnt = 0;
          _u64 marker = 0;
          while (saved_cnt < save_num && marker < nodes.size()) {
            // the id and nodebuf for the new layout.
            std::vector<unsigned> new_layout;
            std::vector<char*> layout_nodebufs;
            _u32 pid = nodes[marker]->pid;
            new_layout.push_back(nodes[marker]->id);
            layout_nodebufs.push_back(nodes[marker]->node_buf);
            // nodes can be kicked out.
            std::vector<unsigned> node_kick_out;
            std::vector<char*> kick_out_nodebuf;
            if (nodes[marker]->fid == disk_fid) {
              // record the neighbors of the node.
              tsl::robin_set<unsigned> nb_set;
              unsigned *node_nbrs = OFFSET_TO_NODE_NHOOD(nodes[marker]->node_buf);
              unsigned nnbrs = *(node_nbrs++);
              for (unsigned m = 0; m < nnbrs; ++m) {
                nb_set.insert(node_nbrs[m]);
              }
              for (unsigned m = 0; m < nodes[marker]->in_blk_.size(); ++m) {
                nb_set.insert(nodes[marker]->in_blk_[m]->id);
              }
              // check whether there are neighbors or in block, if so continue.
              for (_u64 j = 0; j < gp_layout_[pid].size(); j++) {
                if (gp_layout_[pid][j] == nodes[marker]->id) {
                  continue;
                } else if (nb_set.find(gp_layout_[pid][j]) != nb_set.end()) {
                  // find a neighbor in gp block.
                  // TODO: how to replace the original nodes?
                  new_layout.push_back(gp_layout_[pid][j]);
                  layout_nodebufs.push_back(nodes[marker]->sector_buf + j * max_node_len);
                } else {
                  node_kick_out.push_back(gp_layout_[pid][j]);
                  kick_out_nodebuf.push_back(nodes[marker]->sector_buf + j * max_node_len);
                }
              }
              // the block already filled with neighbors, don't consider it now.
              if (new_layout.size() == this->nnodes_per_sector) {
                marker++;
                continue;
              }
            } else {
              // record the neighbors of the node.
              tsl::robin_set<unsigned> nb_set;
              unsigned *node_nbrs = OFFSET_TO_NODE_NHOOD(nodes[marker]->node_buf);
              unsigned nnbrs = *(node_nbrs++);
              for (unsigned m = 0; m < nnbrs; ++m) {
                nb_set.insert(node_nbrs[m]);
              }
              for (unsigned m = 0; m < nodes[marker]->in_blk_.size(); ++m) {
                nb_set.insert(nodes[marker]->in_blk_[m]->id);
              }
              // I think the first node should be nid, so here we can just start from 1?
              for (_u64 j = 1; j < cache_layout_[pid].size(); j++) {
                if (nb_set.find(cache_layout_[pid][j]) != nb_set.end()) {
                  // find a neighbor in gp block.
                  // TODO: how to replace the original nodes?
                  new_layout.push_back(cache_layout_[pid][j]);
                  layout_nodebufs.push_back(nodes[marker]->sector_buf + j * max_node_len);
                } else {
                  node_kick_out.push_back(cache_layout_[pid][j]);
                  kick_out_nodebuf.push_back(nodes[marker]->sector_buf + j * max_node_len);
                }
              }
              // the block already filled with neighbors, don't consider it now.
              if (new_layout.size() == this->nnodes_per_sector) {
                marker++;
                continue;
              }
            }
            // to insert the newly discovered neighbors.
            for (_u64 j = 0; j < nodes[marker]->nb_.size(); j++) {
              if (new_layout.size() < this->nnodes_per_sector) {
                auto nb = nodes[marker]->nb_[j];
                if (nb->node_buf == nullptr) {  // don't know whether it will trigger
                  std::cout << "error node buf!" << std::endl;
                  exit(0);
                }
                new_layout.push_back(nb->id);
                layout_nodebufs.push_back(nb->node_buf);
              } else {
                // TODO: how to replace?
                break;
              }
            }
            // fill the block with kicked out nodes in original block.
            for (_u64 j = 0; j < node_kick_out.size(); j++) {
              if (new_layout.size() < this->nnodes_per_sector) {
                new_layout.push_back(node_kick_out[j]);
                layout_nodebufs.push_back(kick_out_nodebuf[j]);
              } else {
                break;
              }
            }
            // copy data to write buffer
            char* write_buf = write_sector_scratch + saved_cnt * SECTOR_LEN;
            memset(write_buf, 0, SECTOR_LEN);
            for (_u64 w_idx = 0; w_idx < new_layout.size(); w_idx++) {
              memcpy(write_buf + w_idx * max_node_len,
                     layout_nodebufs[w_idx], max_node_len);
            }
            // write the buffer data to SSD
            // Here is the case cache not full.
            if (cur_page_id.load() < tot_cache_page) {
              int w_cache_pid = cur_page_id++;
              // in case concurrent error occurs, since the operation is not atomic.
              if (w_cache_pid >= tot_cache_page) {
                // TODO: do something, disk cache is full.
              }
              auto offset = (static_cast<_u64>(w_cache_pid)) * SECTOR_LEN;
              write_reqs.push_back(AlignedWrite(offset, SECTOR_LEN, write_buf));
              write_fids.push_back(cache_fid);
              // for update cache data.
              new_id2pids.emplace_back(nodes[marker]->id, w_cache_pid);
              new_layouts.push_back(new_layout);
              saved_cnt++;
            } else {
              // TODO: disk cache is full.
              // std::cout << "disk cache full" << std::endl;
            }
            marker++;
          }
          // submit write req and get result.
          int n_ops = io_manager->submit_write_reqs(write_reqs, write_fids, ctx);
          io_manager->get_events(ctx, n_ops);
          // TODO: update cache layout and id2cachepage. Using lock.
          // TODO (Question): how to fix consistency problem?
          std::unique_lock<std::mutex> lk(cache_upt_lock);
          // std::cout << "write page: " << new_id2pids.size() << std::endl;
          for (size_t i = 0; i < new_id2pids.size(); i++) {
            auto nid = new_id2pids[i].first;
            auto cache_pid = new_id2pids[i].second;
            if (id2cache_page_.find(nid) != id2cache_page_.end()) {
              // TODO: write to a cache file, existance.
              if (cache_pid == cache_layout_.size()) {  // append one
                cache_layout_.push_back(new_layouts[i]);
                id2cache_page_[nid] = cache_pid;
              } else {
                // TODO
              }
            } else {
              if (cache_pid == cache_layout_.size()) {  // append one
                cache_layout_.push_back(new_layouts[i]);
                id2cache_page_.insert({nid, cache_pid});
              } else {
                // TODO: write to a existance page, not support now.
                // std::cout << "cache pid not filled" << std::endl;
              }
            }
          }
          lk.unlock();

          // release memory space.
          nodes.clear();
        } else {
          std::this_thread::yield();
        }
        // stop the thread pool
        if (io_stop_.load()) {
          break;
        }
      }
    });
  }

  template<typename T>
  void IndexEngine<T>::stop_io_threads() {
    io_stop_.store(true);
    io_pool->endTaskAsync();
  }

  template class IndexEngine<_u8>;
  template class IndexEngine<_s8>;
  template class IndexEngine<float>;
} // namespace diskann
