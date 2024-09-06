#include "pq_flash_index.h"

namespace diskann {
  template<typename T>

  void PQFlashIndex<T>::generate_node_distance_to_mediod(
    const std::string& freq_save_path,
    const _u32 mem_L) {
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }
    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    float        query_norm = 0;
    const T *    query = data.scratch.aligned_query_T;
    const float *query_float = data.scratch.aligned_query_float;

    uint32_t query_dim = metric == diskann::Metric::INNER_PRODUCT ? this-> data_dim - 1: this-> data_dim;

    _u32         best_medoid = medoids[0];
    {
        auto  global_cache_iter = coord_cache.find(best_medoid);
        T *   node_fp_coords_copy = global_cache_iter->second;
        for (uint32_t i = 0; i < query_dim; i++) {
            data.scratch.aligned_query_float[i] = node_fp_coords_copy[i];
            data.scratch.aligned_query_T[i] = node_fp_coords_copy[i];
            query_norm += node_fp_coords_copy[i] * node_fp_coords_copy[i];
        }
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
    {
        auto  global_cache_iter = coord_cache.find(best_medoid);
        T *   node_fp_coords_copy = global_cache_iter->second;
        float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy,
                                                (unsigned) aligned_dim);
        diskann::cout << "medoid distance to medoid: " << cur_expanded_dist << std::endl;
    }

    auto       query_scratch = &(data.scratch);
    // reset query
    query_scratch->reset();
    // pointers to buffers for data
    T *   data_buf = query_scratch->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    if (mem_L) {
    //   std::vector<unsigned> mem_tags(mem_L);
    //   std::vector<float> mem_dists(mem_L);
    //   std::vector<T*> res = std::vector<T*>();
    //   unsigned start_point = mem_index_->_start;
    //   std::vector<int> distance_to_mediod(mem_index_->_nd, 2e9);
    //   std::vector<bool> vis;
    //   std::vector<unsigned> bfs_queue;

    //   int cur_idx = 0;
    //   bfs_queue.push_back(start_point);
    //   distance_to_mediod[start_point] = 0;
    //   while (cur_idx < bfs_queue.size()) {
    //     unsigned cur_node = bfs_queue[cur_idx++];
    //     vis[cur_node] = true;
    //     for (unsigned &nb : mem_index_->full_graph[cur_node]) {
    //         if (vis[nb]) continue;
    //         distance_to_mediod[nb] = std::min(distance_to_mediod[nb], distance_to_mediod[cur_node] + 1);
    //         bfs_queue.push_back(nb);
    //     }
    //   }
    }

    // std::vector<int> distance_to_mediod;
    // std::vector<bool> vis;
    // distance_to_mediod.resize(num_points);
    // vis.resize(num_points);
    // for (int i = 0; i < num_points; i++) {
    //     distance_to_mediod[i] = 2e9;
    //     vis[i] = false;
    // }
    // std::vector<unsigned> bfs_queue;

    // int cur_idx = 0;
    // bfs_queue.push_back(best_medoid);
    // distance_to_mediod[best_medoid] = 0;
    // while (cur_idx < bfs_queue.size()) {
    //     unsigned cur_node = bfs_queue[cur_idx++];
    //     vis[cur_node] = true;

    //     auto iter = nhood_cache.find(cur_node);
    //     if (iter != nhood_cache.end()) {
    //         _u64 nnr = iter->second.first;
    //         unsigned* node_nbrs = iter->second.second;
    //         for (int m = 0; m < nnr; m++) {
    //             unsigned nb = node_nbrs[m];
    //             if (vis[nb]) continue;
    //             distance_to_mediod[nb] = distance_to_mediod[cur_node] + 1;
    //             bfs_queue.push_back(nb);
    //         }
    //     }
    // }
    std::vector<std::vector<std::pair<_u64, unsigned>>> in_nbrs;
    in_nbrs.resize(num_points);
    for (unsigned i = 0; i < num_points; i++) {
        auto iter = nhood_cache.find(i);
        if (iter != nhood_cache.end()) {
            _u64 nnr = iter->second.first;
            unsigned* node_nbrs = iter->second.second;
            for (int m = 0; m < nnr; m++) {
                unsigned nb = node_nbrs[m];
                in_nbrs[nb].push_back(std::make_pair(nnr, i));
            }
        }
    }
    std::vector<float> scores;
    scores.resize(num_points);
    for (unsigned i = 0; i < num_points; i++) {
        scores[i] = 0;
    }
    scores[best_medoid] = 100.0;
    for (int round = 0; round < 16; round++) {
        std::vector<float> nxt_score;
        nxt_score.resize(num_points);
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned i = 0; i < num_points; i++) {
            float tmp = 0;
            for (unsigned j = 0; j < in_nbrs[i].size(); j++) {
                auto nnr = in_nbrs[i][j].first;
                auto src_i = in_nbrs[i][j].second;
                tmp += 1.0 * scores[src_i] / nnr;
            }
            nxt_score[i] = tmp;
        }
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned i = 0; i < num_points; i++) {
            scores[i] = nxt_score[i] * 0.85 + scores[i] * 0.15;
            if (i == 8115972) {
                scores[i] += 100;
                std::cout << "round: " << i << ", medoid score:" << scores[i] << std::endl;
            }
        }
    }

    // std::vector<float> exact_dists;
    // exact_dists.resize(num_points);

    // auto compute_exact_dists = [&](const char* node_buf, const unsigned id) -> float {
    //   T *node_fp_coords_copy = data_buf;
    //   memcpy(node_fp_coords_copy, node_buf, disk_bytes_per_point);
    //   float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy,
    //                                         (unsigned) aligned_dim);
    //   return cur_expanded_dist;
    // };

    // for (int i = 0; i < num_points; i++) {
    //     auto  global_cache_iter = coord_cache.find(i);
    //     T *   node_fp_coords_copy = global_cache_iter->second;
    //     char node_buf[max_node_len];
    //     memcpy(node_buf, node_fp_coords_copy, disk_bytes_per_point);
    //     exact_dists[i] = compute_exact_dists(node_buf, i);
    // }

    // save exact dist file
    const std::string freq_file = freq_save_path + "_nhops.bin";
    std::ofstream writer(freq_file, std::ios::binary | std::ios::out);
    diskann::cout << "Writing exact distance to medoid: " << freq_file << std::endl;
    writer.write((char *)&num_points, sizeof(unsigned));

    for (size_t i = 0; i < num_points; ++i) {
      writer.write((char *)(&(scores[i])), sizeof(float));
    }
    diskann::cout << "Writing _exact_dist file finished" << std::endl;

  }

  // instantiations
  template class PQFlashIndex<_u8>;
  template class PQFlashIndex<_s8>;
  template class PQFlashIndex<float>;
} // namespace diskann