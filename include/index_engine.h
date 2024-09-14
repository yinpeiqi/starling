// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#include "file_io_manager.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq_table.h"
#include "utils.h"
#include "windows_customizations.h"
#include "index.h"
#include "pq_flash_index_utils.h"
#include "thread_pool.h"
#include "freq_list.h"

#include <tbb/concurrent_queue.h>

#define MAX_GRAPH_DEGREE 512
#define MAX_N_CMPS 16384
#define SECTOR_LEN (_u64) 4096
#define MAX_N_SECTOR_READS 128
#define MAX_PQ_CHUNKS 256

#define FULL_PRECISION_REORDER_MULTIPLIER 3

namespace diskann {
  template<typename T>
  struct DataScratch {
    char *sector_scratch =
        nullptr;          // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
    _u64 sector_idx = 0;  // index of next [SECTOR_LEN] scratch to use

    float *aligned_pqtable_dist_scratch =
        nullptr;  // MUST BE AT LEAST [256 * NCHUNKS]
    float *aligned_dist_scratch =
        nullptr;  // MUST BE AT LEAST diskann MAX_DEGREE
    _u8 *aligned_pq_coord_scratch =
        nullptr;  // MUST BE AT LEAST  [N_CHUNKS * MAX_DEGREE]
    T *    aligned_query_T = nullptr;
    float *aligned_query_float = nullptr;

    tsl::robin_set<_u32> *visited = nullptr;
    tsl::robin_set<_u32> *page_visited = nullptr;
    // false means in queue, true means executed
    tsl::robin_map<_u32, bool>* exact_visited = nullptr;

    void reset() {
      sector_idx = 0;
      visited->clear();  // does not deallocate memory.
      page_visited->clear();
      exact_visited->clear();
    }
  };

  // Node recorded in the search path.
  struct FrontierNode {
    unsigned id;
    unsigned pid;
    int fid;
    char* sector_buf;
    // Pointer to the search path node in the same block.
    // These nodes are execute directly, such that can init the struct here.
    // If a node is not a target, then this field will be empty.
    // We need shared_ptr here, otherwise memory leak will occur since
    // ptr are not correctly release.
    std::vector<std::shared_ptr<FrontierNode>> in_blk_;
    // Neighbors discovered, only executed are recorded.
    std::vector<std::shared_ptr<FrontierNode>> nb_;

    FrontierNode(unsigned id, unsigned pid, int fid) {
        this->id = id;
        this->pid = pid;
        this->fid = fid;
    }
  };

  template<typename T>
  class IndexEngine {
   public:
    DISKANN_DLLEXPORT IndexEngine(
        std::shared_ptr<FileIOManager> &fio_manager,
        diskann::Metric                     metric = diskann::Metric::L2);
    DISKANN_DLLEXPORT ~IndexEngine();

    // load id to page id and graph partition layout
    DISKANN_DLLEXPORT void load_partition_data(const std::string &index_prefix,
        const _u64 nnodes_per_sector = 0, const _u64 num_points = 0);

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load(diskann::MemoryMappedFiles &files,
                               uint32_t num_threads, const char *index_prefix);
#else
    // load compressed data, and obtains the handle to the disk-resident index
    DISKANN_DLLEXPORT int load(uint32_t num_threads, const char *index_prefix,
        const std::string& disk_index_path);
#endif

    DISKANN_DLLEXPORT void load_mem_index(Metric metric, const size_t query_dim,
        const std::string &mem_index_path, const _u32 num_threads,
        const _u32 mem_L);

    DISKANN_DLLEXPORT void page_search(
        const T *query, const _u64 query_num, const _u64 query_aligned_dim, const _u64 k_search, const _u32 mem_L,
        const _u64 l_search, std::vector<_u64>& indices_vec, std::vector<float>& distances_vec,
        const _u64 beam_width, const _u32 io_limit, const bool use_reorder_data = false,
        const float pq_filter_ratio = 1.2f, QueryStats *stats = nullptr);

    std::shared_ptr<FileIOManager> &io_manager;

    DISKANN_DLLEXPORT unsigned get_nnodes_per_sector() { return nnodes_per_sector; }

    void load_disk_cache_data(const std::string &index_prefix);
    void write_disk_cache_layout(const std::string &index_prefix);

   protected:
    DISKANN_DLLEXPORT void use_medoids_data_as_centroids();
    DISKANN_DLLEXPORT void setup_thread_data(_u64 nthreads, _u64 io_threads);
    DISKANN_DLLEXPORT void destroy_thread_data();
    DISKANN_DLLEXPORT void start_io_threads();
    DISKANN_DLLEXPORT void stop_io_threads();

   private:
    // index info
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // nnbrs of node `i`: *(unsigned*) (buf)
    // nbrs of node `i`: ((unsigned*)buf) + 1
    _u64 max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;

    // Data used for searching with re-order vectors
    _u64 ndims_reorder_vecs = 0, reorder_data_start_sector = 0,
         nvecs_per_sector = 0;

    diskann::Metric metric = diskann::Metric::L2;

    // used only for inner product search to re-scale the result value
    // (due to the pre-processing of base during index build)
    float max_base_norm = 0.0f;

    // data info
    _u64 num_points = 0;
    _u64 num_frozen_points = 0;
    _u64 frozen_location = 0;
    _u64 data_dim = 0;
    _u64 disk_data_dim = 0;  // will be different from data_dim only if we use
                             // PQ for disk data (very large dimensionality)
    _u64 aligned_dim = 0;
    _u64 disk_bytes_per_point = 0;

    std::string                        disk_index_file;

    // PQ data
    // n_chunks = # of chunks ndims is split into
    // data: _u8 * n_chunks
    // chunk_size = chunk size of each dimension chunk
    // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    _u8 *             data = nullptr;
    _u64              n_chunks;
    FixedChunkPQTable pq_table;

    // distance comparator
    std::shared_ptr<Distance<T>>     dist_cmp;
    std::shared_ptr<Distance<float>> dist_cmp_float;

    // for very large datasets: we use PQ even for the disk resident index
    bool              use_disk_index_pq = false;
    _u64              disk_pq_n_chunks = 0;
    FixedChunkPQTable disk_pq_table;

    // medoid/start info

    // graph has one entry point by default,
    // we can optionally have multiple starting points
    uint32_t *medoids = nullptr;
    // defaults to 1
    size_t num_medoids;
    // by default, it is empty. If there are multiple
    // centroids, we pick the medoid corresponding to the
    // closest centroid as the starting point of search
    float *centroid_data = nullptr;

    // nhood_cache
    unsigned *                                    nhood_cache_buf = nullptr;
    tsl::robin_map<_u32, std::pair<_u32, _u32 *>> nhood_cache;

    // coord_cache
    T *                       coord_cache_buf = nullptr;
    tsl::robin_map<_u32, T *> coord_cache;

    // thread-specific scratch
    std::vector<IOContext> ctxs;
    std::vector<DataScratch<T>> scratchs;
    // thread pool
    std::shared_ptr<ThreadPool> pool;
    std::shared_ptr<ThreadPool> io_pool;

    _u64                           max_nthreads;
    _u64                           n_io_nthreads;
    bool                           load_flag = false;
    bool                           reorder_data_exists = false;
    _u64                           reoreder_data_offset = 0;

    // in-memory navigation graph
    std::unique_ptr<Index<T, uint32_t>> mem_index_;

    // page search
    std::vector<unsigned> id2page_;
    std::vector<std::vector<unsigned>> gp_layout_;

    // disk cache
    int disk_fid;
    int cache_fid;
    tsl::robin_map<_u32, _u32> id2cache_page_;
    std::vector<std::vector<unsigned>> cache_layout_;
    std::shared_ptr<FreqWindowList> freq_;

    // IO thread pool
    tbb::concurrent_queue<std::vector<std::shared_ptr<FrontierNode>>> path_queue_;
    std::atomic_bool io_stop_;

#ifdef EXEC_ENV_OLS
    // Set to a larger value than the actual header to accommodate
    // any additions we make to the header. This is an outer limit
    // on how big the header can be.
    static const int HEADER_SIZE = SECTOR_LEN;
    char *           getHeaderBytes();
#endif
  };
}  // namespace diskann
