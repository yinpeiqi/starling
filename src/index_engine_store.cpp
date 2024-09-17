// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "logger.h"
#include "index_engine.h"
#include <malloc.h>
#include "percentile_stats.h"

#include <omp.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <thread>
#include "distance.h"
#include "exceptions.h"
#include "parameters.h"
#include "pq_flash_index_utils.h"
#include "timer.h"
#include "utils.h"

#include "cosine_similarity.h"
#include "tsl/robin_set.h"

namespace diskann {
  template<typename T>
  IndexEngine<T>::IndexEngine(std::shared_ptr<FileIOManager> &IOManager,
                                diskann::Metric m)
      : io_manager(IOManager), metric(m) {
    if (m == diskann::Metric::COSINE || m == diskann::Metric::INNER_PRODUCT) {
      if (std::is_floating_point<T>::value) {
        diskann::cout << "Cosine metric chosen for (normalized) float data."
                         "Changing distance to L2 to boost accuracy."
                      << std::endl;
        m = diskann::Metric::L2;
      } else {
        diskann::cerr << "WARNING: Cannot normalize integral data types."
                      << " This may result in erroneous results or poor recall."
                      << " Consider using L2 distance with integral data types."
                      << std::endl;
      }
    }

    this->dist_cmp.reset(diskann::get_distance_function<T>(m));
    this->dist_cmp_float.reset(diskann::get_distance_function<float>(m));
  }

  template<typename T>
  IndexEngine<T>::~IndexEngine() {
#ifndef EXEC_ENV_OLS
    if (data != nullptr) {
      delete[] data;
    }
#endif

    if (centroid_data != nullptr)
      aligned_free(centroid_data);
    // delete backing bufs for nhood and coord cache
    if (nhood_cache_buf != nullptr) {
      delete[] nhood_cache_buf;
      diskann::aligned_free(coord_cache_buf);
    }

    if (load_flag) {
      this->destroy_thread_data();
      io_manager->close();
    }
  }


  template<typename T>
  void IndexEngine<T>::setup_thread_data(_u64 nthreads) {
    pool = std::make_shared<ThreadPool>(nthreads);
    ctxs.resize(nthreads);
    scratchs.resize(nthreads);
    // parallel for
    pool->runTask([&, this](int tid) {
      this->io_manager->register_thread();
      ctxs[tid] = this->io_manager->get_ctx();
      // alloc space for the thread
      DataScratch<T> scratch;
      diskann::alloc_aligned((void **) &scratch.sector_scratch,
                              (_u64) MAX_N_SECTOR_READS * (_u64) SECTOR_LEN,
                              SECTOR_LEN);
      diskann::alloc_aligned(
          (void **) &scratch.aligned_pq_coord_scratch,
          (_u64) MAX_GRAPH_DEGREE * (_u64) MAX_PQ_CHUNKS * sizeof(_u8), 256);
      diskann::alloc_aligned((void **) &scratch.aligned_pqtable_dist_scratch,
                              256 * (_u64) MAX_PQ_CHUNKS * sizeof(float), 256);
      diskann::alloc_aligned((void **) &scratch.aligned_dist_scratch,
                              (_u64) MAX_GRAPH_DEGREE * sizeof(float), 256);
      diskann::alloc_aligned((void **) &scratch.aligned_query_T,
                              this->aligned_dim * sizeof(T), 8 * sizeof(T));
      diskann::alloc_aligned((void **) &scratch.aligned_query_float,
                              this->aligned_dim * sizeof(float),
                              8 * sizeof(float));
      scratch.visited = new tsl::robin_set<_u32>(4096);
      scratch.page_visited = new tsl::robin_set<_u32>(1024);
      scratch.exact_visited = new tsl::robin_map<_u32, bool>(1024);

      memset(scratch.aligned_query_T, 0, this->aligned_dim * sizeof(T));
      memset(scratch.aligned_query_float, 0,
              this->aligned_dim * sizeof(float));
      scratchs[tid] = scratch;
    });
    load_flag = true;
  }

  template<typename T>
  void IndexEngine<T>::destroy_thread_data() {
    diskann::cout << "Clearing scratch" << std::endl;
    assert(this->scratchs.size() == this->max_nthreads);
    for (_u64 tid = 0; tid < this->max_nthreads; tid++) {
      auto &scratch = scratchs[tid];
      diskann::aligned_free((void *) scratch.sector_scratch);
      diskann::aligned_free((void *) scratch.aligned_pq_coord_scratch);
      diskann::aligned_free((void *) scratch.aligned_pqtable_dist_scratch);
      diskann::aligned_free((void *) scratch.aligned_dist_scratch);
      diskann::aligned_free((void *) scratch.aligned_query_float);
      diskann::aligned_free((void *) scratch.aligned_query_T);

      delete scratch.visited;
      delete scratch.page_visited;
    }
    if (use_cache) {
      for (_u64 iotid = 0; iotid < this->n_io_nthreads; iotid++) {
        diskann::aligned_free((void *) disk_write_buffer[iotid]);
      }
    }
    this->io_manager->deregister_all_threads();
  }

  template<typename T>
  void IndexEngine<T>::use_medoids_data_as_centroids() {
    if (centroid_data != nullptr)
      aligned_free(centroid_data);
    alloc_aligned(((void **) &centroid_data),
                  num_medoids * aligned_dim * sizeof(float), 32);
    std::memset(centroid_data, 0, num_medoids * aligned_dim * sizeof(float));

    // borrow ctx
    IOContext &ctx = ctxs[0];
    diskann::cout << "Loading centroid data from medoids vector data of "
                  << num_medoids << " medoid(s)" << std::endl;
    for (uint64_t cur_m = 0; cur_m < num_medoids; cur_m++) {
      auto medoid = medoids[cur_m];
      // read medoid nhood
      char *medoid_buf = nullptr;
      alloc_aligned((void **) &medoid_buf, SECTOR_LEN, SECTOR_LEN);
      std::vector<AlignedRead> medoid_read(1);
      medoid_read[0].len = SECTOR_LEN;
      medoid_read[0].buf = medoid_buf;
      medoid_read[0].offset = NODE_SECTOR_NO(medoid) * SECTOR_LEN;
      io_manager->read(medoid_read, ctx);

      // all data about medoid
      char *medoid_node_buf = OFFSET_TO_NODE(medoid_buf, medoid);

      // add medoid coords to `coord_cache`
      T *medoid_coords = new T[data_dim];
      T *medoid_disk_coords = OFFSET_TO_NODE_COORDS(medoid_node_buf);
      memcpy(medoid_coords, medoid_disk_coords, disk_bytes_per_point);

      if (!use_disk_index_pq) {
        for (uint32_t i = 0; i < data_dim; i++)
          centroid_data[cur_m * aligned_dim + i] = medoid_coords[i];
      } else {
        disk_pq_table.inflate_vector((_u8 *) medoid_coords,
                                     (centroid_data + cur_m * aligned_dim));
      }

      aligned_free(medoid_buf);
      delete[] medoid_coords;
    }
  }

  template<typename T>
  void IndexEngine<T>::load_mem_index(Metric metric, const size_t query_dim, 
      const std::string& mem_index_path, const _u32 num_threads,
      const _u32 mem_L) {
      if (mem_index_path.empty()) {
        diskann::cerr << "mem_index_path is needed" << std::endl;
        exit(1);
      }
      mem_index_ = std::make_unique<diskann::Index<T, uint32_t>>(metric, query_dim, 0, false, true);
      mem_index_->load(mem_index_path.c_str(), num_threads, mem_L);
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  int IndexEngine<T>::load(MemoryMappedFiles &files, uint32_t num_threads,
                            const char *index_prefix) {
#else
  template<typename T>
  int IndexEngine<T>::load(uint32_t num_threads, const char *index_prefix,
                           const std::string& disk_index_path) {
#endif
    std::string pq_table_bin = std::string(index_prefix) + "_pq_pivots.bin";
    std::string pq_compressed_vectors =
        std::string(index_prefix) + "_pq_compressed.bin";
    std::string disk_index_file = disk_index_path; 
    std::string medoids_file = std::string(disk_index_file) + "_medoids.bin";
    std::string centroids_file =
        std::string(disk_index_file) + "_centroids.bin";

    size_t pq_file_dim, pq_file_num_centroids;
#ifdef EXEC_ENV_OLS
    get_bin_metadata(files, pq_table_bin, pq_file_num_centroids, pq_file_dim,
                     METADATA_SIZE);
#else
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim,
                     METADATA_SIZE);
#endif

    this->disk_index_file = disk_index_file;

    if (pq_file_num_centroids != 256) {
      diskann::cout << "Error. Number of PQ centroids is not 256. Exitting."
                    << std::endl;
      return -1;
    }

    this->data_dim = pq_file_dim;
    // will reset later if we use PQ on disk
    this->disk_data_dim = this->data_dim;
    // will change later if we use PQ on disk or if we are using
    // inner product without PQ
    this->disk_bytes_per_point = this->data_dim * sizeof(T);
    this->aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
#ifdef EXEC_ENV_OLS
    diskann::load_bin<_u8>(files, pq_compressed_vectors, this->data, npts_u64,
                           nchunks_u64);
#else
    diskann::load_bin<_u8>(pq_compressed_vectors, this->data, npts_u64,
                           nchunks_u64);
#endif

    this->num_points = npts_u64;
    this->n_chunks = nchunks_u64;

#ifdef EXEC_ENV_OLS
    pq_table.load_pq_centroid_bin(files, pq_table_bin.c_str(), nchunks_u64);
#else
    pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);
#endif

    diskann::cout
        << "Loaded PQ centroids and in-memory compressed vectors. #points: "
        << num_points << " #dim: " << data_dim
        << " #aligned_dim: " << aligned_dim << " #chunks: " << n_chunks
        << std::endl;

    if (n_chunks > MAX_PQ_CHUNKS) {
      std::stringstream stream;
      stream << "Error loading index. Ensure that max PQ bytes for in-memory "
                "PQ data does not exceed "
             << MAX_PQ_CHUNKS << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    std::string disk_pq_pivots_path = this->disk_index_file + "_pq_pivots.bin";
    if (file_exists(disk_pq_pivots_path)) {
      use_disk_index_pq = true;
#ifdef EXEC_ENV_OLS
      // giving 0 chunks to make the pq_table infer from the
      // chunk_offsets file the correct value
      disk_pq_table.load_pq_centroid_bin(files, disk_pq_pivots_path.c_str(), 0);
#else
      // giving 0 chunks to make the pq_table infer from the
      // chunk_offsets file the correct value
      disk_pq_table.load_pq_centroid_bin(disk_pq_pivots_path.c_str(), 0);
#endif
      disk_pq_n_chunks = disk_pq_table.get_num_chunks();
      disk_bytes_per_point =
          disk_pq_n_chunks *
          sizeof(_u8);  // revising disk_bytes_per_point since DISK PQ is used.
      std::cout << "Disk index uses PQ data compressed down to "
                << disk_pq_n_chunks << " bytes per point." << std::endl;
    }

// read index metadata
#ifdef EXEC_ENV_OLS
    // This is a bit tricky. We have to read the header from the
    // disk_index_file. But  this is now exclusively a preserve of the
    // DiskPriorityIO class. So, we need to estimate how many
    // bytes are needed to store the header and read in that many using our
    // 'standard' io_manager approach.
    io_manager->open(disk_index_file, O_DIRECT | O_RDONLY | O_LARGEFILE);
    this->max_nthreads = num_threads;
    this->setup_thread_data(num_threads);

    char *                   bytes = getHeaderBytes();
    ContentBuf               buf(bytes, HEADER_SIZE);
    std::basic_istream<char> index_metadata(&buf);
#else
    std::ifstream index_metadata(disk_index_file, std::ios::binary);
#endif
    _u32 nr, nc;  // metadata itself is stored as bin format (nr is number of
                  // metadata, nc should be 1)
    READ_U32(index_metadata, nr);
    READ_U32(index_metadata, nc);

    _u64 disk_nnodes;
    _u64 disk_ndims;  // can be disk PQ dim if disk_PQ is set to true
    READ_U64(index_metadata, disk_nnodes);
    READ_U64(index_metadata, disk_ndims);

    if (disk_nnodes != num_points) {
      diskann::cout << "Mismatch in #points for compressed data file and disk "
                       "index file: "
                    << disk_nnodes << " vs " << num_points << std::endl;
      return -1;
    }

    size_t medoid_id_on_file;
    READ_U64(index_metadata, medoid_id_on_file);
    READ_U64(index_metadata, max_node_len);
    READ_U64(index_metadata, nnodes_per_sector);
    max_degree = ((max_node_len - disk_bytes_per_point) / sizeof(unsigned)) - 1;

    std::cout << "max node len "<<max_node_len <<" disk bytes "<<disk_bytes_per_point << std::endl;
    if (max_degree > MAX_GRAPH_DEGREE) {
      std::stringstream stream;
      stream << "Error loading index. Ensure that max graph degree (R) does "
                "not exceed "
             << MAX_GRAPH_DEGREE << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    // setting up concept of frozen points in disk index for streaming-DiskANN
    READ_U64(index_metadata, this->num_frozen_points);
    _u64 file_frozen_id;
    READ_U64(index_metadata, file_frozen_id);
    if (this->num_frozen_points == 1)
      this->frozen_location = file_frozen_id;
    if (this->num_frozen_points == 1) {
      diskann::cout << " Detected frozen point in index at location "
                    << this->frozen_location
                    << ". Will not output it at search time." << std::endl;
    }

    READ_U64(index_metadata, this->reorder_data_exists);
    if (this->reorder_data_exists) {
      if (this->use_disk_index_pq == false) {
        throw ANNException(
            "Reordering is designed for used with disk PQ compression option",
            -1, __FUNCSIG__, __FILE__, __LINE__);
      }
      READ_U64(index_metadata, this->reorder_data_start_sector);
      READ_U64(index_metadata, this->ndims_reorder_vecs);
      READ_U64(index_metadata, this->nvecs_per_sector);
    }

    diskann::cout << "Disk-Index File Meta-data: ";
    diskann::cout << "# nodes per sector: " << nnodes_per_sector;
    diskann::cout << ", max node len (bytes): " << max_node_len;
    diskann::cout << ", max node degree: " << max_degree << std::endl;

#ifdef EXEC_ENV_OLS
    delete[] bytes;
#else
    index_metadata.close();
#endif

    // default use page search
    this->load_partition_data(index_prefix, nnodes_per_sector, num_points);

#ifndef EXEC_ENV_OLS
    // open FileIOManager handle to index_file
    std::string index_fname(disk_index_file);
    this->disk_fid = io_manager->open(index_fname, O_DIRECT | O_RDONLY | O_LARGEFILE);
    this->max_nthreads = num_threads;
    this->setup_thread_data(num_threads);

#endif

#ifdef EXEC_ENV_OLS
    if (files.fileExists(medoids_file)) {
      size_t tmp_dim;
      diskann::load_bin<uint32_t>(files, medoids_file, medoids, num_medoids,
                                  tmp_dim);
#else
    if (file_exists(medoids_file)) {
      size_t tmp_dim;
      diskann::load_bin<uint32_t>(medoids_file, medoids, num_medoids, tmp_dim);
#endif

      if (tmp_dim != 1) {
        std::stringstream stream;
        stream << "Error loading medoids file. Expected bin format of m times "
                  "1 vector of uint32_t."
               << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
#ifdef EXEC_ENV_OLS
      if (!files.fileExists(centroids_file)) {
#else
      if (!file_exists(centroids_file)) {
#endif
        diskann::cout
            << "Centroid data file not found. Using corresponding vectors "
               "for the medoids "
            << std::endl;
        use_medoids_data_as_centroids();
      } else {
        size_t num_centroids, aligned_tmp_dim;
#ifdef EXEC_ENV_OLS
        diskann::load_aligned_bin<float>(files, centroids_file, centroid_data,
                                         num_centroids, tmp_dim,
                                         aligned_tmp_dim);
#else
        diskann::load_aligned_bin<float>(centroids_file, centroid_data,
                                         num_centroids, tmp_dim,
                                         aligned_tmp_dim);
#endif
        if (aligned_tmp_dim != aligned_dim || num_centroids != num_medoids) {
          std::stringstream stream;
          stream << "Error loading centroids data file. Expected bin format of "
                    "m times data_dim vector of float, where m is number of "
                    "medoids "
                    "in medoids file.";
          diskann::cerr << stream.str() << std::endl;
          throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                      __LINE__);
        }
      }
    } else {
      num_medoids = 1;
      medoids = new uint32_t[1];
      medoids[0] = (_u32)(medoid_id_on_file);
      use_medoids_data_as_centroids();
    }

    std::string norm_file = std::string(disk_index_file) + "_max_base_norm.bin";

    if (file_exists(norm_file) && metric == diskann::Metric::INNER_PRODUCT) {
      _u64   dumr, dumc;
      float *norm_val;
      diskann::load_bin<float>(norm_file, norm_val, dumr, dumc);
      this->max_base_norm = norm_val[0];
      std::cout << "Setting re-scaling factor of base vectors to "
                << this->max_base_norm << std::endl;
      delete[] norm_val;
    }

    diskann::cout << "done.." << std::endl;
    return 0;
  }

  template<typename T>
  int IndexEngine<T>::init_disk_cache(uint32_t io_threads, bool use_c, float cache_scale, const std::string &index_prefix) {
    this->use_cache = use_c;
    this->cache_scale = cache_scale;
    this->n_io_nthreads = io_threads;
    // setup io pool and ctx, start from core-id [nthreads]
    if (use_cache) {
      if (io_threads > 0) {
        io_pool = std::make_shared<ThreadPool>(io_threads, this->max_nthreads);
        w_ctxs.resize(io_threads);
        disk_write_buffer.resize(io_threads);
        io_pool->runTask([&, this](int tid) {
          this->io_manager->register_thread();
          w_ctxs[tid] = this->io_manager->get_ctx();
          diskann::alloc_aligned((void **) &(disk_write_buffer[tid]),
                                (_u64) MAX_N_SECTOR_READS * (_u64) SECTOR_LEN,
                                SECTOR_LEN);

        });
      }
      // load cache data
      load_disk_cache_data(index_prefix);
      // init freq infos
      freq_ = std::make_shared<FreqWindowList>(num_points);
      // init tot page size.
      tot_cache_page = (int) (cache_scale * gp_layout_.size());
    }
    return 0;
  }

  template<typename T>
  void IndexEngine<T>::load_disk_cache_data(const std::string &index_prefix) {
    std::string disk_cache_file = 
      std::string(index_prefix) + "_disk_cache" + std::to_string(cache_scale).substr(0, 4) + ".index";
    std::string disk_cache_layout_file = 
      std::string(index_prefix) + "_disk_cache_partition" + std::to_string(cache_scale).substr(0, 4) + ".bin";
    this->cache_fid = io_manager->open(disk_cache_file, O_DIRECT | O_RDWR | O_CREAT | O_LARGEFILE);
    if (file_exists(disk_cache_layout_file)) {
      std::ifstream cache_part(disk_cache_layout_file, std::ios::binary | std::ios::in);
      _u32 partition_nums;
      cache_part.read((char *) &partition_nums, sizeof(_u32));
      this->cache_layout_.resize(partition_nums);
      cur_page_id.store(partition_nums);
      for (_u32 i = 0; i < partition_nums; i++) {
        _u32 s;
        cache_part.read((char *) &s, sizeof(_u32));
        this->cache_layout_[i].resize(s);
        cache_part.read((char *) cache_layout_[i].data(), sizeof(_u32) * s);
      }
      _u32 node_id, page_id;
      _u32 id2page_size;
      cache_part.read((char *) &id2page_size, sizeof(_u32));
      for (_u32 i = 0; i < id2page_size; i++) {
        cache_part.read((char *) &node_id, sizeof(_u32));
        cache_part.read((char *) &page_id, sizeof(_u32));
        this->id2cache_page_.insert({node_id, page_id});
      }
      diskann::cout << "Read disk cache with " << partition_nums << " nodes." << std::endl;
    } else {
      cur_page_id.store(0);
    }
  }

  template<typename T>
  void IndexEngine<T>::write_disk_cache_layout(const std::string &index_prefix) {
    std::string disk_cache_layout_file =
      std::string(index_prefix) + "_disk_cache_partition" + std::to_string(cache_scale).substr(0, 4) + ".bin";
    std::ofstream cache_part(disk_cache_layout_file, std::ios::binary | std::ios::out | std::ios::trunc);
    _u32 tot_size = cache_layout_.size();
    cache_part.write((char *) &tot_size, sizeof(_u32));
    for (_u32 i = 0; i < tot_size; i++) {
      _u32 s = cache_layout_[i].size();
      cache_part.write((char *) &s, sizeof(_u32));
      cache_part.write((char *) cache_layout_[i].data(), sizeof(_u32) * s);
    }
    tot_size = this->id2cache_page_.size();
    cache_part.write((char *) &tot_size, sizeof(_u32));
    for (auto& pair : this->id2cache_page_) {
      cache_part.write((char *) &(pair.first), sizeof(_u32));
      cache_part.write((char *) &(pair.second), sizeof(_u32));
    }
    diskann::cout << "Write disk cache with " << cache_layout_.size() << " blocks, contains " << tot_size << " nodes." << std::endl;
  }

  template<typename T>
  void IndexEngine<T>::load_partition_data(const std::string &index_prefix,
      const _u64 nnodes_per_sector, const _u64 num_points) {
    std::string partition_file = index_prefix + "_partition.bin";
    std::ifstream part(partition_file);
    _u64          C, partition_nums, nd;
    part.read((char *) &C, sizeof(_u64));
    part.read((char *) &partition_nums, sizeof(_u64));
    part.read((char *) &nd, sizeof(_u64));
    if (nnodes_per_sector && num_points &&
        (C != nnodes_per_sector || nd != num_points)) {
      diskann::cerr << "partition information not correct." << std::endl;
      exit(-1);
    }
    diskann::cout << "Partition meta: C: " << C << " partition_nums: " << partition_nums
              << " nd: " << nd << std::endl;
    this->gp_layout_.resize(partition_nums);
    for (unsigned i = 0; i < partition_nums; i++) {
      unsigned s;
      part.read((char *) &s, sizeof(unsigned));
      this->gp_layout_[i].resize(s);
      part.read((char *) gp_layout_[i].data(), sizeof(unsigned) * s);
    }
    this->id2page_.resize(nd);
    part.read((char *) id2page_.data(), sizeof(unsigned) * nd);
    diskann::cout << "Load partition data done." << std::endl;
  }

  // instantiations
  template class IndexEngine<_u8>;
  template class IndexEngine<_s8>;
  template class IndexEngine<float>;

}  // namespace diskann
