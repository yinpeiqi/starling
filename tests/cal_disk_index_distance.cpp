// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <atomic>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <pq_flash_index.h>
#include <set>
#include <string.h>
#include <time.h>
#include <boost/program_options.hpp>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"
#include "percentile_stats.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

namespace po = boost::program_options;


template<typename T>
int cal_disk_index_distance(
    diskann::Metric& metric,
    const std::string& index_path_prefix,
    const std::string& mem_index_path,
    const std::string& freq_save_path,
    const std::string& query_file,
    const std::string& disk_file_path,
    const unsigned num_threads,
    const unsigned num_nodes_to_cache,
    const _u32 mem_L) {
  // load query bin
  T*        query = nullptr;
  size_t    query_num, query_dim, query_aligned_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);

  std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
  reader.reset(new WindowsAlignedFileReader());
#else
  reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
  reader.reset(new LinuxAlignedFileReader());
#endif

  std::unique_ptr<diskann::PQFlashIndex<T>> _pFlashIndex(
      new diskann::PQFlashIndex<T>(reader, false, metric));

  int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str(), disk_file_path);

  if (res != 0) {
    return res;
  }

  // load in-memory navigation graph
  if (mem_L) {
    _pFlashIndex->load_mem_index(metric, query_aligned_dim, mem_index_path, num_threads, mem_L);
  }

  std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

  // cache bfs levels
  std::vector<uint32_t> node_list;
  diskann::cout << "Caching " << num_nodes_to_cache
                << " BFS nodes around medoid(s)" << std::endl;
  //_pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
  if (num_nodes_to_cache > 0) {
    diskann::cout << "assume all nodes in cache." << std::endl;
    for (uint32_t i = 0; i < num_nodes_to_cache; i++) {
      node_list.push_back(i);
    }
  }
  _pFlashIndex->load_cache_list(node_list);
  node_list.clear();
  node_list.shrink_to_fit();

  omp_set_num_threads(num_threads);

  diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  diskann::cout.precision(2);

  _pFlashIndex->generate_node_distance_to_mediod(freq_save_path, mem_L);

  diskann::aligned_free(query);
  return 0;
}

int main(int argc, char** argv) {
  std::string data_type, dist_fn, index_path_prefix,
      query_file, disk_file_path, freq_save_path, mem_index_path;
  unsigned              num_threads, num_nodes_to_cache;
  unsigned              mem_L;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                       "distance function <l2/mips/fast_l2>");
    desc.add_options()("index_path_prefix",
                       po::value<std::string>(&index_path_prefix)->required(),
                       "Path prefix to the index");
    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "Query file in binary format");
    desc.add_options()(
        "num_nodes_to_cache",
        po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
        "Beamwidth for search");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    desc.add_options()("mem_L", po::value<unsigned>(&mem_L)->default_value(0),
                       "The L of the in-memory navigation graph while searching. Use 0 to disable");
    desc.add_options()("disk_file_path", po::value<std::string>(&disk_file_path)->required(),
                       "The path of the disk file (_disk.index in the original DiskANN)");
    desc.add_options()("freq_save_path", po::value<std::string>(&freq_save_path)->required(),
                       "frequency file save path");
    desc.add_options()("mem_index_path", po::value<std::string>(&mem_index_path)->default_value(""),
                       "The prefix path of the mem_index");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  diskann::Metric metric;
  if (dist_fn == std::string("mips")) {
    metric = diskann::Metric::INNER_PRODUCT;
  } else if (dist_fn == std::string("l2")) {
    metric = diskann::Metric::L2;
  } else if (dist_fn == std::string("cosine")) {
    metric = diskann::Metric::COSINE;
  } else {
    std::cout << "Unsupported distance function. Currently only L2/ Inner "
                 "Product/Cosine are supported."
              << std::endl;
    return -1;
  }

  if ((data_type != std::string("float")) &&
      (metric == diskann::Metric::INNER_PRODUCT)) {
    std::cout << "Currently support only floating point data for Inner Product."
              << std::endl;
    return -1;
  }

  try {
    if (data_type == std::string("float"))
      return cal_disk_index_distance<float>(metric, index_path_prefix, mem_index_path, freq_save_path,
            query_file, disk_file_path, num_threads, num_nodes_to_cache, mem_L);
    else if (data_type == std::string("int8"))
      return cal_disk_index_distance<int8_t>(metric, index_path_prefix, mem_index_path, freq_save_path,
            query_file, disk_file_path, num_threads, num_nodes_to_cache, mem_L);
    else if (data_type == std::string("uint8"))
      return cal_disk_index_distance<uint8_t>(metric, index_path_prefix, mem_index_path, freq_save_path,
            query_file, disk_file_path, num_threads, num_nodes_to_cache, mem_L);
    else {
      std::cerr << "Unsupported data type. Use float or int8 or uint8"
                << std::endl;
      return -1;
    }
  } catch (const std::exception& e) {
    std::cout << std::string(e.what()) << std::endl;
    diskann::cerr << "Index search failed." << std::endl;
    return -1;
  }
}
