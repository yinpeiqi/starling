#pragma once
#include "utils.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

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
    char* node_buf;
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
        this->fid = fid;  // here we consider we have only one file.
    }
  };
}