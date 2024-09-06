#include <omp.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include "partition_and_pq.h"
#include "utils.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <typeinfo>
#include <utility>
#include <vector>


int main(int argc, char** argv) {
  // if (argc != 2) {
  //   std::cout << argv[0]
  //             << "freq_file_path"
  //             << std::endl;
  //   exit(-1);
  // }

  std::string freq_file("/raid0_data/starling/bigann_10M_M32_R48_L128_B0.3/FREQ/NQ_0_BM_4_L_100_T_64/_freq.bin");

  std::ifstream freq_reader(freq_file, std::ios_base::binary);
  unsigned npts = 0;
  freq_reader.read((char*) &npts, sizeof(unsigned));
  std::vector<unsigned> freq_vec(npts);
  freq_reader.read((char*) freq_vec.data(), sizeof(unsigned) * npts);

  std::string dist_file("/raid0_data/starling/bigann_10M_M32_R48_L128_B0.3/FREQ/NQ_0_BM_4_L_100_T_1/_nhops.bin");

  std::ifstream dist_reader(dist_file, std::ios_base::binary);
  unsigned npt2 = 0;
  dist_reader.read((char*) &npt2, sizeof(unsigned));
  if (npts != npt2) {
    std::cout << "mismatch:" << npts << " " << npt2 << std::endl;
    exit(-1);
  }
  std::vector<float> dist_vec(npts);
  dist_reader.read((char*) dist_vec.data(), sizeof(float) * npts);

  std::vector<std::pair<unsigned, std::pair<unsigned, float>>> freq_pair_vec(npts);
  for (unsigned i = 0; i < npts; i++) {
    freq_pair_vec[i] = std::make_pair(i, std::make_pair(freq_vec[i], dist_vec[i]));
  }
  std::sort(
      freq_pair_vec.begin(), freq_pair_vec.end(),
      [](std::pair<unsigned, std::pair<unsigned, float>>& a, std::pair<unsigned, std::pair<unsigned, float>>& b) {
        return a.second.second > b.second.second;
      });

  for (unsigned i = 0; i < npts; i++) {
    std::cout << freq_pair_vec[i].first << " " << freq_pair_vec[i].second.first << " " << freq_pair_vec[i].second.second << std::endl;
  }

  return 0;
}
