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
  if (argc != 2) {
    std::cout << argv[0]
              << "freq_file_path"
              << std::endl;
    exit(-1);
  }

  std::string freq_file(argv[1]);

  std::ifstream freq_reader(freq_file, std::ios_base::binary);

  unsigned npts = 0;
  freq_reader.read((char*) &npts, sizeof(unsigned));
  std::vector<unsigned> freq_vec(npts);
  freq_reader.read((char*) freq_vec.data(), sizeof(unsigned) * npts);

  std::vector<std::pair<unsigned, unsigned>> freq_pair_vec(npts);
  for (unsigned i = 0; i < npts; i++) {
    freq_pair_vec[i] = std::make_pair(i, freq_vec[i]);
  }
  std::sort(
      freq_pair_vec.begin(), freq_pair_vec.end(),
      [](std::pair<unsigned, unsigned>& a, std::pair<unsigned, unsigned>& b) {
        return a.second > b.second;
      });
  for (unsigned i = 0; i < npts; i++) {
    std::cout << freq_pair_vec[i].first << " " << freq_pair_vec[i].second << std::endl;
  }

  return 0;
}
