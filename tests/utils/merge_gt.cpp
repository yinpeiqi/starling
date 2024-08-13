// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"

void block_convert(std::ifstream& reader, std::ofstream& writer, _u32* read_buf,
                   _u32* write_buf, _u64 npts, _u64 ndims) {
  reader.read((char*) read_buf,
              npts * (ndims * sizeof(_u32) + sizeof(unsigned)));
  for (_u64 i = 0; i < npts; i++) {
    memcpy(write_buf + i * ndims, (read_buf + i * (ndims + 1)) + 1,
           ndims * sizeof(_u32));
  }
  writer.write((char*) write_buf, npts * ndims * sizeof(_u32));
}

void block_convert2(std::ifstream& reader, std::ofstream& writer, float* read_buf,
                   float* write_buf, _u64 npts, _u64 ndims) {
  reader.read((char*) read_buf,
              npts * (ndims * sizeof(float) + sizeof(unsigned)));
  for (_u64 i = 0; i < npts; i++) {
    memcpy(write_buf + i * ndims, (read_buf + i * (ndims + 1)) + 1,
           ndims * sizeof(float));
  }
  writer.write((char*) write_buf, npts * ndims * sizeof(float));
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << argv[0] << " input_ivecs output_bin input_fvecs" << std::endl;
    exit(-1);
  }
  std::ifstream reader(argv[1], std::ios::binary | std::ios::ate);
  _u64          fsize = reader.tellg();
  reader.seekg(0, std::ios::beg);

  unsigned ndims_u32;
  reader.read((char*) &ndims_u32, sizeof(unsigned));
  reader.seekg(0, std::ios::beg);
  _u64 ndims = (_u64) ndims_u32;
  _u64 npts = fsize / ((ndims + 1) * sizeof(_u32));
  std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims
            << std::endl;

  _u64 blk_size = 131072;
  _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;
  std::cout << "# blks: " << nblks << std::endl;
  std::ofstream writer(argv[2], std::ios::binary);
  int           npts_s32 = (_s32) npts;
  int           ndims_s32 = (_s32) ndims;
  writer.write((char*) &npts_s32, sizeof(_s32));
  writer.write((char*) &ndims_s32, sizeof(_s32));
  _u32* read_buf = new _u32[npts * (ndims + 1)];
  _u32* write_buf = new _u32[npts * ndims];
  for (_u64 i = 0; i < nblks; i++) {
    _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
    block_convert(reader, writer, read_buf, write_buf, cblk_size, ndims);
    std::cout << "Block #" << i << " written" << std::endl;
  }

  delete[] read_buf;
  delete[] write_buf;

  std::ifstream reader2(argv[3], std::ios::binary | std::ios::ate);
  fsize = reader2.tellg();
  reader2.seekg(0, std::ios::beg);

  reader2.read((char*) &ndims_u32, sizeof(unsigned));
  reader2.seekg(0, std::ios::beg);
  ndims = (_u64) ndims_u32;
  npts = fsize / ((ndims + 1) * sizeof(float));
  std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims
            << std::endl;

  nblks = ROUND_UP(npts, blk_size) / blk_size;
  std::cout << "# blks: " << nblks << std::endl;

  npts_s32 = (_s32) npts;
  ndims_s32 = (_s32) ndims;

  float* read_buf2 = new float[npts * (ndims + 1)];
  float* write_buf2 = new float[npts * ndims];
  for (_u64 i = 0; i < nblks; i++) {
    _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
    block_convert2(reader2, writer, read_buf2, write_buf2, cblk_size, ndims);
    std::cout << "Block #" << i << " written" << std::endl;
  }
  
  
  delete[] read_buf2;
  delete[] write_buf2;

  reader.close();
  writer.close();
}
