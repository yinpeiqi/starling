#!/bin/sh

# Switch dataset in the config_local.sh file by calling the desired function

DATA_DIR="/raid0_data"

#################
#   BIGANN10M   #
#################
dataset_sift1M() {
  BASE_PATH=${DATA_DIR}/sift/sift_base.fbin
  QUERY_FILE=${DATA_DIR}/sift/sift_query.fbin
  GT_FILE=${DATA_DIR}/sift/sift_query_base_gt100.ibin 
  PREFIX=bigann_10m
  DATA_TYPE=float
  DIST_FN=l2
  B=0.03
  K=10
  DATA_DIM=128
  DATA_N=1000000
}

dataset_bigann1M() {
  BASE_PATH=${DATA_DIR}/bigann/bigann_base_1M.bin
  QUERY_FILE=${DATA_DIR}/bigann/bigann_query.bin
  GT_FILE=${DATA_DIR}/bigann/gnd/computed_gt_1000_1M.bin
  PREFIX=bigann_1M
  DATA_TYPE=uint8
  DIST_FN=l2
  B=0.03
  K=10
  DATA_DIM=128
  DATA_N=1000000
}

dataset_bigann5M() {
  BASE_PATH=${DATA_DIR}/bigann/bigann_base_5M.bin
  QUERY_FILE=${DATA_DIR}/bigann/bigann_query.bin
  GT_FILE=${DATA_DIR}/bigann/gnd/computed_gt_1000_5M.bin 
  PREFIX=bigann_5M
  DATA_TYPE=uint8
  DIST_FN=l2
  B=0.15
  K=10
  DATA_DIM=128
  DATA_N=5000000
}

dataset_bigann10M() {
  BASE_PATH=${DATA_DIR}/bigann/bigann_base_10M.bin
  QUERY_FILE=${DATA_DIR}/bigann/bigann_query.bin
  GT_FILE=${DATA_DIR}/bigann/gnd/computed_gt_1000_10M.bin 
  PREFIX=bigann_10M
  DATA_TYPE=uint8
  DIST_FN=l2
  B=0.3
  K=10
  DATA_DIM=128
  DATA_N=10000000
}

dataset_bigann20M() {
  BASE_PATH=${DATA_DIR}/bigann/bigann_base_20M.bin
  QUERY_FILE=${DATA_DIR}/bigann/bigann_query.bin
  GT_FILE=${DATA_DIR}/bigann/gnd/computed_gt_1000_20M.bin 
  PREFIX=bigann_20M
  DATA_TYPE=uint8
  DIST_FN=l2
  B=0.6
  K=10
  DATA_DIM=128
  DATA_N=20000000
}

dataset_bigann50M() {
  BASE_PATH=${DATA_DIR}/bigann/bigann_base_50M.bin
  QUERY_FILE=${DATA_DIR}/bigann/bigann_query.bin
  GT_FILE=${DATA_DIR}/bigann/gnd/computed_gt_1000_50M.bin 
  PREFIX=bigann_50M
  DATA_TYPE=uint8
  DIST_FN=l2
  B=1.5
  K=10
  DATA_DIM=128
  DATA_N=50000000
}

dataset_bigann100M() {
  BASE_PATH=${DATA_DIR}/bigann/bigann_base_100M.bin
  QUERY_FILE=${DATA_DIR}/bigann/bigann_query.bin
  GT_FILE=${DATA_DIR}/bigann/gnd/computed_gt_1000_100M.bin 
  PREFIX=bigann_100M
  DATA_TYPE=uint8
  DIST_FN=l2
  B=3
  K=10
  DATA_DIM=128
  DATA_N=100000000
}
