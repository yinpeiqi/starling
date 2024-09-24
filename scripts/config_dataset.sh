#!/bin/sh

# Switch dataset in the config_local.sh file by calling the desired function

DATA_DIR="/ssd4_data"

#################
#   BIGANN10M   #
#################
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

dataset_bigann100M() {
  BASE_PATH=${DATA_DIR}/bigann/bigann_base_100M.bin
  QUERY_FILE=${DATA_DIR}/bigann/bigann_query.bin
  GT_FILE=${DATA_DIR}/bigann/gnd/computed_gt_1000_100M.bin 
  PREFIX=bigann_100M
  DATA_TYPE=uint8
  DIST_FN=l2
  B=3.3
  K=10
  DATA_DIM=128
  DATA_N=100000000
}
