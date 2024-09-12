#!/bin/sh
source config_dataset.sh

# Choose the dataset by uncomment the line below
# If multiple lines are uncommented, only the last dataset is effective
dataset_bigann10M

##################
#   Disk Build   #
##################
R=48
BUILD_L=128
M=32   # 500G for graph build
BUILD_T=64

##################
#       SQ       #
##################
USE_SQ=0

##################################
#   In-Memory Navigation Graph   #
##################################
MEM_R=24
MEM_BUILD_L=128
MEM_ALPHA=1.2
MEM_RAND_SAMPLING_RATE=0.1
MEM_USE_FREQ=0
MEM_FREQ_USE_RATE=0.01  # not use

##########################
#   Generate Frequency   #
##########################
FREQ_QUERY_FILE=$QUERY_FILE
FREQ_QUERY_CNT=0 # Set 0 to use all (default)
FREQ_BM=4
FREQ_L=100 # only support one value at a time for now
FREQ_T=1
FREQ_CACHE=0
FREQ_MEM_L=0 # non-zero to enable
FREQ_MEM_TOPK=10

#######################
#   Graph Partition   #
#######################
GP_TIMES=16
GP_T=64
GP_LOCK_NUMS=0 # will lock nodes at init, the lock_node_nums = partition_size * GP_LOCK_NUMS
GP_USE_FREQ=1 # use freq file to partition graph, if USE_FREQ=2 means replace freq with dist
GP_CUT=4096 # the graph's degree will been limited at 4096
GP_SCALE_F=1 # the scale factor.

##############
#   Search   #
##############
BM_LIST=(4)
T_LIST=(1)
# T_LIST=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 20 24 28 32 40 48 56 64)
# T_LIST=(16 20 24 28 32 40 48 56 64)
CACHE=0
MEM_L=10 # non-zero to enable

USE_ENGINE=1 # whether use search engine
# Page Search
USE_PAGE_SEARCH=1 # Set 0 for beam search, 1 for page search (default)
PS_USE_RATIO=0.3
PQ_FILTER_RATIO=1.2

# KNN
LS="15 20 25 30 35 37 40 42 45 47 50 55 60 70 80"

# Range search
RS_LS="80"
RS_ITER_KNN_TO_RANGE_SEARCH=1 # 0 for custom search, 1 for iterating via KNN, combine with USE_PAGE_SEARCH
KICKED_SIZE=0 # non-zero to reuse intermediate states during page search
RS_CUSTOM_ROUND=0 # set when use custom search, 0 for all pages within radius
