# assume this is your data path
DATA_DIR="/ssd4_data"

cd ${DATA_DIR}
mkdir bigann
cd bigann
# download data.
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz

# unzip data
gunzip ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
gunzip ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz

# make sure you have already built the code, here are sample steps:
# cd ~/starling/scripts
# ./run_benchmark.sh release
${DATA_DIR}/starling/release/tests/utils/bvecs_to_bin ${DATA_DIR}/bigann/bigann_query.bvecs ${DATA_DIR}/bigann/bigann_query.bin
${DATA_DIR}/starling/release/tests/utils/bvecs_to_bin ${DATA_DIR}/bigann/bigann_base.bvecs ${DATA_DIR}/bigann/bigann_base_10M.bin 10000000
${DATA_DIR}/starling/release/tests/utils/bvecs_to_bin ${DATA_DIR}/bigann/bigann_base.bvecs ${DATA_DIR}/bigann/bigann_base_100M.bin 100000000
# ${DATA_DIR}/starling/release/tests/utils/bvecs_to_bin ${DATA_DIR}/bigann/bigann_base.bvecs ${DATA_DIR}/bigann/bigann_base_1000M.bin 1000000000

for size in 10 100; do
    ${DATA_DIR}/starling/release/tests/utils/compute_groundtruth  --data_type uint8 --dist_fn l2 --base_file ${DATA_DIR}/bigann/bigann_base_${size}M.bin --query_file  ${DATA_DIR}/bigann/bigann_query.bin --gt_file ${DATA_DIR}/bigann/gnd/computed_gt_1000_${size}M.bin --K 1000
done

# To test the code: (just using the default config_local.sh is enough)
# cd ~/starling/scripts
# ./run_benchmark.sh release build
# ./run_benchmark.sh release build_mem
# ./run_benchmark.sh release gp
# ./run_benchmark.sh release search knn
