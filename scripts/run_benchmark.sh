#!/bin/bash

set -e
# set -x

source config_local.sh

INDEX_PREFIX_PATH="${PREFIX}_M${M}_R${R}_L${BUILD_L}_B${B}/"
MEM_SAMPLE_PATH="${INDEX_PREFIX_PATH}SAMPLE_RATE_${MEM_RAND_SAMPLING_RATE}/"
MEM_INDEX_PATH="${INDEX_PREFIX_PATH}MEM_R_${MEM_R}_L_${MEM_BUILD_L}_ALPHA_${MEM_ALPHA}_MEM_USE_FREQ${MEM_USE_FREQ}_RANDOM_RATE${MEM_RAND_SAMPLING_RATE}_FREQ_RATE${MEM_FREQ_USE_RATE}/"
GP_PATH="${INDEX_PREFIX_PATH}GP_TIMES_${GP_TIMES}_LOCK_${GP_LOCK_NUMS}_GP_USE_FREQ${GP_USE_FREQ}_CUT${GP_CUT}_SCALE${GP_SCALE_F}/"
FREQ_PATH="${INDEX_PREFIX_PATH}FREQ/NQ_${FREQ_QUERY_CNT}_BM_${FREQ_BM}_L_${FREQ_L}_T_${FREQ_T}/"

SUMMARY_FILE_PATH="${DATA_DIR}/starling/${INDEX_PREFIX_PATH}/summary.log"

print_usage_and_exit() {
  echo "Usage: ./run_benchmark.sh [debug/release] [build/build_mem/freq/gp/search] [knn/range]"
  exit 1
}

check_dir_and_make_if_absent() {
  local dir=$1
  if [ -d $dir ]; then
    echo "Directory $dir is already exit. Remove or rename it and then re-run."
    exit 1
  else
    mkdir -p ${dir}
  fi
}

case $1 in
  debug)
    cmake -DCMAKE_BUILD_TYPE=Debug .. -B ${DATA_DIR}/starling/debug
    EXE_PATH=${DATA_DIR}/starling/debug
  ;;
  release)
    cmake -DCMAKE_BUILD_TYPE=Release .. -B ${DATA_DIR}/starling/release
    EXE_PATH=${DATA_DIR}/starling/release
  ;;
  *)
    print_usage_and_exit
  ;;
esac
pushd $EXE_PATH
make -j
popd

mkdir -p ${DATA_DIR}/starling && cd ${DATA_DIR}/starling

date
case $2 in
  build)
    check_dir_and_make_if_absent ${INDEX_PREFIX_PATH}
    echo "Building disk index..."
    time ${EXE_PATH}/tests/build_disk_index \
      --data_type $DATA_TYPE \
      --dist_fn $DIST_FN \
      --data_path $BASE_PATH \
      --index_path_prefix $INDEX_PREFIX_PATH \
      -R $R \
      -L $BUILD_L \
      -B $B \
      -M $M \
      -T $BUILD_T > ${INDEX_PREFIX_PATH}build.log
    cp ${INDEX_PREFIX_PATH}_disk.index ${INDEX_PREFIX_PATH}_disk_beam_search.index
  ;;
  sq)
    cp  ${INDEX_PREFIX_PATH}_disk_beam_search.index ${INDEX_PREFIX_PATH}_disk.index 
    time ${EXE_PATH}/tests/utils/sq ${INDEX_PREFIX_PATH} > ${INDEX_PREFIX_PATH}sq.log
  ;;
  build_mem)
    if [ ${MEM_USE_FREQ} -eq 1 ]; then
      if [ ! -d ${FREQ_PATH} ]; then
        echo "Seems you have not gen the freq file, run this script again: ./run_benchmark.sh [debug/release] freq [knn/range]"
        exit 1;
      fi
      echo "Parsing freq file..."
      time ${EXE_PATH}/tests/utils/parse_freq_file ${DATA_TYPE} ${BASE_PATH} ${FREQ_PATH}_freq.bin ${FREQ_PATH} ${MEM_FREQ_USE_RATE} 
      MEM_DATA_PATH=${FREQ_PATH}
    else
      mkdir -p ${MEM_SAMPLE_PATH}
      echo "Generating random slice..."
      time ${EXE_PATH}/tests/utils/gen_random_slice $DATA_TYPE $BASE_PATH $MEM_SAMPLE_PATH $MEM_RAND_SAMPLING_RATE > ${MEM_SAMPLE_PATH}sample.log
      MEM_DATA_PATH=${MEM_SAMPLE_PATH}
    fi
    echo "Building memory index..."
    check_dir_and_make_if_absent ${MEM_INDEX_PATH}
    time ${EXE_PATH}/tests/build_memory_index \
      --data_type ${DATA_TYPE} \
      --dist_fn ${DIST_FN} \
      --data_path ${MEM_DATA_PATH} \
      --index_path_prefix ${MEM_INDEX_PATH}_index \
      -R ${MEM_R} \
      -L ${MEM_BUILD_L} \
      --alpha ${MEM_ALPHA} > ${MEM_INDEX_PATH}build.log
  ;;
  freq)
    # check_dir_and_make_if_absent ${FREQ_PATH}
    FREQ_LOG="${FREQ_PATH}freq.log"

    DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk_beam_search.index
    if [ ! -f $DISK_FILE_PATH ]; then
      DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk.index
    fi

    echo "Generating frequency file... ${FREQ_LOG}"
    time ${EXE_PATH}/tests/search_disk_index_save_freq \
              --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --freq_save_path $FREQ_PATH \
              --query_file $FREQ_QUERY_FILE \
              --expected_query_num $FREQ_QUERY_CNT \
              --gt_file $GT_FILE \
              -K $K \
              --result_path ${FREQ_PATH}result \
              --num_nodes_to_cache ${FREQ_CACHE} \
              -T $FREQ_T \
              -L $FREQ_L \
              -W $FREQ_BM \
              --mem_L ${FREQ_MEM_L} \
              --use_page_search 0 \
              --disk_file_path ${DISK_FILE_PATH} > ${FREQ_LOG}
  ;;
  dist)
    # check_dir_and_make_if_absent ${FREQ_PATH}
    FREQ_LOG="${FREQ_PATH}_dist_freq.log"

    DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk_beam_search.index
    if [ ! -f $DISK_FILE_PATH ]; then
      DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk.index
    fi

    echo "Generating distance (frequency) file... ${FREQ_LOG}"
    time ${EXE_PATH}/tests/cal_disk_index_distance \
              --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --freq_save_path $FREQ_PATH \
              --query_file $FREQ_QUERY_FILE \
              --num_nodes_to_cache ${DATA_N} \
              -T $FREQ_T \
              --mem_L ${FREQ_MEM_L} \
              --mem_index_path ${MEM_INDEX_PATH}_index \
              --disk_file_path ${DISK_FILE_PATH} > ${FREQ_LOG}
  ;;
  dgp)
    # DiskANN graph partition
    DGP_PATH="${INDEX_PREFIX_PATH}_DISKANN/"
    OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
    if [ ! -f "$OLD_INDEX_FILE" ]; then
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
    fi
    #using sq index file to gp
    GP_DATA_TYPE=$DATA_TYPE
    if [ $USE_SQ -eq 1 ]; then 
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
      GP_DATA_TYPE=uint8
    fi
    if [ -d ${DGP_PATH} ]; then
      echo "Directory ${DGP_PATH} is already exit."
      GP_FILE_PATH=${DGP_PATH}_part.bin
      if [ -f "${GP_FILE_PATH}" ]; then
        cp ${OLD_INDEX_FILE} ${INDEX_PREFIX_PATH}_disk.index
        cp ${GP_FILE_PATH} ${INDEX_PREFIX_PATH}_partition.bin
        echo "copy done."
        exit 0
      else
        echo "${GP_FILE_PATH} not found."
        exit 1
      fi
    else
      mkdir -p ${DGP_PATH}
      GP_FILE_PATH=${DGP_PATH}_part.bin
      echo "Running graph partition... ${GP_FILE_PATH}.log"
      time ${EXE_PATH}/graph_partition/partitioner --index_file ${OLD_INDEX_FILE} \
        --data_type $GP_DATA_TYPE --gp_file $GP_FILE_PATH -T $GP_T --diskann 1 > ${GP_FILE_PATH}.log

      cp ${OLD_INDEX_FILE} ${INDEX_PREFIX_PATH}_disk.index
      cp ${GP_FILE_PATH} ${INDEX_PREFIX_PATH}_partition.bin
    fi
  ;;
  gp)
    # graph partition with scaled
    if [ -d ${GP_PATH} ]; then
      echo "Directory ${GP_PATH} is already exit."
      if [ -f "${GP_PATH}_part_tmp.index" ]; then
        echo "copy ${GP_PATH}_part_tmp.index to ${INDEX_PREFIX_PATH}_disk.index."
        GP_FILE_PATH=${GP_PATH}_part.bin
        cp ${GP_PATH}_part_tmp.index ${INDEX_PREFIX_PATH}_disk.index
        cp ${GP_FILE_PATH} ${INDEX_PREFIX_PATH}_partition.bin
        exit 0
      else
        echo "${GP_PATH}_part_tmp.index not found."
        exit 1
      fi
    else
      mkdir -p ${GP_PATH}
      # check_dir_and_make_if_absent ${GP_PATH}
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
      if [ ! -f "$OLD_INDEX_FILE" ]; then
        OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
      fi
      #using sq index file to gp
      GP_DATA_TYPE=$DATA_TYPE
      if [ $USE_SQ -eq 1 ]; then 
        OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
        GP_DATA_TYPE=uint8
      fi
      GP_FILE_PATH=${GP_PATH}_part.bin
      echo "Running graph partition... ${GP_FILE_PATH}.log"
      if [ ${GP_USE_FREQ} -eq 1 ]; then
        echo "use freq"
        time ${EXE_PATH}/graph_partition/partitioner --index_file ${OLD_INDEX_FILE} \
          --data_type $GP_DATA_TYPE --gp_file $GP_FILE_PATH -T $GP_T --ldg_times $GP_TIMES --freq_file ${FREQ_PATH}_freq.bin --lock_nums ${GP_LOCK_NUMS} --cut ${GP_CUT} --scale ${GP_SCALE_F} > ${GP_FILE_PATH}.log
      else
        echo "use dist"
        if [ ${GP_USE_FREQ} -eq 2 ]; then
          gdb ${EXE_PATH}/graph_partition/partitioner -ex "run --index_file ${OLD_INDEX_FILE} \
            --data_type $GP_DATA_TYPE --gp_file $GP_FILE_PATH -T $GP_T --ldg_times $GP_TIMES --freq_file ${FREQ_PATH}_nhops.bin --lock_nums ${GP_LOCK_NUMS} --cut ${GP_CUT} --scale ${GP_SCALE_F} --dist_freq 1 > ${GP_FILE_PATH}.log"
        else
        echo "not using freq or dist"
          gdb ${EXE_PATH}/graph_partition/partitioner -ex "run --index_file ${OLD_INDEX_FILE} \
            --data_type $GP_DATA_TYPE --gp_file $GP_FILE_PATH -T $GP_T --ldg_times $GP_TIMES --scale ${GP_SCALE_F} > ${GP_FILE_PATH}.log"
        fi
      fi

      echo "Running relayout... ${GP_PATH}relayout.log"
      time ${EXE_PATH}/tests/utils/index_relayout ${OLD_INDEX_FILE} ${GP_FILE_PATH} > ${GP_PATH}relayout.log
      if [ ! -f "${INDEX_PREFIX_PATH}_disk_beam_search.index" ]; then
        mv $OLD_INDEX_FILE ${INDEX_PREFIX_PATH}_disk_beam_search.index
      fi
      #TODO: Use only one index file
      cp ${GP_PATH}_part_tmp.index ${INDEX_PREFIX_PATH}_disk.index
      cp ${GP_FILE_PATH} ${INDEX_PREFIX_PATH}_partition.bin
    fi
  ;;
  search)
    mkdir -p ${INDEX_PREFIX_PATH}/search
    mkdir -p ${INDEX_PREFIX_PATH}/result
    if [ ! -d "$INDEX_PREFIX_PATH" ]; then
      echo "Directory $INDEX_PREFIX_PATH is not exist. Build it first?"
      exit 1
    fi

    # choose the disk index file by settings
    DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk.index
    if [ $USE_PAGE_SEARCH -eq 1 ]; then
      if [ ! -f ${INDEX_PREFIX_PATH}_partition.bin ]; then
        echo "Partition file not found. Run the script with gp option first."
        exit 1
      fi
      echo "Using Page Search"
    else
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
      if [ -f ${OLD_INDEX_FILE} ]; then
        DISK_FILE_PATH=$OLD_INDEX_FILE
      else
        echo "make sure you have not gp the index file"
      fi
      echo "Using Beam Search"
    fi

    log_arr=()
    case $3 in
      gdbknn)
        for BW in ${BM_LIST[@]}
        do
          for T in ${T_LIST[@]}
          do
            SEARCH_LOG=${INDEX_PREFIX_PATH}search/search_SQ${USE_SQ}_K${K}_CACHE${CACHE}_BW${BW}_T${T}_MEML${MEM_L}_MEMK${MEM_TOPK}_MEM_USE_FREQ${MEM_USE_FREQ}_PS${USE_PAGE_SEARCH}_USE_RATIO${PS_USE_RATIO}_GP_USE_FREQ${GP_USE_FREQ}_GP_LOCK_NUMS${GP_LOCK_NUMS}_GP_CUT${GP_CUT}.log
            echo "Searching... log file: ${SEARCH_LOG}"
            sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; 
            if [ $USE_ENGINE -eq 1 ]; then
              gdb ${EXE_PATH}/tests/search_disk_index_use_engine -ex "run --data_type $DATA_TYPE \
                --dist_fn $DIST_FN \
                --index_path_prefix $INDEX_PREFIX_PATH \
                --query_file $QUERY_FILE \
                --gt_file $GT_FILE \
                -K $K \
                --result_path ${INDEX_PREFIX_PATH}result/result \
                --num_nodes_to_cache $CACHE \
                -T $T \
                -L ${LS} \
                -W $BW \
                --mem_L ${MEM_L} \
                --mem_index_path ${MEM_INDEX_PATH}_index \
                --use_page_search ${USE_PAGE_SEARCH} \
                --pq_ratio ${PQ_FILTER_RATIO} \
                --disk_file_path ${DISK_FILE_PATH} \
                --use_sq ${USE_SQ} \
                --io_T ${IO_T} \
                --use_cache ${USE_CACHE}       > ${SEARCH_LOG}" 
              log_arr+=( ${SEARCH_LOG} )
            else
              gdb ${EXE_PATH}/tests/search_disk_index -ex "run --data_type $DATA_TYPE \
                --dist_fn $DIST_FN \
                --index_path_prefix $INDEX_PREFIX_PATH \
                --query_file $QUERY_FILE \
                --gt_file $GT_FILE \
                -K $K \
                --result_path ${INDEX_PREFIX_PATH}result/result \
                --num_nodes_to_cache $CACHE \
                -T $T \
                -L ${LS} \
                -W $BW \
                --mem_L ${MEM_L} \
                --mem_index_path ${MEM_INDEX_PATH}_index \
                --use_page_search ${USE_PAGE_SEARCH} \
                --use_ratio ${PS_USE_RATIO} \
                --disk_file_path ${DISK_FILE_PATH} \
                --use_sq ${USE_SQ}       > ${SEARCH_LOG}" 
              log_arr+=( ${SEARCH_LOG} )
            fi
          done
        done
      ;;
      knn)
        for BW in ${BM_LIST[@]}
        do
          for T in ${T_LIST[@]}
          do
            SEARCH_LOG=${INDEX_PREFIX_PATH}search/search_SQ${USE_SQ}_K${K}_CACHE${CACHE}_BW${BW}_T${T}_MEML${MEM_L}_MEMK${MEM_TOPK}_MEM_USE_FREQ${MEM_USE_FREQ}_PS${USE_PAGE_SEARCH}_USE_RATIO${PS_USE_RATIO}_GP_USE_FREQ${GP_USE_FREQ}_GP_LOCK_NUMS${GP_LOCK_NUMS}_GP_CUT${GP_CUT}.log
            echo "Searching... log file: ${SEARCH_LOG}"
            if [ $USE_ENGINE -eq 1 ]; then
              sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; ${EXE_PATH}/tests/search_disk_index_use_engine --data_type $DATA_TYPE \
                --dist_fn $DIST_FN \
                --index_path_prefix $INDEX_PREFIX_PATH \
                --query_file $QUERY_FILE \
                --gt_file $GT_FILE \
                -K $K \
                --result_path ${INDEX_PREFIX_PATH}result/result \
                --num_nodes_to_cache $CACHE \
                -T $T \
                -L ${LS} \
                -W $BW \
                --mem_L ${MEM_L} \
                --mem_index_path ${MEM_INDEX_PATH}_index \
                --use_page_search ${USE_PAGE_SEARCH} \
                --pq_ratio ${PQ_FILTER_RATIO} \
                --disk_file_path ${DISK_FILE_PATH} \
                --use_sq ${USE_SQ} \
                --io_T ${IO_T} \
                --use_cache ${USE_CACHE}       > ${SEARCH_LOG} 
              log_arr+=( ${SEARCH_LOG} )
            else
              sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; ${EXE_PATH}/tests/search_disk_index --data_type $DATA_TYPE \
                --dist_fn $DIST_FN \
                --index_path_prefix $INDEX_PREFIX_PATH \
                --query_file $QUERY_FILE \
                --gt_file $GT_FILE \
                -K $K \
                --result_path ${INDEX_PREFIX_PATH}result/result \
                --num_nodes_to_cache $CACHE \
                -T $T \
                -L ${LS} \
                -W $BW \
                --mem_L ${MEM_L} \
                --mem_index_path ${MEM_INDEX_PATH}_index \
                --use_page_search ${USE_PAGE_SEARCH} \
                --use_ratio ${PS_USE_RATIO} \
                --disk_file_path ${DISK_FILE_PATH} \
                --use_sq ${USE_SQ}       > ${SEARCH_LOG} 
              log_arr+=( ${SEARCH_LOG} )
            fi
          done
        done
      ;;
      range)
        for BW in ${BM_LIST[@]}
        do
          for T in ${T_LIST[@]}
          do
            SEARCH_LOG=${INDEX_PREFIX_PATH}search/search_RADIUS${RADIUS}_CACHE${CACHE}_BW${BW}_T${T}_PS${USE_PAGE_SEARCH}_PS_RATIO${PS_USE_RATIO}_ITER_KNN${RS_ITER_KNN_TO_RANGE_SEARCH}_MEM_L${MEM_L}.log
            echo "Searching... log file: ${SEARCH_LOG}"
            sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; ${EXE_PATH}/tests/range_search_disk_index \
              --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --num_nodes_to_cache $CACHE \
              -T $T \
              -W $BW \
              --query_file $QUERY_FILE \
              --gt_file $GT_FILE \
              --range_threshold $RADIUS \
              -L $RS_LS \
              --disk_file_path ${DISK_FILE_PATH} \
              --use_page_search ${USE_PAGE_SEARCH} \
              --iter_knn_to_range_search ${RS_ITER_KNN_TO_RANGE_SEARCH} \
              --use_ratio ${PS_USE_RATIO} \
              --mem_index_path ${MEM_INDEX_PATH}_index \
              --mem_L ${MEM_L} \
              --custom_round_num ${RS_CUSTOM_ROUND} \
              --kicked_size ${KICKED_SIZE} \
              > ${SEARCH_LOG}
            log_arr+=( ${SEARCH_LOG} )
          done
        done
      ;;
      *)
        print_usage_and_exit
      ;;
    esac
    if [ ${#log_arr[@]} -ge 1 ]; then
      TITLES=$(cat ${log_arr[0]} | grep -E "^\s+L\s+")
      for f in "${log_arr[@]}"
      do
        printf "$f\n" | tee -a $SUMMARY_FILE_PATH
        printf "${TITLES}\n" | tee -a $SUMMARY_FILE_PATH
        cat $f | grep -E "([0-9]+(\.[0-9]+\s+)){5,}" | tee -a $SUMMARY_FILE_PATH
        printf "\n\n" >> $SUMMARY_FILE_PATH
      done
    fi
  ;;
  *)
    print_usage_and_exit
  ;;
esac
