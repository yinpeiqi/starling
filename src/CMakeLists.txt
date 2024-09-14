# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 17)

set(STARLING_DEFAULT page_search_starling.cpp)
set(DISKANN beam_search_with_cache.cpp)
set(PIPELINED_PQ pipeline_PQ_filter.cpp)

if(MSVC)
    add_subdirectory(dll)
else()
    #file(GLOB CPP_SOURCES *.cpp)
    set(CPP_SOURCES ann_exception.cpp aux_utils.cpp distance.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp natural_number_map.cpp
        natural_number_set.cpp memory_mapper.cpp partition_and_pq.cpp
        pq_flash_index.cpp logger.cpp utils.cpp page_search.cpp visit_freq.cpp
        generate_node_dist.cpp index_engine_search.cpp index_engine_store.cpp
        file_io_manager.cpp range_search.cpp page_search_impl/${PIPELINED_PQ})
    add_library(${PROJECT_NAME} ${CPP_SOURCES})
    add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
    target_link_libraries(${PROJECT_NAME} tbb)
endif()
install()
