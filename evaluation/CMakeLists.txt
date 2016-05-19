cmake_minimum_required(VERSION 2.8.12)
project(hdmarker_eval)

include(../external/flexdeplists/FlexDepLists.cmake)

dep_lists_append(hdmarker)
dep_lists_append(ucalib)
dep_lists_append(MetaMat)

dep_lists_pkg_search()


set(CMAKE_CXX_FLAGS "-Wall -g -O3 -std=c++11 -Wall")

dep_lists_prepare_env()

add_executable(eval eval.cpp)
target_link_libraries(eval ${HDMARKER_EVAL_LIBRARIES})
