cmake_minimum_required(VERSION 2.8.12)
project(hdmarker_eval)
string(TOUPPER ${PROJECT_NAME} P_N_U)

find_package(hdmarker REQUIRED)
find_package(ucalib REQUIRED)
find_package(MetaMat REQUIRED)


set(CMAKE_CXX_FLAGS "-Wall -g -O3 -std=c++11 -Wall")

include_directories(${HDMARKER_INCLUDE_DIRS})
include_directories(${UCALIB_INCLUDE_DIRS})
include_directories(${METAMAT_INCLUDE_DIRS})

link_directories(${HDMARKER_LIBRARY_DIRS})
link_directories(${UCALIB_LIBRARY_DIRS})
link_directories(${METAMAT_LIBRARY_DIRS})

add_executable(eval eval.cpp)
target_link_libraries(eval ${HDMARKER_LIBRARIES} ${UCALIB_LIBRARIES} ${METAMAT_LIBRARIES})
