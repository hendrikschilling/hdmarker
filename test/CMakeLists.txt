cmake_minimum_required(VERSION 2.6)
project(hdmarker-buildtest)

list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -Wall -fopenmp")

find_package(hdmarker REQUIRED)

foreach(file ${HDMARKER_INCLUDE_DIRS})
  message("include: ${file}")
endforeach()
foreach(file ${HDMARKER_LIBRARY_DIRS})
  message("link dir: ${file}")
endforeach()
foreach(file ${HDMARKER_LIBRARIES})
  message("link: ${file}")
endforeach()

include_directories(${HDMARKER_INCLUDE_DIRS})
link_directories(${HDMARKER_LIBRARY_DIRS})

add_executable(buildtest buildtest.cpp)

target_link_libraries(buildtest ${HDMARKER_LIBRARIES})
