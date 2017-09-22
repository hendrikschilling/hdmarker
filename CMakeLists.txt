cmake_minimum_required(VERSION 2.8.12)
project(hdmarker)

#####################################################
## INIT
#####################################################
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/flexdeplists")
message("CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
#set(VAD_VERBOSE TRUE)
# TODO integrate
include(VAD_Wrappers)
# TODO integrate into VAD framework
include(VigraAddDep)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(CMakeParseArguments)

#####################################################
## ENVIRONMENT
#####################################################
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  add_definitions(-DCOMPILER_CLANG_X86)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -fno-omit-frame-pointer -fPIC -std=c++11 -march=native -fopenmp")
  add_definitions(-DCOMPILER_GCC_X86)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /noOy /W2 /EHsc /openmp")
endif()

# default to static on win shared on others
if (NOT DEFINED BUILD_SHARED_LIBS)
  if (WIN32)
    option(BUILD_SHARED_LIBS "build shared libraries" OFF)
  else()
    option(BUILD_SHARED_LIBS "build shared libraries" ON)
  endif()
endif()

#####################################################
## COMPILE
#####################################################
if (BUILD_SHARED_LIBS)
  add_definitions(-D${BUILD_SHARED_LIBS})
endif()

include_directories(src/lib/)

add_library(hdmarker src/lib/hdmarker.cpp src/lib/gridstore.cpp src/lib/timebench.cpp src/lib/subpattern.cpp)
vad_link(hdmarker ceres opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_calib3d)
vad_headers(hdmarker src/lib/hdmarker.hpp src/lib/gridstore.hpp src/lib/subpattern.hpp)
#target_link_libraries(hdmarker PUBLIC ${${_PNU}_LIB})
#target_link_libraries(hdmarker PRIVATE ${${_PNU}_PRIVATE_LIB})
#list(APPEND ${_PNU}_EXPORT_LIBS hdmarker)

add_executable(hdmarker_test src/bin/test.cpp)
target_link_libraries(hdmarker_test hdmarker)

add_dependencies(hdmarker hdmarker-header-export)

set(CONF_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/include)
#target_include_directories(ucalib INTERFACE)
export(TARGETS hdmarker FILE ${hdmarker_BINARY_DIR}/hdmarkerTargets.cmake)
#export(PACKAGE hdmarker)

include(cmake/GenConfig.cmake)

#####################################################
## INSTALL
#####################################################
#dep_lists_export_local()
