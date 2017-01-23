cmake_minimum_required(VERSION 2.8.12)
project(hdmarker)


#####################################################
## INIT
#####################################################
include(external/flexdeplists/FlexDepLists.cmake)
dep_lists_init()

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
## DEPENDENCIES
#####################################################
#metamat
dep_lists_append(OpenCV OpenCV_INCLUDE_DIRS "" OpenCV_LIBS)
dep_lists_append(Ceres)


dep_lists_pkg_search()
set(${_PNU}_HEADERS src/lib/hdmarker.hpp src/lib/subpattern.hpp src/lib/gridstore.hpp)
dep_lists_prepare_env()

message("hdmnarker should use oipencv inc: ${OpenCV_INCLUDE_DIRS}")

#####################################################
## COMPILE
#####################################################
add_library(hdmarker src/lib/hdmarker.cpp src/lib/gridstore.cpp src/lib/timebench.cpp src/lib/subpattern.cpp)
target_link_libraries(hdmarker PUBLIC ${${_PNU}_LIB})
target_link_libraries(hdmarker PRIVATE ${${_PNU}_PRIVATE_LIB})
list(APPEND ${_PNU}_EXPORT_LIBS hdmarker)

add_dependencies(hdmarker hdmarker-header-export)

#####################################################
## INSTALL
#####################################################
#dep_lists_export_local()
