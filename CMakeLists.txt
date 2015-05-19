cmake_minimum_required(VERSION 2.6)
project(hdmarker)


#for now we always assume X86!
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # using regular Clang or AppleClang
    set(ALL_BUT_LINKER_CONFLICT_FLAGS "-O3 -g -mssse3 -Wall")
    set(CMAKE_SHARED_LINKER_FLAGS "-O3 -g -mssse3 -Wall")
    add_definitions(-DCOMPILER_CLANG_X86)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "-Wall -g -O1 -std=c++11 -Wall")
    add_definitions(-DCOMPILER_GCC_X86)
    add_definitions(-DCOMPILER_GCC)
endif()

if (APPLE)
    add_definitions(-DMACHACKS)
    message(WARNING "APPLE system detected, disabling timing/benchmarking")
    #for cmake < 3.0
    if (NOT DEFINED CMAKE_MACOSX_RPATH)
        set(CMAKE_MACOSX_RPATH 0)
    endif()
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/usr/include/valgrind/")

add_library(hdmarker SHARED hdmarker.cpp gridstore.cpp timebench.cpp)

target_link_libraries(hdmarker ${OpenCV_LIBS})

add_executable(hdmarker_generate marker_gen.cpp)
add_executable(hdmarker_test test.cpp)
#add_executable(hdmarker_scaletest testscales.cpp)
target_link_libraries(hdmarker_generate ${OpenCV_LIBS} hdmarker)
target_link_libraries(hdmarker_test ${OpenCV_LIBS} hdmarker)
#target_link_libraries(hdmarker_scaletest ${OpenCV_LIBS} hdmarker)

configure_file(hdmarker.pc.in hdmarker.pc @ONLY)

install(FILES hdmarker.pc DESTINATION lib/pkgconfig)
install(FILES hdmarker.hpp gridstore.hpp DESTINATION include/hdmarker)
install(TARGETS hdmarker_test hdmarker_generate hdmarker
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib/static)