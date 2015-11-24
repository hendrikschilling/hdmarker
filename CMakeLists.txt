cmake_minimum_required(VERSION 2.8.12)
project(hdmarker)
string(TOUPPER ${PROJECT_NAME} P_N_U)

#for now we always assume X86!
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # using regular Clang or AppleClang
    set(ALL_BUT_LINKER_CONFLICT_FLAGS "-O3 -g -mssse3 -Wall")
    set(CMAKE_SHARED_LINKER_FLAGS "-O3 -g -mssse3 -Wall")
    add_definitions(-DCOMPILER_CLANG_X86)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "-Wall -g -O3 -std=c++11 -Wall -flto=8 -fopenmp")
    add_definitions(-DCOMPILER_GCC_X86)
    add_definitions(-DCOMPILER_GCC)
endif()

if (APPLE)
    add_definitions(-DMACHACKS)
    message(WARNING "APPLE system detected, disabling timing/benchmarking")
    #for cmake < 3.0
    if (NOT DEFINED CMAKE_MACOSX_RPATH)
        set(CMAKE_MACOSX_RPATH 1)
    endif()
endif()


#ceres
list(APPEND HDMARKER_PRIVATE_PKG Ceres)
list(APPEND HDMARKER_PRIVATE_PKG_INC CERES_INCLUDE_DIRS)
list(APPEND HDMARKER_PRIVATE_PKG_LIB CERES_LIBRARIES)
#not existing according to ceres ...Config.cmake
#list(APPEND HDMARKER_PKG_LINK CERES_LIBRARY_DIRS)

#OpenCV
list(APPEND HDMARKER_PKG OpenCV)
list(APPEND HDMARKER_PKG_INC OpenCV_INCLUDE_DIRS)
list(APPEND HDMARKER_PKG_LIB OpenCV_LIBS)

foreach(PACKAGE ${HDMARKER_PKG})
  find_package(${PACKAGE} REQUIRED)
endforeach()
foreach(INCLUDE ${HDMARKER_PKG_INC})
  list(APPEND HDMARKER_DEPENDS_INC ${${INCLUDE}})
endforeach()
foreach(LIBDIR ${HDMARKER_PKG_LINK})
  list(APPEND HDMARKER_DEPENDS_LINK ${${LIBDIR}})
endforeach()
foreach(LIB ${HDMARKER_PKG_LIB})
  list(APPEND HDMARKER_DEPENDS_LIB ${${LIB}})
endforeach()

foreach(PACKAGE ${HDMARKER_PRIVATE_PKG})
  find_package(${PACKAGE} REQUIRED)
endforeach()
foreach(INCLUDE ${HDMARKER_PRIVATE_PKG_INC})
  list(APPEND HDMARKER_PRIVATE_DEPENDS_INC ${${INCLUDE}})
endforeach()
foreach(LIBDIR ${HDMARKER_PRIVATE_PKG_LINK})
  list(APPEND HDMARKER_PRIVATE_DEPENDS_LINK ${${LIBDIR}})
endforeach()
foreach(LIB ${HDMARKER_PRIVATE_PKG_LIB})
  list(APPEND HDMARKER_PRIVATE_DEPENDS_LIB ${${LIB}})
endforeach()

#put headers in right place
list(APPEND HDMARKER_PROJECT_HEADERS hdmarker.hpp subpattern.hpp gridstore.hpp)
foreach(header ${HDMARKER_PROJECT_HEADERS})
  get_filename_component(_FILE ${header} NAME)
  configure_file(${header} "include/hdmarker/${_FILE}" COPYONLY)
endforeach()

include_directories(${HDMARKER_DEPENDS_INC} ${HDMARKER_PRIVATE_DEPENDS_INC})
link_directories(${HDMARKER_DEPENDS_LINK} ${HDMARKER_PRIVATE_DEPENDS_LINK})

add_library(hdmarker SHARED hdmarker.cpp gridstore.cpp timebench.cpp subpattern.cpp)
add_library(hdmarker_subpattern SHARED subpattern.cpp)

target_link_libraries(hdmarker PUBLIC ${HDMARKER_DEPENDS_LIB})
list(APPEND HDMARKER_PROJECT_LIBRARIES hdmarker)

target_link_libraries(hdmarker_subpattern PUBLIC ${HDMARKER_DEPENDS_LIB})
target_link_libraries(hdmarker_subpattern PRIVATE ${HDMARKER_PRIVATE_DEPENDS_LIB})
target_link_libraries(hdmarker PUBLIC hdmarker_subpattern)
list(APPEND HDMARKER_PROJECT_LIBRARIES hdmarker_subpattern)

add_executable(hdmarker_generate marker_gen.cpp)
target_link_libraries(hdmarker_generate ${HDMARKER_PROJECT_LIBRARIES})

add_executable(hdmarker_test test.cpp)
target_link_libraries(hdmarker_test ${HDMARKER_PROJECT_LIBRARIES})

install(FILES hdmarker.hpp subpattern.hpp gridstore.hpp DESTINATION include/hdmarker)
install(TARGETS hdmarker hdmarker_subpattern
            EXPORT ${PROJECT_NAME}
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
install(TARGETS hdmarker_test hdmarker_generate 
            RUNTIME DESTINATION bin)
#export


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
include(CMakePackageConfigListHelpers)

#############################
# local global
#############################
set(CMAKECONFIG_PKG ${${P_N_U}_PKG})
set(CMAKECONFIG_PKG_INC ${${P_N_U}_PKG_INC})
set(CMAKECONFIG_PKG_LINK ${${P_N_U}_PKG_LINK})
set(CMAKECONFIG_PKG_LIB ${${P_N_U}_PKG_LIB})

set(CMAKECONFIG_INC "include")
set(CMAKECONFIG_LIB ${${P_N_U}_PROJECT_LIBRARIES})


#############################
# export local
#############################
set(CMAKECONFIG_CMAKE_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKECONFIG_LINK ${CMAKE_CURRENT_BINARY_DIR})

set(INSTALL_PREFIX_BACKUP ${CMAKE_INSTALL_PREFIX})
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
configure_package_config_file(cmake/projectConfig.cmake.in
                              "${CMAKECONFIG_CMAKE_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION "${CMAKECONFIG_CMAKE_DIR}"
                              PATH_VARS CMAKECONFIG_PKG CMAKECONFIG_PKG_INC CMAKECONFIG_PKG_LINK CMAKECONFIG_PKG_LIB CMAKECONFIG_INC CMAKECONFIG_LINK CMAKECONFIG_LIB)
set(CMAKE_INSTALL_PREFIX ${INSTALL_PREFIX_BACKUP})


#############################
# make install export
#############################
set(CMAKECONFIG_CMAKE_DIR "lib/cmake/${PROJECT_NAME}")

set(CMAKECONFIG_LINK "lib")

configure_package_config_file(cmake/projectConfig.cmake.in
                              "config/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION "${CMAKECONFIG_CMAKE_DIR}"
                              PATH_VARS CMAKECONFIG_PKG CMAKECONFIG_PKG_INC CMAKECONFIG_PKG_LINK CMAKECONFIG_PKG_LIB CMAKECONFIG_INC CMAKECONFIG_LINK CMAKECONFIG_LIB)


#################################

# for the installed package
#set(${P_N_U}_INSTALL_INCLUDE_DIR "include" "include/hdmarker")
#set(${P_N_U}_INSTALL_LIBRARY_DIR "lib")
#set(${P_N_U}_INSTALL_LIBRARIES ${${P_N_U}_LINK_LIBRARIES})
#set(${P_N_U}_INSTALL_CMAKE_DIR "lib/cmake/clif")
#configure_package_config_file(cmake/projectConfig.cmake.in
#                              "config/${PROJECT_NAME}Config.cmake"
#                              INSTALL_DESTINATION ${${P_N_U}_INSTALL_CMAKE_DIR}
#                              PATH_VARS ${P_N_U}_INSTALL_INCLUDE_DIR ${P_N_U}_INSTALL_LIBRARY_DIR ${P_N_U}_INSTALL_LIBRARIES ${P_N_U}_INSTALL_CMAKE_DIR)

# Install the clifConfig.cmake and clifConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}/config/${PROJECT_NAME}Config.cmake"
  DESTINATION ${CMAKECONFIG_CMAKE_DIR})

#export targets for installation
install(EXPORT ${PROJECT_NAME} DESTINATION ${CMAKECONFIG_CMAKE_DIR})
install(FILES ${${P_N_U}_FULLPATH_HEADERS} DESTINATION include/${PROJECT_NAME} )

#local export into binary dir (may be used with cmake .. -DCMAKE_PREFIX_PATH=<...> or with cmake-gui)
if (${CMAKE_VERSION} VERSION_LESS 3.0.0)
  #TODO use a generated list of targets!
  export(TARGETS hdmarker hdmarker_subpattern FILE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.cmake)
else()
  message(${CMAKE_VERSION})
  export(EXPORT ${PROJECT_NAME} FILE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.cmake)
endif()

#export into cmakes local package cache - should be directly visible to cmake of local user
export(PACKAGE ${PROJECT_NAME})
