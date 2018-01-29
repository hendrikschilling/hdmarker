#include(VigraAddDep)

set(GIT_REPO "https://github.com/opencv/opencv.git")

function(vad_system)
  message("run VAD_SYSTEM for OpenCV")
  vad_system_default(${ARGN})
endfunction()

function(vad_live)
  message("run VAD_LIVE for OpenCV")
  
  git_clone(OpenCV)
  # FIXME repair CMAKE_BINARY_DIR for add_subidred opencv build?
  
  file(READ "${VAD_EXTERNAL_ROOT}/OpenCV/cmake/OpenCVExtraTargets.cmake" FILECONTENT)
  string(REPLACE "ADD_CUSTOM_TARGET(uninstall " "ADD_CUSTOM_TARGET(uninstall-opencv " FILECONTENT ${FILECONTENT})
  string(REPLACE "set_target_properties(uninstall " "set_target_properties(uninstall-opencv " FILECONTENT ${FILECONTENT})
  file(WRITE "${VAD_EXTERNAL_ROOT}/OpenCV/cmake/OpenCVExtraTargets.cmake" ${FILECONTENT})
  
  set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(BUILD_WITH_STATIC_CRT OFF CACHE BOOL "" FORCE)
  
  # example of disabling something
  #set(BUILD_opencv_flann OFF CACHE BOOL "" FORCE)
  
  #not workgin as intended...
  #cmake_policy(VERSION 2.8)
  #cmake_policy(SET CMP0022 OLD)
  add_subdirectory("${VAD_EXTERNAL_ROOT}/OpenCV" "${CMAKE_BINARY_DIR}/external/OpenCV")
  
  add_library(OPENCV::OPENCV INTERFACE IMPORTED)
  
  message("cmake binary dir: ${CMAKE_BINARY_DIR}")
  
  file(GLOB _OPENCV_MODULES_LIST "${VAD_EXTERNAL_ROOT}/OpenCV/modules/*/")
  message("opencv modules: ")
  foreach(D ${_OPENCV_MODULES_LIST})
    if(IS_DIRECTORY ${D})
      get_filename_component(D_NAME ${D} NAME)
      if(TARGET "opencv_${D_NAME}")
        list(APPEND _OPENCV_TGTS "opencv_${D_NAME}")
        list(APPEND _OPENCV_INC "${D}/include")
        set_target_properties(opencv_${D_NAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${D}/include;${CMAKE_BINARY_DIR};${VAD_EXTERNAL_ROOT}/OpenCV/include")
        message("module: ${D_NAME}")
      else()
        message("skipping module ${D_NAME} (target opencv_${D_NAME} does not exist!)")
      endif()
    endif()
  endforeach()
  
  #set_target_properties(OPENCV::OPENCV PROPERTIES INTERFACE_LINK_LIBRARIES "${_OPENCV_TGTS}")
  #set_target_properties(OPENCV::OPENCV PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${VAD_EXTERNAL_ROOT}/OpenCV/include;${CMAKE_BINARY_DIR};${_OPENCV_INC}")
  
  # FIXME 
  set(OpenCV_INCLUDE_DIRS "${VAD_EXTERNAL_ROOT}/OpenCV/include;${_OPENCV_INC};${CMAKE_BINARY_DIR}" CACHE STRING "" FORCE)
  set(OpenCV_LIBS "${_OPENCV_TGTS}" CACHE STRING "" FORCE)
  set(OPENCV_FOUND TRUE CACHE BOOL "" FORCE)

endfunction()
