list(APPEND _VAD_AUTODEP_PROJECT_LIST "${CMAKE_PROJECT_NAME}")

if(VAD_HendriksWrappers_Included)
    return()
endif()

# Mark as included.
set(VAD_HendriksWrappers_Included YES)

# FIXME list can be ambiguous (e.g. fnmatch -> metamat!)
set(_TARGET_PACKAGE_LIST 
cliini;
cliini;
cliini-cpp;
cliini;
METAMAT::METAMAT;
MetaMat;
ceres;
Ceres;
HDF5::HDF5;
HDF5;
hdf5_cpp;
HDF5;
OPENCV::OPENCV;
OpenCV;
boost_system;
Boost;
boost_filesystem;
Boost;
unit_test_framework;
Boost;
hdmarker;
hdmarker;
fnmatch;
fnmatch;
FNMATCH::FNMATCH;
fnmatch;
metamat;
MetaMat;
Boost::system;
Boost;
Boost::filesystem;
Boost;
Boost::unit_test_framework;
Boost;
ceres_hack;
Ceres;
ceres_hack3;
Ceres;
opencv_core;
OpenCV;
opencv_imgproc;
OpenCV;
opencv_highgui;
OpenCV;
opencv_shape;
OpenCV;
opencv_objdetect;
OpenCV;
opencv_calib3d;
OpenCV;
mm-mesh;
mm-mesh;
ucalib;
ucalib;
Qt5::Widgets;
Qt5Widgets;
Qt5::Core;
Qt5Core;
Qt5::Network;
Qt5Network;
Qt5::Gui;
Qt5Gui;
clif;
clif;
clif-qt;
clif;
vigraimpex;
Vigra;
snoptimizer BRDF-SNOptimizer)

# TODO check system dependency!
function(vad_autodep_pkg _PKG_NAME _REQ_NAME)
  if (VAD_BUILD_${_PKG_NAME}_FROM_GIT)
    vigra_add_dep(${_PKG_NAME} LIVE)
  else()
    # check for system dependency
    
    vigra_add_dep(${_PKG_NAME} SYSTEM QUIET)
    vad_dep_satisfied(${_PKG_NAME})
    
    #no system dep found
    if(NOT VAD_DEP_${_PKG_NAME}_SATISFIED)
      list(FIND _VAD_AUTODEP_MISSING_PKGS ${_PKG_NAME} _IDX_PKG_NAME)
      if (_IDX_PKG_NAME EQUAL -1)
        set(_VAD_AUTODEP_MISSING_PKGS "${_VAD_AUTODEP_MISSING_PKGS};${_PKG_NAME}" CACHE STRING "" FORCE)
        set(_VAD_AUTODEP_REQ_${_PKG_NAME} "${_REQ_NAME}" CACHE STRING "" FORCE)
      endif()
    endif()
  endif()
endfunction()

function(vad_headers _TGT)
  set(_vad_headers_PREFIX)
  cmake_parse_arguments(_vad_headers "" "PREFIX" "" ${ARGN})

  if (_vad_headers_PREFIX)
    set(_HEADER_PREFIX ${_vad_headers_PREFIX})
  else()
    string(TOLOWER ${_TGT} _HEADER_PREFIX)
  endif()
  
  #generate at configure time
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/${_HEADER_PREFIX})
                 
  set(_HEADER_DEPLIST)
  foreach(_H ${_vad_headers_UNPARSED_ARGUMENTS})
    get_filename_component(_H_ABS ${_H} ABSOLUTE)
    get_filename_component(_H_NAME ${_H} NAME)
  
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/${_HEADER_PREFIX}/${_H_NAME}
                       COMMAND ${CMAKE_COMMAND} -E copy ${_H_ABS} ${CMAKE_CURRENT_BINARY_DIR}/include/${_HEADER_PREFIX}/${_H_NAME}
                       DEPENDS ${_H_ABS})
    list(APPEND _HEADER_DEPLIST ${CMAKE_CURRENT_BINARY_DIR}/include/${_HEADER_PREFIX}/${_H_NAME})
  endforeach()
  
  #only for local source
  # TODO make this an option?
  
  add_custom_target(${_TGT}-header-export ALL DEPENDS ${_HEADER_DEPLIST}) 
  #set_target_properties(${_TGT}-header-export PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/include/${_HEADER_PREFIX})
  # TODO public / private option?
  set_property(TARGET ${_TGT} APPEND PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/include)
  set_property(TARGET ${_TGT} APPEND PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/include/${_HEADER_PREFIX})
  set_property(TARGET ${_TGT} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/include)
  add_dependencies(${_TGT} ${_TGT}-header-export)
endfunction()

# FIXME for testing ... only PUBLIC, no Debug/Release, etc...
function(vad_link TARGT)

  foreach(L ${ARGN})
    if(TARGET ${L})
      target_link_libraries(${TARGT} PUBLIC ${L})
    else()   
      list(FIND _TARGET_PACKAGE_LIST ${L} _IDX_TGT)
      if (_IDX_TGT EQUAL -1)
        message(FATAL_ERROR "target ${L} does no exist and no corresponding package was found in")
      endif()
      math(EXPR _IDX_PKG "${_IDX_TGT} + 1")
      list(GET _TARGET_PACKAGE_LIST ${_IDX_PKG} _PKG_NAME)
            
      vad_autodep_pkg(${_PKG_NAME} ${CMAKE_PROJECT_NAME})
      
      if (TARGET ${L})
        target_link_libraries(${TARGT} PUBLIC ${L})
      else()
        # TODO check if BUILD_FROM  GIT is true if yes abort!
        message("requested target ${TARGT} PUBLIC ${L} not found!")
      endif()
    endif()
  endforeach()
endfunction()

# TODO hacky alternative to manually calling this function:
# use variable_watch of CMAKE_BACKWARDS_COMPATIBILITY (which is called by cmake at the end of configure :-D )
function(vad_auto_deps_show)
  list(FIND _VAD_AUTODEP_PROJECT_LIST ${CMAKE_PROJECT_NAME} IDX)
  if(IDX EQUAL -1)
    message(FATAL_ERROR "could not find this project ${CMAKE_PROJECT_NAME} in _VAD_AUTODEP_PROJECT_LIST, did you include VAD_HendriksWrappers.cmake?")
  endif()
  list(REMOVE_ITEM _VAD_AUTODEP_PROJECT_LIST ${CMAKE_PROJECT_NAME})
  if (NOT _VAD_AUTODEP_PROJECT_LIST)
    #recursively iterate all dependencies

    set(_unhandeld_missing true)
    while(_unhandeld_missing)
      set(_unhandeld_missing false)
      foreach(PKG ${_VAD_AUTODEP_MISSING_PKGS})
        if (NOT _handled_missing_${PKG})
          set(_handled_missing_${PKG} true)
          set(_unhandeld_missing true)
          # FIXME this is a direct copy, create a shared function/macro
          find_file(VAD_${PKG}_FILE VAD_${PKG}.cmake ${CMAKE_MODULE_PATH})
          if(VAD_${PKG}_FILE)
            message(STATUS "Dep search: VAD file 'VAD_${PKG}.cmake' was found at '${VAD_${PKG}_FILE}'. The VAD file will now be parsed.")
            vad_reset_hooks()
            include(${VAD_${PKG}_FILE})
            vad_deps(${PKG})
            vad_reset_hooks()
          else()
            message("Ohohhh no file  ${VAD_${PKG}_FILE} found!")
          endif()
        endif()
      endforeach()
    endwhile()
  
    if (_VAD_AUTODEP_MISSING_PKGS)
      message("missings PKGs: ")
      foreach(PKG ${_VAD_AUTODEP_MISSING_PKGS})
        message("  - ${PKG} (required by ${_VAD_AUTODEP_REQ_${PKG}})")
        option(VAD_BUILD_${PKG}_FROM_GIT "integrate LIVE source into project" off)
      endforeach()
    
      #cleanup
      set(_VAD_AUTODEP_MISSING_PKGS "" CACHE STRING "" FORCE)
      
      # show possible live packages
      # TODO also allow packages found in system to be built from source!
      message("")
      message("unfullfilled dependencies - either install and point cmake to the missing depdendencies or use VAD_BUILD_***_FROM_GIT to integrate the respective package into the project.")
      message(FATAL_ERROR "")
    endif()
  endif()
  
  #cleanup
  set(_VAD_AUTODEP_MISSING_PKGS "" CACHE STRING "" FORCE)
endfunction()
