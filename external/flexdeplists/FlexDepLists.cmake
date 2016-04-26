#####################################################
## VARIABLE DOCUMENTATION
#####################################################
#Meta:
#  PNU              - project name in uppercase - prefixed to (nearly) all variables. e.g. ${PNU}_PKG, etc.
#
#Dependencies (packages and respective cmake variables)
#  _PKG               - required cmake package names - passed to find_package() - will also be included from ...Config.cmake
#  _PKG_OPT           - optional cmake package names - passed to find_package() - will also be included from ...Config.cmake, ..._USE_${..._PKG_OPT} will be set to true if optional pkg is found
# _PRIVATE_PKG
# _PRIVATE_PKG_OPT
###
#  _PKG_INC           - variable names to add to include_directories (from the respecive find_package() calls)
#  _PKG_LINK          - variable names to add to link_directories
#  _PKG_LIB           - variable names to add to link_libraries
#
#actual directories/libraries from dependencies
#  _INC          - actual content from _PKG_INC vars
#  _LINK         - link dirs
#  _LIB          - libs

#  _PRIVATE_INC  - actual content from _PKG_INC vars
#  _PRIVATE_LINK - link dirs
#  _PRIVATE_LIB  - libs
#
#  _EXPORT_LIBS       - targets (libraries) for installation, export and to put into _LIBRARIES into ..Config.cmake
#  _EXPORT_BINS       - targets (executables) to install
#
#
#  _BUILD_TYPE        - SHARED or "" (set depending on compiler)
#
#  _PROJECT_HEADERS
#  _PROJECT_LIBRARIES

# POSSIBLE PROBLEMS
# 
# - package found are marked with FDP_HAVE_SEARCHED_${PACKAGE} checking only if
#   FDP_HAVE_SEARCHED_${PACKAGE}_COMPONENTS are the same, changes in environment
#   will be ignored!

#set(FDP_VERBOSE 3)

# used to include CMakePackageConfigListHelpers.cmake and projectConfig.cmake.in
set(FLEXDEPLISTS_DIR ${CMAKE_CURRENT_LIST_DIR})

include(CMakeParseArguments)

if (NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

function(dep_lists_pkg_found PKG RET)
  string(TOLOWER ${PKG} LOW)
  string(TOUPPER ${PKG} UP)
  
  set(${RET} FALSE PARENT_SCOPE)
  
  if (${PKG}_FOUND)
    set(${RET} TRUE PARENT_SCOPE)
      return()
  endif()
  if (${LOW}_FOUND)
    set(${RET} TRUE PARENT_SCOPE)
      return()
  endif()
  if (${UP}_FOUND)
    set(${RET} TRUE PARENT_SCOPE)
      return()
  endif()
  
  if (${PNU}_${PKG}_FOUND_INDICATOR)
    if (${${PNU}_${PKG}_FOUND_INDICATOR})
      set(${RET} TRUE PARENT_SCOPE)
      return()
    endif()
  endif()
endfunction()

macro(dep_lists_clean_list LIST)
  if (DEFINED ${LIST})
    list(REMOVE_DUPLICATES ${LIST})
  endif()
endmacro()

macro(dep_lists_msg_info MSG)
  if (FDP_VERBOSE)
    if (2 LESS ${FDP_VERBOSE})
      message(${MSG})
    endif()
  endif()
endmacro()

#like cmake_parse_arguments but keeps empty args
macro(dep_lists_parse _dlp_NAME _dlp_OPTS _dlp_SINGLE _dlp_MULTI)
  cmake_policy(SET CMP0011 NEW)
  if (POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
  endif()

  set(_dlp_ARGS "")
  set(_dlp_ARGLIST "${ARGN}")
  
  foreach(_dlp_ARG IN LISTS _dlp_ARGLIST)
    if("${_dlp_ARG}" STREQUAL "")
      list(APPEND _dlp_ARGS "DEP_LISTS_PARSE_EMPTY")
    else()
      list(APPEND _dlp_ARGS ${_dlp_ARG})
    endif()
  endforeach()
  
  cmake_parse_arguments("${_dlp_NAME}" "${_dlp_OPTS}" "${_dlp_SINGLE}" "${_dlp_MULTI}" ${_dlp_ARGS})
  
  set(_dlp_ARGS ${dep_lists_append_UNPARSED_ARGUMENTS})
  set(dep_lists_append_UNPARSED_ARGUMENTS "")
  foreach(_dlp_ARG ${_dlp_ARGS})
    if("${_dlp_ARG}" STREQUAL "DEP_LISTS_PARSE_EMPTY")
      list(APPEND dep_lists_append_UNPARSED_ARGUMENTS "")
    else()
      list(APPEND dep_lists_append_UNPARSED_ARGUMENTS ${_dlp_ARG})
    endif()
  endforeach()
endmacro()

#if components are specified those are serialized into ${PNU}_PKG_COMPONENTS
macro(dep_lists_check_find PACKAGE RET PNU)
  if (FDP_HAVE_SEARCHED_${PACKAGE} AND "${FDP_HAVE_SEARCHED_${PACKAGE}_COMPONENTS}" STREQUAL "${${PNU}_${PACKAGE}_COMPONENTS}")
    #message("already searched for ${PACKAGE}")
    if(FDP_HAVE_FOUND_${PACKAGE})
      set(${RET} TRUE)
    else()
      set(${RET} FALSE)
    endif()
  else()
    dep_lists_msg_info("search ${PACKAGE}")
    set(FDP_HAVE_SEARCHED_${PACKAGE} true)
    set(FDP_HAVE_SEARCHED_${PACKAGE}_COMPONENTS ${${PNU}_${PACKAGE}_COMPONENTS})
    
    if (${PNU}_${PACKAGE}_COMPONENTS)
      find_package(${PACKAGE} QUIET COMPONENTS ${${PNU}_${PACKAGE}_COMPONENTS} ${${PNU}_${PACKAGE}_FIND_FLAGS})
    else()
      find_package(${PACKAGE} QUIET ${${PNU}_${PACKAGE}_FIND_FLAGS})
    endif()
    
    string(TOLOWER ${PACKAGE} PKG_LOW)
    
    #check if pkg was found (various conventions...)
    dep_lists_pkg_found(${PACKAGE} FOUND)
    
    if (NOT FOUND)
      list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/find/${PKG_LOW})
      
      if (${PNU}_${PACKAGE}_COMPONENTS)
        find_package(${PACKAGE} QUIET COMPONENTS ${${PNU}_${PACKAGE}_COMPONENTS} ${${PNU}_${PACKAGE}_FIND_FLAGS})
      else()
        find_package(${PACKAGE} QUIET ${${PNU}_${PACKAGE}_FIND_FLAGS})
      endif()
      
      dep_lists_pkg_found(${PACKAGE} FOUND)
    endif()
    
    if (FOUND)
      dep_lists_msg_info("found ${PACKAGE} - found")
      set(${RET} TRUE)
      list(APPEND ${PNU}_FEATURES ${RET})
      
      if (${PNU}_${PACKAGE}_COMPONENTS)
        foreach(COMPONENT ${${PNU}_${PACKAGE}_COMPONENTS})
          list(APPEND ${PNU}_PKG_COMPONENTS "${PNU}_${PACKAGE}_COMPONENTS")
          list(APPEND ${PNU}_PKG_COMPONENTS "${COMPONENT}")
        endforeach()
      endif()
      
      
      
      set(FDP_HAVE_FOUND_${PACKAGE} true)
    else()
      dep_lists_msg_info("${PACKAGE} - missing")
      set(${RET} FALSE)
      set(FDP_HAVE_FOUND_${PACKAGE} false)
    endif()
  endif()
endmacro()

macro(dep_lists_exec_find PNU PKG SUCC FAIL)
  cmake_policy(SET CMP0011 NEW)
  if (POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
  endif()

  string(TOUPPER ${PKG} _FDP_PKG_UP)
  dep_lists_check_find(${PKG} ${PNU}_WITH_${_FDP_PKG_UP} ${PNU})
  if (${PNU}_WITH_${_FDP_PKG_UP})
    if (NOT "${SUCC}" STREQUAL "")
      list(APPEND ${SUCC} ${PKG})
    endif()
  else()
    if (NOT "${FAIL}" STREQUAL "")
      list(APPEND ${FAIL} ${PKG})
    endif()
  endif()

endmacro(dep_lists_exec_find)

macro(dep_lists_pkg_search)
  
  if (ARGV0)
    set(_FDP_PNU ${ARGV0})
  else()
    string(TOUPPER ${PROJECT_NAME} _FDP_PNU)
  endif()

  #message("")
  #message("searching dependencies for ${_FDP_PNU}:")
  
  foreach(PACKAGE ${${_FDP_PNU}_PKG_OPT})
    dep_lists_exec_find(${_FDP_PNU} ${PACKAGE} ${_FDP_PNU}_EXPORT_DEPS ${_FDP_PNU}_MISSING_OPTIONAL)
  endforeach()
  
  foreach(PACKAGE ${${_FDP_PNU}_PRIVATE_PKG_OPT})
    dep_lists_exec_find(${_FDP_PNU} ${PACKAGE} ""                      ${_FDP_PNU}_MISSING_OPTIONAL)
  endforeach()
  
  dep_lists_clean_list(${_FDP_PNU}_MISSING_OPTIONAL)
  
  foreach(PACKAGE ${${_FDP_PNU}_PKG})
    dep_lists_exec_find(${_FDP_PNU} ${PACKAGE} ${_FDP_PNU}_EXPORT_DEPS ${_FDP_PNU}_MISSING_REQUIRED)
  endforeach()
  
  foreach(PACKAGE ${${_FDP_PNU}_PRIVATE_PKG})
    dep_lists_exec_find(${_FDP_PNU} ${PACKAGE} ""                      ${_FDP_PNU}_MISSING_REQUIRED)
  endforeach()
    
  dep_lists_clean_list(${_FDP_PNU}_MISSING_REQUIRED)
  
  foreach(VAR ${${_FDP_PREFIX}_EXPORT_VARS})
    list(APPEND ${_FDP_PREFIX}_EXPORT_VARS_VALUES "VAR" "${VAR}")
  endforeach()
    
  if (${_FDP_PNU}_MISSING_OPTIONAL)
    message("${BoldRed}missing OPTIONAL packages for ${_FDP_PNU}:")
    foreach(PACKAGE ${${_FDP_PNU}_MISSING_OPTIONAL}${ColourReset})
      message("   ${BoldRed}${PACKAGE}${ColourReset}")
    endforeach()
  endif()
  
  if (${_FDP_PNU}_MISSING_REQUIRED)
    message("${BoldRed}missing REQUIRED packages for ${_FDP_PNU}::")
    foreach(PACKAGE ${${_FDP_PNU}_MISSING_REQUIRED}${ColourReset})
      message("   ${BoldRed}${PACKAGE}${ColourReset}")
    endforeach()
    if (NOT DEP_LISTS_SOFT_FAIL)
      message(FATAL_ERROR "required package(s) not found, exiting.")
      message("${ColourReset}")
    else()
      message("${ColourReset}")
      return()
    endif()
  endif()
  #output formatting
  if (${_FDP_PNU}_MISSING_OPTIONAL)
    message("")
  endif()
endmacro(dep_lists_pkg_search)

macro(dep_lists_opt_get _FDP_LIST _FDP_IDX _FDP_OUT)
  list(LENGTH ${_FDP_LIST} _FDP_LIST_L)
  if (${_FDP_IDX} LESS _FDP_LIST_L)
    list(GET ${_FDP_LIST} ${_FDP_IDX} ${_FDP_OUT})
  else()
    set(${_FDP_OUT} "")
  endif()
endmacro(dep_lists_opt_get)

macro(dep_lists_prepare_env)
  
  if (ARGV0)
    set(_FDP_PNU ${ARGV0})
  else()
    string(TOUPPER ${PROJECT_NAME} _FDP_PNU)
  endif()

  #####################################################
  ## SET INCLUDES, LIBS, ... (public)
  #####################################################
  foreach(INCLUDE ${${_FDP_PNU}_PKG_INC})
    if (NOT ("${${INCLUDE}}" MATCHES ".*-NOTFOUND"))
      list(APPEND ${_FDP_PNU}_INC ${${INCLUDE}})
      list(APPEND ${_FDP_PNU}_PKG_INC_FOUND ${INCLUDE})
    else()
      # FIXME remove including in deps!
    endif()
  endforeach()
  dep_lists_clean_list(${_FDP_PNU}_INC)

  foreach(LIBDIR ${${_FDP_PNU}_PKG_LINK})
    if (NOT ("${${LIBDIR}}" MATCHES ".*-NOTFOUND"))
      list(APPEND ${_FDP_PNU}_LINK ${${LIBDIR}})
      list(APPEND ${_FDP_PNU}_PKG_LINK_FOUND ${LIBDIR})
    else()
      # FIXME remove including in deps!
    endif()
  endforeach()
  dep_lists_clean_list(${_FDP_PNU}_LINK)

  foreach(LIB ${${_FDP_PNU}_PKG_LIB})
    if (NOT ("${${LIB}}" MATCHES ".*-NOTFOUND"))
      dep_lists_msg_info("add lib var content: ${LIB} : ${${LIB}}")
      list(APPEND ${_FDP_PNU}_LIB ${${LIB}})
      list(APPEND ${_FDP_PNU}_PKG_LIB_FOUND ${LIB})
    else()
      dep_lists_msg_info("lib var not found: ${LIB} : ${${LIB}}")
      # FIXME remove including in deps!
    endif()
  endforeach()
  # WARNING no dep_lists_clean_list because it can contain (repeated) build type specifiesr (like debug, optimized, ...)
  #dep_lists_clean_list(${_FDP_PNU}_LIB)

  #####################################################
  ## SET INCLUDES, LIBS, ... (private)
  #####################################################

  foreach(INCLUDE ${${_FDP_PNU}_PRIVATE_PKG_INC})
    if (NOT ("${${INCLUDE}}" MATCHES ".*-NOTFOUND"))
      list(APPEND ${_FDP_PNU}_PRIVATE_INC ${${INCLUDE}})
    else()
      # FIXME remove including in deps!
    endif()
  endforeach()
  dep_lists_clean_list(${_FDP_PNU}_PRIVATE_INC)

  foreach(LIBDIR ${${_FDP_PNU}_PRIVATE_PKG_LINK})
    if (NOT ("${${LIBDIR}}" MATCHES ".*-NOTFOUND"))
      list(APPEND ${_FDP_PNU}_PRIVATE_LINK ${${LIBDIR}})
    else()
      # FIXME remove including in deps!
    endif()
  endforeach()
  dep_lists_clean_list(${_FDP_PNU}_PRIVATE_LINK)

  foreach(LIB ${${_FDP_PNU}_PRIVATE_PKG_LIB})
    if (NOT ("${${LIB}}" MATCHES ".*-NOTFOUND"))
      list(APPEND ${_FDP_PNU}_PRIVATE_LIB ${${LIB}})
    else()
      # FIXME remove including in deps!
    endif()
  endforeach()
  # WARNING no dep_lists_clean_list because it can contain (repeated) build type specifiesr (like debug, optimized, ...)
  #dep_lists_clean_listdep_lists_clean_list(${_FDP_PNU}_PRIVATE_LIB)


  #####################################################
  ## actually inc/link DIRS (from above)
  #####################################################
  dep_lists_msg_info("FlexDepLists include for ${_FDP_PNU}: ${${_FDP_PNU}_INC} ${${_FDP_PNU}_PRIVATE_INC}")
  include_directories(${${_FDP_PNU}_INC} ${${_FDP_PNU}_PRIVATE_INC})
  link_directories(${${_FDP_PNU}_LINK} ${${_FDP_PNU}_PRIVATE_LINK})
  
  foreach(F ${${_FDP_PNU}_FEATURES})
    set(${F} true)
    string(REPLACE "-" "_" F ${F})
    add_definitions(-D${F})
  endforeach()
  
  list(APPEND ${_FDP_PNU}_LIBRARIES ${${_FDP_PNU}_LIB})
  list(APPEND ${_FDP_PNU}_INCLUDE_DIRS ${${_FDP_PNU}_INC})
  list(APPEND ${_FDP_PNU}_LIBRARY_DIRS ${${_FDP_PNU}_LIB})
  
  string(TOLOWER ${_FDP_PNU} PNL)
  
#####################################################
## copy headers to common location
#####################################################
  if (dep_lists_append_HEADER_PREFIX)
    set(_FDP_HEADER_PREFIX ${dep_lists_append_HEADER_PREFIX})
  else()
    string(TOLOWER ${_FDP_PNU} _FDP_HEADER_PREFIX)
  endif()
  
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/${_FDP_HEADER_PREFIX}/ COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/include/${_FDP_HEADER_PREFIX})
  set(_FPD_HEADER_DEPLIST ${CMAKE_CURRENT_BINARY_DIR}/include/${_FDP_HEADER_PREFIX})
                 
  foreach(_H ${${_FDP_PNU}_HEADERS})
    if (IS_ABSOLUTE ${_H})
      set(_SRC_H ${_H})
    else()
      set(_SRC_H ${CMAKE_CURRENT_SOURCE_DIR}/${_H})
    endif()
    get_filename_component(_H ${_H} NAME)
    dep_lists_msg_info("copy header: ${_SRC_H} -> ${CMAKE_CURRENT_BINARY_DIR}/include/${_FDP_HEADER_PREFIX}/")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/${_FDP_HEADER_PREFIX}/${_H}
                       COMMAND ${CMAKE_COMMAND} -E copy ${_SRC_H} ${CMAKE_CURRENT_BINARY_DIR}/include/${_FDP_HEADER_PREFIX}/
                       DEPENDS ${_SRC_H})
    list(APPEND _FPD_HEADER_DEPLIST ${CMAKE_CURRENT_BINARY_DIR}/include/${_FDP_HEADER_PREFIX}/${_H})
  endforeach()
  
  dep_lists_msg_info("add target ${PNL}-header-export (${_FPD_HEADER_DEPLIST})")
  add_custom_target(${PNL}-header-export ALL DEPENDS ${_FPD_HEADER_DEPLIST})
endmacro(dep_lists_prepare_env)

macro(dep_lists_append _FDP_NAME)
  cmake_policy(SET CMP0011 NEW)
  if (POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
  endif()

  set(dep_lists_append_UNPARSED_ARGUMENTS "")
  dep_lists_parse(dep_lists_append "OPTIONAL;PRIVATE" "PREFIX;FOUND_INDICATOR" "COMPONENTS;FIND_FLAGS" "${ARGN}")
  
  string(TOUPPER ${_FDP_NAME} _FDP_NAME_UPPER)
  
  #output prefix (normally upper case project name)
  if (dep_lists_append_PREFIX)
    set(_FDP_PREFIX ${dep_lists_append_PREFIX})
  else()
    string(TOUPPER ${PROJECT_NAME} _FDP_PREFIX)
  endif()
  
  if (dep_lists_append_FOUND_INDICATOR)
    set(${_FDP_PREFIX}_${_FDP_NAME}_FOUND_INDICATOR ${dep_lists_append_FOUND_INDICATOR})
    list(APPEND ${_FDP_PREFIX}_EXPORT_VARS_VALUES "${_FDP_PREFIX}_${_FDP_NAME}_FOUND_INDICATOR" "${dep_lists_append_FOUND_INDICATOR}")
  endif()
  
  if (dep_lists_append_COMPONENTS)
    set(${_FDP_PREFIX}_${_FDP_NAME}_COMPONENTS ${dep_lists_append_COMPONENTS})
  endif()
  
  if (dep_lists_append_FIND_FLAGS)
    set(${_FDP_PREFIX}_${_FDP_NAME}_FIND_FLAGS ${dep_lists_append_FIND_FLAGS})
    list(APPEND ${_FDP_PREFIX}_EXPORT_VARS_VALUES "${_FDP_PREFIX}_${_FDP_NAME}_FIND_FLAGS" "${dep_lists_append_FIND_FLAGS}")
  endif()
    
  dep_lists_opt_get(dep_lists_append_UNPARSED_ARGUMENTS 0 _FDP_A0)
  dep_lists_opt_get(dep_lists_append_UNPARSED_ARGUMENTS 1 _FDP_A1)
  dep_lists_opt_get(dep_lists_append_UNPARSED_ARGUMENTS 2 _FDP_A2)
  
  if (_FDP_A0)
    set(_FDP_INC ${_FDP_A0})
  else()
    set(_FDP_INC ${_FDP_NAME_UPPER}_INCLUDE_DIRS)
  endif()
  
  if (_FDP_A1)
    set(_FDP_LINK ${_FDP_A1})
  else()
    set(_FDP_LINK ${_FDP_NAME_UPPER}_LIBRARY_DIRS)
  endif()
  
  if (_FDP_A2)
    set(_FDP_LIB ${_FDP_A2})
  else()
    set(_FDP_LIB ${_FDP_NAME_UPPER}_LIBRARIES)
  endif()
  
  dep_lists_msg_info("PNU: ${_FDP_PREFIX}")
  dep_lists_msg_info("NAME: ${_FDP_NAME}")
  dep_lists_msg_info("INC: ${_FDP_INC}")
  dep_lists_msg_info("LINK: ${_FDP_LINK}")
  dep_lists_msg_info("LIB: ${_FDP_LIB}")
  dep_lists_msg_info("OPTIONAL: ${dep_lists_append_OPTIONAL}")
  dep_lists_msg_info("PRIVATE: ${dep_lists_append_PRIVATE}")
  
  if (dep_lists_append_PRIVATE)
    set(_FDP_PREFIX ${_FDP_PREFIX}_PRIVATE)
  endif()
  set(_FDP_PREFIX ${_FDP_PREFIX}_PKG)
  
  if (dep_lists_append_OPTIONAL)
    list(APPEND ${_FDP_PREFIX}_OPT ${_FDP_NAME})
    dep_lists_clean_list(${_FDP_PREFIX}_OPT)
  else()
    list(APPEND ${_FDP_PREFIX} ${_FDP_NAME})
    dep_lists_clean_list(${${_FDP_PREFIX}})
  endif()
  
  list(APPEND ${_FDP_PREFIX}_INC  "${_FDP_INC}")
  dep_lists_clean_list(${_FDP_PREFIX}_INC)
  list(APPEND ${_FDP_PREFIX}_LINK "${_FDP_LINK}")
  dep_lists_clean_list(${_FDP_PREFIX}_LINK)
  list(APPEND ${_FDP_PREFIX}_LIB  "${_FDP_LIB}")
  dep_lists_clean_list(${_FDP_PREFIX}_LIB)
endmacro(dep_lists_append)

macro(dep_lists_init)
  cmake_policy(SET CMP0011 NEW)
  if (POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
  endif()

  file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
  
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
  
  # FIXME add Debug, etc. on win!
  if (WIN32)
    # TODO for windows or only for MSVC?
    set(CMAKE_DEBUG_POSTFIX "d")
  
    foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
            string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
            set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
            set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
            set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    endforeach()
  endif()
  
  string(TOUPPER ${PROJECT_NAME} PNU)
endmacro()

# install ${${PNU}_HEADERS} into CMAKE_CURRENT_BINARY_DIR/${PNL}/
# FIXME only works for relative path headers atm!
function(dep_lists_export_local)

  set(dep_lists_append_UNPARSED_ARGUMENTS "")
  cmake_parse_arguments(dep_lists_append "DIRECT_HEADER_INCLUDE" "HEADER_PREFIX" "" ${ARGN})
  dep_lists_opt_get(dep_lists_append_UNPARSED_ARGUMENTS 0 _FDP_A0)

  #project name prefix (for list names)
  if (_FDP_A0)
    set(_FDP_PNU ${_FDP_A0})
  else()
    string(TOUPPER ${PROJECT_NAME} _FDP_PNU)
  endif()
  
  string(TOLOWER ${_FDP_PNU} PNL)

  include(${FLEXDEPLISTS_DIR}/CMakePackageConfigListHelpers.cmake)

  #####################################################
  ## ...Config.cmake generation
  #####################################################
  set(CMAKECONFIG_PKG ${${_FDP_PNU}_EXPORT_DEPS})
  set(CMAKECONFIG_PKG_INC ${${_FDP_PNU}_PKG_INC_FOUND})
  set(CMAKECONFIG_PKG_LINK ${${_FDP_PNU}_PKG_LINK_FOUND})
  set(CMAKECONFIG_PKG_LIB ${${_FDP_PNU}_PKG_LIB_FOUND})
  set(CMAKECONFIG_PKG_EXPORT_VARS ${${_FDP_PNU}_EXPORT_VARS_VALUES})
  
  
  set(CMAKECONFIG_PKG_COMPONENTS ${${PNU}_PKG_COMPONENTS})

  if (dep_lists_append_DIRECT_HEADER_INCLUDE)
    set(CMAKECONFIG_INC "include/${PNL}") #in build dir - headers were already copied above
  else()
    set(CMAKECONFIG_INC "include") #in build dir - headers were already copied above
  endif()
  
  if (WIN32)
    set(CMAKECONFIG_LIB "")
    foreach(LIB ${${_FDP_PNU}_EXPORT_LIBS})
      list(APPEND CMAKECONFIG_LIB optimized ${LIB} debug ${LIB}d)
    endforeach()
  else()
    set(CMAKECONFIG_LIB ${${_FDP_PNU}_EXPORT_LIBS}) # our libs to link on import
  endif()
  set(CMAKECONFIG_FEATURES ${${_FDP_PNU}_FEATURES})


  #####################################################
  ## local config.cmake
  #####################################################
  set(CMAKECONFIG_CMAKE_DIR ${CMAKE_CURRENT_BINARY_DIR})
  # FIXME multiple configs?
  set(CMAKECONFIG_LINK ${CMAKE_CURRENT_BINARY_DIR}/lib)

  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
  
  set(_FDP_PCFILE "${FLEXDEPLISTS_DIR}/projectConfig.cmake.in")
  file(READ ${_FDP_PCFILE} _FDP_PCCONTENT)
  string(FIND "${_FDP_PCCONTENT}" "projectConfig.cmake.in"  _FDP_PCLINK)
  if (0 LESS ${_FDP_PCLINK})
	set(_FDP_PCFILE "cmake/${_FDP_PCCONTENT}")
  endif()
    
  configure_package_config_file( ${_FDP_PCFILE}
                                "${CMAKECONFIG_CMAKE_DIR}/${PROJECT_NAME}Config.cmake"
                                INSTALL_DESTINATION "${CMAKECONFIG_CMAKE_DIR}"
                                PATH_VARS CMAKECONFIG_PKG CMAKECONFIG_PKG_INC CMAKECONFIG_PKG_LINK CMAKECONFIG_PKG_LIB CMAKECONFIG_INC CMAKECONFIG_LINK CMAKECONFIG_LIB CMAKECONFIG_PKG_COMPONENTS CMAKECONFIG_PKG_EXPORT_VARS)
  export(PACKAGE ${PROJECT_NAME})
  
  
  #####################################################
  ## copy find macros
  #####################################################
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/find/)
    file(COPY cmake/find DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
  endif()
endfunction()

# args: TITLE BOOLVAR [LENGTH]
function(msg_yesno TITLE BOOLVAR)

  cmake_parse_arguments(msg_yesno "NO_COLOR" "COLOR" "" ${ARGN})
  
  #output prefix (normally upper case project name)
  if (msg_yesno_NO_COLOR)
    set(_FDP_PREFIX ${dep_lists_append_PREFIX})
  else()
    string(TOUPPER ${PROJECT_NAME} _FDP_PREFIX)
  endif()
  
  dep_lists_opt_get(msg_yesno_UNPARSED_ARGUMENTS 0 LENGTH)
  
  if (NOT LENGTH)
    set(LENGTH 30)
  endif()
  
  set(FILLED "${TITLE}                                                               ")
  string(SUBSTRING ${FILLED} 0 ${LENGTH} CUT)
  
  if (msg_yesno_NO_COLOR)
    if (${BOOLVAR})
      message("${CUT}- yes")
    else()
      message("${CUT}- no")
    endif()
  elseif (msg_yesno_COLOR)
    if (${BOOLVAR})
      message("${${msg_yesno_COLOR}}${CUT}- yes${ColourReset}")
    else()
      message("${${msg_yesno_COLOR}}${CUT}- no${ColourReset}")
    endif()
  else()
    if (${BOOLVAR})
      message("${Green}${CUT}- yes${ColourReset}")
    else()
      message("${BoldRed}${CUT}- no${ColourReset}")
    endif()
  endif()
  
endfunction(msg_yesno)

macro(dep_lists_pkg_recheck PKG)
  set(FDP_HAVE_SEARCHED_${PKG} "")
  dep_lists_pkg_search()
endmacro()
