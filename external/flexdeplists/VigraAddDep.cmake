if(VigraAddDepIncluded)
    return()
endif()

# Store the path to this file as a cached, hidden variable.
set(VigraAddDepPath "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "")

get_property(_VigraAddDepIncludedOnce GLOBAL PROPERTY VigraAddDepIncludedOnce)
if(NOT _VigraAddDepIncludedOnce)
  # On first inclusion, we erase the global cache variable representing the list of
  # directories containing the libraries added via add_library().
  set(_VAD_LIBRARY_DIR_LIST "" CACHE INTERNAL "")
  function(VAD_CREATE_ACTIVATE)
    if(WIN32)
      set(VAD_LIBRARY_DIR_LIST "${_VAD_LIBRARY_DIR_LIST}")
      configure_file("${VigraAddDepPath}/vad_activate.bat.in" "${CMAKE_BINARY_DIR}/.vad/vad_activate.bat.generate" @ONLY)
      configure_file("${VigraAddDepPath}/vad_path.in" "${CMAKE_BINARY_DIR}/.vad/vad_path.generate" @ONLY)
    endif()
  endfunction()
  variable_watch(_VAD_LIBRARY_DIR_LIST VAD_CREATE_ACTIVATE)
  set_property(GLOBAL PROPERTY VigraAddDepIncludedOnce TRUE)
  if(WIN32)
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/vad_activate_$<CONFIG>.bat" INPUT "${CMAKE_CURRENT_BINARY_DIR}/.vad/vad_activate.bat.generate")
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/vad_deactivate.bat" INPUT "${VigraAddDepPath}/vad_deactivate.bat.in")
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/.vad/vad_path_$<CONFIG>" INPUT "${CMAKE_CURRENT_BINARY_DIR}/.vad/vad_path.generate")
  endif()
endif()

include(CMakeParseArguments)
include(VAD_target_properties)

#hardcoded package name aliases (TODO put in extra file? (and allow user to provide extra overrides))
set(_VAD_Eigen_ALIAS Eigen3)

if(VAD_EXTERNAL_ROOT)
  message(STATUS "The root of external dependencies has been specified by the user: ${VAD_EXTERNAL_ROOT}")
else()
  set(VAD_EXTERNAL_ROOT ${PROJECT_SOURCE_DIR}/external)
  message(STATUS "The root of external dependencies has not been been specified by the user, setting it to the default: ${VAD_EXTERNAL_ROOT}")
endif()

function(git_clone _VAD_NAME)
  find_package(Git REQUIRED VAD_IGNORE_DEP)

  # Check if a git repo has been determined for the dependency.
  if(NOT VAD_${_VAD_NAME}_GIT_REPO)
    message(FATAL_ERROR "A git clone was requested for dependency ${_VAD_NAME}, but no repository variable has been set.")
  endif()

  # Don't do anything if the repo has been cloned already.
  if(EXISTS "${VAD_EXTERNAL_ROOT}/${_VAD_NAME}")
    message(STATUS "The path '${VAD_EXTERNAL_ROOT}/${_VAD_NAME}' already exists, skipping clone.")
    return()
  endif()

  message(STATUS "Git cloning repo '${VAD_${_VAD_NAME}_GIT_REPO}' into '${VAD_EXTERNAL_ROOT}/${_VAD_NAME}'.")

  unset(GIT_COMMAND_ARGS)
  message(VAD_${_VAD_NAME}_GIT_CLONE_OPTS: ${VAD_${_VAD_NAME}_GIT_CLONE_OPTS})
  # Build the command line options for the clone command.
  list(APPEND GIT_COMMAND_ARGS "clone" "${VAD_${_VAD_NAME}_GIT_REPO}" "${_VAD_NAME}")
  if(VAD_${_VAD_NAME}_GIT_CLONE_OPTS)
    list(INSERT GIT_COMMAND_ARGS 1 ${VAD_${_VAD_NAME}_GIT_CLONE_OPTS})
  endif()

  # Run the clone command.
  execute_process(COMMAND "${GIT_EXECUTABLE}" ${GIT_COMMAND_ARGS}
    WORKING_DIRECTORY "${VAD_EXTERNAL_ROOT}"
    RESULT_VARIABLE RES
    ERROR_VARIABLE OUT
    OUTPUT_VARIABLE OUT)
  if(RES)
    message(FATAL_ERROR "The clone command for '${_VAD_NAME}' failed. The command arguments were: ${GIT_COMMAND_ARGS}\n\nThe output is:\n====\n${OUT}\n====")
  endif()

  message(STATUS "'${_VAD_NAME}' was successfully cloned into '${VAD_EXTERNAL_ROOT}/${_VAD_NAME}'")
endfunction()

# Override the builtin add_library() function so that the new library is appended to a global list
# if it is a non-global imported target. It will then be turned into a global imported target by find_package().
# We will also record the library location in a generator expression.
function(add_library _VAD_NAME)
  # Check if the target is an imported non-global one. In that case, add it to the list of non-global imported
  # targets.
  list(FIND ARGN "IMPORTED" _IDX_IMP)
  list(FIND ARGN "GLOBAL" _IDX_GLOB)
  if(NOT _IDX_IMP EQUAL -1 AND _IDX_GLOB EQUAL -1)
    if(VAD_VERBOSE)
      message(STATUS "Adding target '${_VAD_NAME}' to the list of non-global imported targets.")
    endif()
    get_property(_LIB_LIST GLOBAL PROPERTY _VAD_IMPORTED_NOGLOBAL_LIST)
    list(APPEND _LIB_LIST "${_VAD_NAME}")
    set_property(GLOBAL PROPERTY _VAD_IMPORTED_NOGLOBAL_LIST ${_LIB_LIST})
  endif()

  # Call the original add_library() function.
  _add_library(${_VAD_NAME} ${ARGN})

  # If the target is not INTERFACE or ALIAS, we will record its location.
  list(FIND ARGN "INTERFACE" _IDX_IFACE)
  list(FIND ARGN "ALIAS" _IDX_ALIAS)
  if(_IDX_IFACE EQUAL -1 AND _IDX_ALIAS EQUAL -1)
    set(_LIBRARY_DIR_LIST "${_VAD_LIBRARY_DIR_LIST}")
    list(APPEND _LIBRARY_DIR_LIST "$<TARGET_FILE_DIR:${_VAD_NAME}>")
    list(REMOVE_DUPLICATES _LIBRARY_DIR_LIST)
    set(_VAD_LIBRARY_DIR_LIST "${_LIBRARY_DIR_LIST}" CACHE INTERNAL "")
  endif()
endfunction()

# An utility function to turn an imported target into a GLOBAL imported target.
# Imported targets have special visibility rules: they are not visible outside the subdir/subproject
# from which they were defined.
function(vad_make_imported_target_global _VAD_NAME)
  # Check the target actually exists.
  if(NOT TARGET "${_VAD_NAME}")
    message(FATAL_ERROR "'vad_make_imported_target_global()' was called with argument '${_VAD_NAME}', but a target with that name does not exist.")
  endif()

  # Check if the target is imported. If it not, we just exit.
  get_target_property(IMP_PROP "${_VAD_NAME}" IMPORTED)
  if(NOT IMP_PROP)
    get_target_property(_IMP_TYPE "${_VAD_NAME}" TYPE)
    message(FATAL_ERROR "Target '${_VAD_NAME}' is not IMPORTED but of type \"${_IMP_TYPE}\", no need to make it global.")
  endif()

  # The strategy here is as follows:
  # - we create a new imported target with global visibility, and we copy the properties of the original target;
  # - we create an INTERFACE target depending on the new target above;
  # - we create an alias for the INTERFACE target with the name of the original target.
  # It is necessary to go through the double indirection because of certain CMake rules regarding the allowed target
  # names.
  if(VAD_VERBOSE)
    message(STATUS "Turning IMPORTED target '${_VAD_NAME}' into a GLOBAL target.")
  endif()
  get_target_property(TARGET_TYPE ${_VAD_NAME} TYPE)
  if(VAD_VERBOSE)
    message(STATUS "Target type is: ${TARGET_TYPE}")
  endif()
  if(TARGET_TYPE MATCHES "INTERFACE_LIBRARY")
    add_library(_VAD_${_VAD_NAME}_STUB INTERFACE IMPORTED GLOBAL)
  else()
    add_library(_VAD_${_VAD_NAME}_STUB UNKNOWN IMPORTED GLOBAL)
  endif()
  foreach(TPROP ${VAD_TARGET_PROPERTIES})
      if(TARGET_TYPE MATCHES "INTERFACE_LIBRARY")
        list(FIND VAD_INTERFACE_LIBRARY_BLACKLISTED_PROPERTIES "${TPROP}" _IDX)
        if(NOT _IDX EQUAL -1)
          continue()
        endif()
      endif()
      get_target_property(PROP ${_VAD_NAME} "${TPROP}")
      if(PROP)
          if(VAD_VERBOSE)
            message(STATUS "Copying property '${TPROP}' of IMPORTED target '${_VAD_NAME}': '${PROP}'")
          endif()
          set_property(TARGET _VAD_${_VAD_NAME}_STUB PROPERTY "${TPROP}" "${PROP}")
      endif()
  endforeach()

  # Create the final aliases. We need to remove any double colon from the original name as they are not allowed
  # in the names of targets in this context.
  string(REPLACE "::" "" NAME_NO_COLONS "${_VAD_NAME}")
  add_library(_VAD_${NAME_NO_COLONS}_STUB_INTERFACE INTERFACE)
  target_link_libraries(_VAD_${NAME_NO_COLONS}_STUB_INTERFACE INTERFACE _VAD_${_VAD_NAME}_STUB)
  add_library("${_VAD_NAME}" ALIAS _VAD_${NAME_NO_COLONS}_STUB_INTERFACE)

  # We need to remove the library dir for the original target from the global list:
  # the target is now just an alias with no TARGET_FILE_DIR property. Its actual
  # TARGET_FILE_DIR property has been recorded in the STUB targets, which export
  # this information in the global list.
  set(_LIBRARY_DIR_LIST "${_VAD_LIBRARY_DIR_LIST}")
  list(REMOVE_ITEM _LIBRARY_DIR_LIST "$<TARGET_FILE_DIR:${_VAD_NAME}>")
  set(_VAD_LIBRARY_DIR_LIST "${_LIBRARY_DIR_LIST}")
endfunction()

# Override the builtin find_package() function to just call vigra_add_dep(). The purpose of this override is to make
# sure that any dep added via find_package() goes through the VAD machinery.
function(find_package _VAD_NAME)
  vigra_add_dep(${_VAD_NAME} SYSTEM ${ARGN})
  # TODO emulate REQUIRED behavior - check wether something was found?
#   list(FIND "${ARGN}" "REQUIRED" _req_idx)
#   if (NOT _req_idx EQUAL -1)
#     message(FATAL_ERROR "required package ${_VAD_NAME} not found - will exit")
#   endif()
endfunction()

function(var_is_cached _VAR _CACHED_RESVAR)
  get_property(_vad_cv_isdefined CACHE ${_VAR} PROPERTY TYPE)
  # cmake cache vars do not need a type -> UNINITIALIZED -> but then they are also NOT STORED in the cache!
  if ("${_vad_cv_isdefined}" STREQUAL "" OR "${_vad_cv_isdefined}" STREQUAL "UNINITIALIZED")
   set(${_CACHED_RESVAR} FALSE PARENT_SCOPE)
  else()
   set(${_CACHED_RESVAR} TRUE PARENT_SCOPE)
  endif()
endfunction()

function(make_imported_targets_global)
  get_property(_IMP_NOGLOB_LIB_LIST GLOBAL PROPERTY _VAD_IMPORTED_NOGLOBAL_LIST)
  foreach(_NEWLIB ${_IMP_NOGLOB_LIB_LIST})
    vad_make_imported_target_global("${_NEWLIB}")
  endforeach()
  set_property(GLOBAL PROPERTY _VAD_IMPORTED_NOGLOBAL_LIST "")
endfunction()

# In addition to calling the builtin find_package(), this function
# will take care of exporting as global variables the variables defined by the builtin find_package(), and to make
# the imported targets defined by the builtin find_package() global.
macro(find_package_plus_no_import _VAD_NAME)
  message(STATUS "Invoking the patched 'find_package()' function for dependency ${_VAD_NAME}.")
  # As a first step we want to erase from the cache all the variables that were exported by a previous call
  # to find_package_plus(). These variables are stored in a variable called VAD_NEW_VARS_${_VAD_NAME}, which we will
  # also remove.
  foreach(_NEWVAR ${VAD_NEW_VARS_${_VAD_NAME}})
    if ("${VAD_${_NEWVAR}_RESET_OWNER}" STREQUAL "${_VAD_NAME}")
      unset(${_NEWVAR} CACHE)
      unset(${_NEWVAR})
    else()
      message(WARNING "inconsistent variable ownership (${_VAD_NAME} tried to unset ${_NEWVAR} which was detected as set by ${VAD_${_NEWVAR}_RESET_OWNER})")
    endif()
  endforeach()

  # Get the list of the currently defined variables.
  get_cmake_property(_OLD_VARIABLES_${_VAD_NAME} VARIABLES)
  # Call the original find_package().
  _find_package(${_VAD_NAME} ${ARGN})
  # Detect new variables defined by find_package() and make them cached.
  get_cmake_property(_NEW_VARIABLES_${_VAD_NAME} VARIABLES)
  # Remove duplicates in the new vars list.
  list(REMOVE_DUPLICATES _NEW_VARIABLES_${_VAD_NAME})
  # Create a lower case version of the package name. We will use this in string matching below.
  string(TOLOWER "${_VAD_NAME}" _NAME_LOW)
  # Detect the new variables by looping over the new vars list and comparing its elements to the old vars.
  # We will append the detected variables to the _DIFFVARS list.
  set(_DIFFVARS)
  foreach(_NEWVAR ${_NEW_VARIABLES_${_VAD_NAME}})
      string(TOLOWER "${_NEWVAR}" _NEWVAR_LOW)
      # We forcefully exclude variables named in a certain way:
      # - starts with underscore "_",
      # - vad-related variables,
      # - variables set by CMake's builtin find_package,
      # - args* (these are arguments to macros/functions).
      if(NOT _NEWVAR_LOW MATCHES "^_" AND NOT _NEWVAR_LOW MATCHES "^vad" AND NOT _NEWVAR_LOW MATCHES "^find_package" AND NOT _NEWVAR_LOW MATCHES "^argv[0-9]*" AND NOT _NEWVAR_LOW MATCHES "^argc" AND NOT _NEWVAR_LOW MATCHES "^argn")
        
        list(FIND _OLD_VARIABLES_${_VAD_NAME} "${_NEWVAR}" _NEWVARIDX)
        
        var_is_cached(${_NEWVAR} _NEWVAR_IS_CACHED)
        
        # variables might first be added to the cache by a package and then overwritten by the user
        # these should stay in cache and stay in new_vars list
        if (_NEWVAR_IS_CACHED)
          #message("newvar ${_NEWVAR} is cached")
          if (VAD_${_NEWVAR}_RESET_OWNER STREQUAL ${_VAD_NAME})
            # force overwrite if it was cached by us (else ignore it), and leave/readd to diffvars
            string(FIND "${VAD_NEW_VARS_${_VAD_NAME}}" ${_NEWVAR} _IS_NEWVAR_IDX)
            if (NOT _IS_NEWVAR_IDX EQUAL -1)
              get_property(_NEWVAR_CACHETYPE CACHE ${_NEWVAR} PROPERTY TYPE)
              # FIXME get INTERNAL + ADVANCED flag? why do we rewrite this exaclty?
              set(${_NEWVAR} ${${_NEWVAR}} CACHE ${_NEWVAR_CACHETYPE} "" FORCE)
              list(APPEND _DIFFVARS ${_NEWVAR})
              set(VAD_${_NEWVAR}_RESET_OWNER ${_VAD_NAME} CACHE INTERNAL "")
            else()
              #message("skipping variable ${_NEWVAR} is ours but not known ${VAD_${_NEWVAR}_RESET_OWNER}")
            endif()
          else()
            #message("skipping variable ${_NEWVAR} is cached by \"${VAD_${_NEWVAR}_RESET_OWNER}\" (I'm ${_VAD_NAME})")
          endif()
        elseif(_NEWVARIDX EQUAL -1)
            # New var was not found among the old ones. We check if it is not an internal variable,
            # in which case we will add it to the cached variables.
            # cached variables are also skipped (should not be added to diffvars so they will not be reset!
              # Make sure we don't store multiline strings in the cache, as that is not supported.
              string(REPLACE "\n" ";" _NEWVAR_NO_NEWLINES "${${_NEWVAR}}")
              if(VAD_VERBOSE)
                message(STATUS "Storing new variable in cache: '${_NEWVAR}:${_NEWVAR_NO_NEWLINES}'")
              endif()
              if(_NEWVAR_LOW MATCHES "librar" OR _NEWVAR_LOW MATCHES "include")
                # Variables which are likely to represent lib paths or include dirs are set as string variables,
                # so that they are visible from the GUI.
                set(${_NEWVAR} ${_NEWVAR_NO_NEWLINES} CACHE STRING "")
              else()
                # Otherwise, mark them as internal vars.
                set(${_NEWVAR} ${_NEWVAR_NO_NEWLINES} CACHE INTERNAL "")
              endif()
              list(APPEND _DIFFVARS ${_NEWVAR})
              set(VAD_${_NEWVAR}_RESET_OWNER ${_VAD_NAME} CACHE INTERNAL "")
        else()
          #message("skipping known variable ${_VAD_NAME}")
        endif() #new var
      endif() #exclude match
  endforeach()
  # Store a list of the variables that were exported as cache variables.
  set(VAD_NEW_VARS_${_VAD_NAME} ${_DIFFVARS} CACHE INTERNAL "")
endmacro()

function(vad_add_var _VARNAME _VARVAL)
  if (NOT _VAD_NAME)
    message(FATAL_ERROR "vad_add_var requires _VAD_NAME to be set!")
  endif()
  set(${_VARNAME} ${_VARVAL} CACHE INTERNAL "")
  set(VAD_NEW_VARS_${_VAD_NAME} "${VAD_NEW_VARS_${_VAD_NAME}};${_VARNAME}" CACHE INTERNAL "")
  set(VAD_${_VARNAME}_RESET_OWNER ${_VAD_NAME} CACHE INTERNAL "")
endfunction()

# In addition to calling the builtin find_package(), this function
# will take care of exporting as global variables the variables defined by the builtin find_package(), and to make
# the imported targets defined by the builtin find_package() global.
function(find_package_plus _VAD_NAME)
  find_package_plus_no_import(${_VAD_NAME} ${ARGN})

  # Now take care of turning any IMPORTED non-GLOBAL target defined by a find_package() call into a GLOBAL one.
  make_imported_targets_global()
endfunction()

# The default vad_system() function.
function(vad_system_default _VAD_NAME)
  message(STATUS "Invoking the default implementation of vad_system() for dependency ${_VAD_NAME}.")
  find_package_plus(${_VAD_NAME} ${ARGN})
endfunction()

function(vad_system_call _VAD_NAME)
  message("call vad_system: ${_VAD_NAME} ${ARGN}")
  vad_system(${_VAD_NAME} ${ARGN})
  # Check the FOUND flag.
  if(NOT ${_VAD_NAME}_FOUND)
    set(VAD_${_VAD_NAME}_SYSTEM_NOT_FOUND TRUE CACHE INTERNAL "")
  else()
    # for recursive calls the inner might fail and the outer succeed
    set(VAD_${_VAD_NAME}_SYSTEM_NOT_FOUND FALSE CACHE INTERNAL "")
  endif()
endfunction()

# A function to reset the hooks that are optionally defined in VAD files. Calling this function
# will reset the hooks to their default implementations.
function(vad_reset_hooks)
  function(vad_system)
    message("call vad_system_default: [${_VAD_NAME}] ${ARGN} - ${ARGC}")
    vad_system_default(${ARGN})
  endfunction()
  function(vad_live _VAD_NAME)
    message(STATUS "Invoking the default implementation of vad_live() for dependency ${_VAD_NAME}.")
    git_clone(${_VAD_NAME})
    add_subdirectory("${VAD_EXTERNAL_ROOT}/${_VAD_NAME}" "${VAD_EXTERNAL_ROOT}/${_VAD_NAME}/build_external_dep")
  endfunction()
  function(vad_deps _VAD_NAME)
  endfunction()
  unset(GIT_REPO PARENT_SCOPE)
  unset(GIT_CLONE_OPTS PARENT_SCOPE)
endfunction()

function(vad_dep_satisfied _VAD_NAME)
  get_property(_VAD_DEP_${_VAD_NAME}_SATISFIED GLOBAL PROPERTY VAD_DEP_${_VAD_NAME}_SATISFIED)
  if(_VAD_DEP_${_VAD_NAME}_SATISFIED)
    set(VAD_DEP_${_VAD_NAME}_SATISFIED TRUE PARENT_SCOPE)
  else()
    set(VAD_DEP_${_VAD_NAME}_SATISFIED FALSE PARENT_SCOPE)
  endif()
endfunction()

function(vigra_add_dep _VAD_NAME)
  string(FIND "${_vigra_add_dep_running_${_VAD_NAME}_ARGS}" "VAD_PROT_REC_START${ARGN}VAD_PROT_REC_STOP" _ved_rec_prot)

  if (NOT _ved_rec_prot EQUAL -1)
    return()
  endif()

  # Reset the hooks.
  vad_reset_hooks()
  
  if (_VAD_${_VAD_NAME}_ALIAS)
    set(_VAD_NAME ${_VAD_${_VAD_NAME}_ALIAS})
  endif()

  # Parse the options.
  set(options SYSTEM LIVE VAD_IGNORE_DEP)
  set(oneValueArgs GIT_REPO GIT_BRANCH GIT_COMMIT)
  set(multiValueArgs GIT_CLONE_OPTS)
  cmake_parse_arguments(ARG_VAD_${_VAD_NAME} "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT ARG_VAD_${_VAD_NAME}_VAD_IGNORE_DEP)
    message(STATUS "Declaring '${_VAD_NAME}' as a dependency for '${PROJECT_NAME}'")
    get_property(_VAD_${PROJECT_NAME}_DEP_LIST GLOBAL PROPERTY VAD_${PROJECT_NAME}_DEP_LIST)
    list(APPEND _VAD_${PROJECT_NAME}_DEP_LIST "${_VAD_NAME}")
    list(REMOVE_DUPLICATES _VAD_${PROJECT_NAME}_DEP_LIST)
    message(STATUS "Full dep list for '${PROJECT_NAME}': ${_VAD_${PROJECT_NAME}_DEP_LIST}")
    set_property(GLOBAL PROPERTY VAD_${PROJECT_NAME}_DEP_LIST "${_VAD_${PROJECT_NAME}_DEP_LIST}")
  endif()

  # Validate options.
  # SYSTEM and LIVE cannot be present at the same time.
  if(ARG_VAD_${_VAD_NAME}_SYSTEM AND ARG_VAD_${_VAD_NAME}_LIVE)
    message(FATAL_ERROR "Dependency '${_VAD_NAME}' was added both as a SYSTEM dependency and as a LIVE dependency: only one choice can be active.")
  endif()

  # Check if the dependency is already satisfied.
  vad_dep_satisfied(${_VAD_NAME})
  if(VAD_DEP_${_VAD_NAME}_SATISFIED)
    message(STATUS "Dependency '${_VAD_NAME}' already satisfied, skipping.")
    return()
  endif()

  if(NOT ARG_VAD_${_VAD_NAME}_SYSTEM AND NOT ARG_VAD_${_VAD_NAME}_LIVE)
    # If neither SYSTEM nor LIVE were specified, assume SYSTEM.
    message(STATUS "Dependency '${_VAD_NAME}' not specified as a SYSTEM or LIVE dependency, assuming SYSTEM.")
    set(ARG_VAD_${_VAD_NAME}_SYSTEM YES)
  elseif(ARG_VAD_${_VAD_NAME}_SYSTEM)
    # SYSTEM dependency required.
    message(STATUS "Dependency '${_VAD_NAME}' specified as a SYSTEM dependency.")
  else()
    # LIVE dependency required.
    message(STATUS "Dependency '${_VAD_NAME}' specified as a LIVE dependency.")
  endif()

  # if(VAD_GIT_BRANCH AND VAD_GIT_COMMIT)
  #   message(FATAL_ERROR "At most one of GIT_BRANCH and GIT_COMMIT can be used.")
  # endif()

  # First thing, we try to read the dep properties from the VAD file, if provided.
  find_file(VAD_${_VAD_NAME}_FILE VAD_${_VAD_NAME}.cmake ${CMAKE_MODULE_PATH})
  if(VAD_${_VAD_NAME}_FILE)
    message(STATUS "VAD file 'VAD_${_VAD_NAME}.cmake' was found at '${VAD_${_VAD_NAME}_FILE}'. The VAD file will now be parsed.")
    include(${VAD_${_VAD_NAME}_FILE})
    if(GIT_REPO)
      if(ARG_VAD_${_VAD_NAME}_LIVE)
        message(STATUS "VAD file for dependency ${_VAD_NAME} specifies git repo: ${GIT_REPO}")
      endif()
      set(VAD_${_VAD_NAME}_GIT_REPO_FILE ${GIT_REPO})
      unset(GIT_REPO)
    endif()
    if(GIT_CLONE_OPTS)
      if(ARG_VAD_${_VAD_NAME}_LIVE)
        message(STATUS "VAD file for dependency ${_VAD_NAME} specifies git clone options: ${GIT_CLONE_OPTS}")
      endif()
      set(VAD_${_VAD_NAME}_GIT_CLONE_OPTS_FILE ${GIT_CLONE_OPTS})
      unset(GIT_CLONE_OPTS)
    else()
      message("${GIT_CLONE_OPTS}")
      message("still have opts: VAD_${_VAD_NAME}_GIT_CLONE_OPTS_FILE ${VAD_${_VAD_NAME}_GIT_CLONE_OPTS_FILE}")
    endif()
    # TODO: branch/commit.
    if(GIT_BRANCH)
      message(STATUS "Git branch for dependency ${_VAD_NAME} read from VAD file: ${GIT_BRANCH}")
      set(VAD_GIT_BRANCH_CUR ${GIT_BRANCH})
      unset(GIT_BRANCH)
    endif()
    if(GIT_COMMIT)
      message(STATUS "Git commit for dependency ${_VAD_NAME} read from VAD file: ${GIT_COMMIT}")
      set(VAD_GIT_COMMIT_CUR ${GIT_COMMIT})
      unset(GIT_COMMIT)
    endif()
  else()
    message(STATUS "No VAD file 'VAD_${_VAD_NAME}.cmake' was found.")
  endif()
  # Unset the variable from cache.
  unset(VAD_${_VAD_NAME}_FILE CACHE)

  # TODO: same for branch and commit.
  if(VAD_${_VAD_NAME}_GIT_REPO)
    # Git repo was passed from command line or this is not the first run.
    if(ARG_VAD_${_VAD_NAME}_LIVE)
      message(STATUS "Final git repo address for dependency ${_VAD_NAME} is from cache: ${VAD_${_VAD_NAME}_GIT_REPO}")
    endif()
  else()
    if(ARG_VAD_${_VAD_NAME}_GIT_REPO)
      # Git repo passed as function argument, overrides setting from file.
      if(ARG_VAD_${_VAD_NAME}_LIVE)
        message(STATUS "Final git repo address for dependency ${_VAD_NAME} is from function argument: ${ARG_VAD_${_VAD_NAME}_GIT_REPO}")
      endif()
      set(VAD_${_VAD_NAME}_GIT_REPO ${ARG_VAD_${_VAD_NAME}_GIT_REPO} CACHE INTERNAL "")
    elseif(VAD_${_VAD_NAME}_GIT_REPO_FILE)
      # Git repository coming from the file.
      if(ARG_VAD_${_VAD_NAME}_LIVE)
        message(STATUS "Final git repo address for dependency ${_VAD_NAME} is from file: ${VAD_${_VAD_NAME}_GIT_REPO_FILE}")
      endif()
      set(VAD_${_VAD_NAME}_GIT_REPO ${VAD_${_VAD_NAME}_GIT_REPO_FILE} CACHE INTERNAL "")
    endif()
  endif()

  if(VAD_${_VAD_NAME}_GIT_CLONE_OPTS)
    # Git clone options were passed from command line or this is not the first run.
    if(ARG_VAD_${_VAD_NAME}_LIVE)
      message(STATUS "Final git clone options for dependency ${_VAD_NAME} are from cache: ${VAD_${_VAD_NAME}_GIT_CLONE_OPTS}")
    endif()
  else()
    if(ARG_VAD_${_VAD_NAME}_GIT_CLONE_OPTS)
      # Git clone options passed as function argument, overrides setting from file.
      if(ARG_VAD_${_VAD_NAME}_LIVE)
        message(STATUS "Final git clone options for dependency ${_VAD_NAME} are from function argument: ${ARG_VAD_${_VAD_NAME}_GIT_CLONE_OPTS}")
      endif()
      set(VAD_${_VAD_NAME}_GIT_CLONE_OPTS ${ARG_VAD_${_VAD_NAME}_GIT_CLONE_OPTS} CACHE INTERNAL "")
    elseif(VAD_${_VAD_NAME}_GIT_CLONE_OPTS_FILE)
      # Git clone options coming from the file.
      if(ARG_VAD_${_VAD_NAME}_LIVE)
        message(STATUS "Final git clone options for dependency ${_VAD_NAME} are from file: ${VAD_${_VAD_NAME}_GIT_CLONE_OPTS_FILE}")
      endif()
      set(VAD_${_VAD_NAME}_GIT_CLONE_OPTS ${VAD_${_VAD_NAME}_GIT_CLONE_OPTS_FILE} CACHE INTERNAL "")
    endif()
  endif()

  if(ARG_VAD_${_VAD_NAME}_SYSTEM)
    # endless recursion protection vad system might call find_package again...
    string(LENGTH "${_vigra_add_dep_running_${_VAD_NAME}_ARGS}" _vad_add_dep_prot_args_len)
    list(APPEND _vigra_add_dep_running_${_VAD_NAME}_ARGS "VAD_PROT_REC_START${ARGN}VAD_PROT_REC_STOP")
    message("call sys call: ${_VAD_NAME} ${ARG_VAD_${_VAD_NAME}_UNPARSED_ARGUMENTS}")
    vad_system_call(${_VAD_NAME} ${ARG_VAD_${_VAD_NAME}_UNPARSED_ARGUMENTS})
    string(SUBSTRING "${_vigra_add_dep_running_${_VAD_NAME}_ARGS}" 0 ${_vad_add_dep_prot_args_len} _vigra_add_dep_running_${_VAD_NAME}_ARGS)
    
    if(VAD_${_VAD_NAME}_SYSTEM_NOT_FOUND)
      message(STATUS "Dependency ${_VAD_NAME} was not found system-wide, vigra_add_dep() will exit without marking the dependency as satisfied.")
      unset(VAD_${_VAD_NAME}_SYSTEM_NOT_FOUND CACHE)
      vad_reset_hooks()
      return()
    endif()
  elseif(ARG_VAD_${_VAD_NAME}_LIVE)
    # Create the external deps directory if it does not exist already.
    if(NOT EXISTS "${VAD_EXTERNAL_ROOT}")
      message(STATUS "Directory '${VAD_EXTERNAL_ROOT}' does not exist, creating it.")
      file(MAKE_DIRECTORY "${VAD_EXTERNAL_ROOT}")
    endif()
    vad_live(${_VAD_NAME})
  endif()

  # Mark the dep as satisfied.
  message(STATUS "Marking dependency ${_VAD_NAME} as satisfied.")
  # NOTE: the flag be set as a "global" variable, that is available from everywhere, but not in the cache - otherwise
  # subsequent cmake runs will break. Emulate global variables via global properties:
  # http://stackoverflow.com/questions/19345930/cmake-lost-in-the-concept-of-global-variables-and-parent-scope-or-add-subdirec
  set_property(GLOBAL PROPERTY VAD_DEP_${_VAD_NAME}_SATISFIED YES)

  # Reset the hooks at the end as well for cleanup purposes.
  vad_reset_hooks()
endfunction()

# Mark as included.
set(VigraAddDepIncluded YES)
