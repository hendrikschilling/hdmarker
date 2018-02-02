set(_VAD_NAME vad_cmake)

if(VAD_EXTERNAL_ROOT)
  if (VAD_VERBOSE)
    message(STATUS "The root of external dependencies has been specified by the user: ${VAD_EXTERNAL_ROOT}")
  endif()
else()
  set(VAD_EXTERNAL_ROOT ${PROJECT_SOURCE_DIR}/external)
  if (VAD_VERBOSE)
    message(STATUS "The root of external dependencies has not been been specified by the user, setting it to the default: ${VAD_EXTERNAL_ROOT}")
  endif()
endif()

# Don't do anything if the repo has been cloned already.
if(EXISTS "${VAD_EXTERNAL_ROOT}/${_VAD_NAME}")
  if (VAD_VERBOSE)
    message(STATUS "The path '${VAD_EXTERNAL_ROOT}/${_VAD_NAME}' already exists, skipping clone.")
  endif()
else()
  find_package(Git REQUIRED VAD_IGNORE_DEP)
  SET(VAD_vad_cmake_GIT_REPO "git@github.com:hendrikschilling/vigra_cmake.git")

  message(STATUS "Git cloning repo '${VAD_${_VAD_NAME}_GIT_REPO}' into '${VAD_EXTERNAL_ROOT}/${_VAD_NAME}'.")

  unset(GIT_COMMAND_ARGS)
  if (VAD_VERBOSE)
    message(VAD_${_VAD_NAME}_GIT_CLONE_OPTS: ${VAD_${_VAD_NAME}_GIT_CLONE_OPTS})
  endif()
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

  if (VAD_VERBOSE)
    message(STATUS "'${_VAD_NAME}' was successfully cloned into '${VAD_EXTERNAL_ROOT}/${_VAD_NAME}'")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${VAD_EXTERNAL_ROOT}/${_VAD_NAME}")
include(VAD_Wrappers)
include(VigraAddDep)
