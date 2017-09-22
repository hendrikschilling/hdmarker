# --------------------------------------------------------------------------------------------
#  Installation for CMake Module:  OpenCVConfig.cmake
#  ${BIN_DIR}/OpenCVConfig.cmake              -> For use *without* "make install"
# -------------------------------------------------------------------------------------------

configure_file("${${CMAKE_PROJECT_NAME}_SOURCE_DIR}/cmake/${CMAKE_PROJECT_NAME}Config.cmake.in" "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake" @ONLY)


