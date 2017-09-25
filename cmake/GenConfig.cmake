# --------------------------------------------------------------------------------------------
#  Installation for CMake Module:  OpenCVConfig.cmake
#  ${BIN_DIR}/OpenCVConfig.cmake              -> For use *without* "make install"
# -------------------------------------------------------------------------------------------

configure_file("${hdmarker_SOURCE_DIR}/cmake/hdmarkerConfig.cmake.in" "${hdmarker_BINARY_DIR}/hdmarkerConfig.cmake" @ONLY)


