add_executable(extractMarker demo.cpp)
target_link_libraries(extractMarker hdmarker)
set_property(TARGET extractMarker PROPERTY CXX_STANDARD 11)
