add_executable(extractMarker demo.cpp)
target_link_libraries(extractMarker hdmarker)
set_property(TARGET extractMarker PROPERTY CXX_STANDARD 11)

add_executable(hdmarker-tool demo2.cpp)
target_link_libraries(hdmarker-tool hdmarker)
set_property(TARGET hdmarker-tool PROPERTY CXX_STANDARD 11)
