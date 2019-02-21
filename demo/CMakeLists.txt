add_executable(extractMarker demo.cpp)
target_link_libraries(extractMarker hdmarker)
set_property(TARGET extractMarker PROPERTY CXX_STANDARD 11)

add_executable(hdmarker-tool demo2.cpp)
target_link_libraries(hdmarker-tool hdmarker)
set_property(TARGET hdmarker-tool PROPERTY CXX_STANDARD 11)

find_package(Boost COMPONENTS system filesystem iostreams REQUIRED)


add_library(hdmarker-simple-calib-lib simplecalib.cpp)
target_link_libraries(hdmarker-simple-calib-lib
  hdmarker
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  )
set_property(TARGET hdmarker-simple-calib-lib PROPERTY CXX_STANDARD 11)

add_executable(hdmarker-simple-calib simple-calib.cpp)
target_link_libraries(hdmarker-simple-calib
  hdmarker
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  hdmarker-simple-calib-lib
  )
set_property(TARGET hdmarker-simple-calib PROPERTY CXX_STANDARD 11)
