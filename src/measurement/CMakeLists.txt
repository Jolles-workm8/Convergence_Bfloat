# Create a library called "measurement" which includes the source file "measurement.cpp".
# The extension is already found. Any number of sources could be listed here.
add_library(measurement measurement.cpp)

set_target_properties(measurement PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(measurement LINK_PUBLIC setup io)



#target_link_libraries(measurement LINK_PUBLIC io)

# Make sure the compiler can find include files for our measurement library
# when other libraries or executables link to measurement
target_include_directories (measurement PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
