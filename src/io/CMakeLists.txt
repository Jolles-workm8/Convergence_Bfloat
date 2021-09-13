# Create a library called "io" which includes the source file "io.cpp".
# The extension is already found. Any number of sources could be listed here.
add_library(io io.hpp)

set_target_properties(io PROPERTIES LINKER_LANGUAGE CXX)

# Make sure the compiler can find include files for our io library
# when other libraries or executables link to io
target_include_directories (io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
