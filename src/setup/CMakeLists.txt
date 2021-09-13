# Create a library called "setup" which includes the source file "setup.cpp".
# The extension is already found. Any number of sources could be listed here.
add_library(setup setup.cpp)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
  target_link_libraries(setup LINK_PUBLIC bfloat OpenMP::OpenMP_CXX)
else()
  target_link_libraries(setup LINK_PUBLIC bfloat)
endif()


# Make sure the compiler can find include files for our setup library
# when other libraries or executables link to setup
target_include_directories (setup PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
