# Create a library called "bfloat" which includes the source file "bfloat.cpp".
# The extension is already found. Any number of sources could be listed here.
add_library(bfloat bfloat.cpp)

add_library(xsmm xsmm.hpp)
add_library(vanilla vanilla.hpp)

add_executable(bfloat_test bfloat.test.cpp)
add_executable(xsmm_test xsmm_test.cpp)
add_executable(vanilla_test vanilla_test.cpp)

target_link_libraries(bfloat_test LINK_PUBLIC Catch2::Catch2WithMain bfloat)


# Make sure the compiler can find include files for our bfloat library
# when other libraries or executables link to bfloat
target_include_directories (bfloat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
