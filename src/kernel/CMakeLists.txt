# Create a library called "bfloat" or "vanilla" which includes the source files.
# The extension is already found. Any number of sources could be listed here.


add_library(bfloat bfloat.cpp)

add_library(MMxsmm MMxsmm.cpp)
set_target_properties(MMxsmm PROPERTIES LINKER_LANGUAGE CXX)

add_library(vanilla vanilla.cpp)
set_target_properties(vanilla PROPERTIES LINKER_LANGUAGE CXX)

find_package(OpenMP)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/submodules/libxsmm/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/submodules/libxsmm)

if(OpenMP_CXX_FOUND)
  target_link_libraries(vanilla LINK_PUBLIC bfloat OpenMP::OpenMP_CXX)
else()
  target_link_libraries(vanilla LINK_PUBLIC bfloat)
endif()

target_link_libraries(MMxsmm xsmm m pthread dl c lapack blas lapack)

# Make sure the compiler can find include files for our libraries 
# when other libraries or executables link to them
target_include_directories(bfloat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(vanilla PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(MMxsmm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

