cmake_minimum_required(VERSION 3.10)

project(bfloat16_fp32_convergence LANGUAGES CXX)

set(CMAKE_CXX_COMPILER /usr/bin/g++-10)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -march=cooperlake")

find_package(OpenMP)

add_subdirectory(bfloat)
add_subdirectory(io)
add_subdirectory(measurement)
add_subdirectory(setup)

add_subdirectory(submodules/Catch2)


add_executable(kernel driver.cpp)
  # Link the executable to the matrix/measurement libraries. Since the libraries have
# public include directories we will use those link directories when building
# the driver

if(OpenMP_CXX_FOUND)
  target_link_libraries(kernel LINK_PUBLIC measurement setup OpenMP::OpenMP_CXX)
else()
  target_link_libraries(kernel LINK_PUBLIC measurement setup)
endif()
