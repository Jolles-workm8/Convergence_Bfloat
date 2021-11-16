cmake_minimum_required(VERSION 3.16.3)

project(bfloat16_fp32_convergence LANGUAGES CXX)

set(CMAKE_CXX_COMPILER /usr/bin/g++-10)
set(CMAKE_CXX_FLAGS "-Wall -Wparentheses -Wextra -march=cooperlake -fno-stack-protector -fopenmp")



#add_library(${CMAKE_CURRENT_LIST_DIR}/submodules/libxsmm/includ)
#add_subdirectory(submodules/Catch2)
#add_subdirectory(submodules/libxsmm)

set(SUBMODULES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/submodules) 

add_subdirectory(${SUBMODULES_PATH}/Catch2)


add_subdirectory(src)
add_subdirectory(test)

enable_testing()

