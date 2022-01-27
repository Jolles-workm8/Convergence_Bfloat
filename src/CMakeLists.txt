add_subdirectory(kernel)
add_subdirectory(io)
add_subdirectory(measurement)
add_subdirectory(setup)


add_executable(bf16MM main.cpp)
add_executable(vector vector.cpp)


  # Link the executable to the matrix/measurement libraries. Since the libraries have
# public include directories we will use those link directories when building
# the driver
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
  target_link_libraries(bf16MM LINK_PUBLIC measurement setup OpenMP::OpenMP_CXX)
  target_link_libraries(vector LINK_PUBLIC measurement setup OpenMP::OpenMP_CXX)
else()
  target_link_libraries(bf16MM LINK_PUBLIC measurement setup)
  target_link_libraries(vector LINK_PUBLIC measurement setup)
endif()
