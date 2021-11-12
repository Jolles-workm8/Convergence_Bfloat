
add_executable(bfloat.test bfloat.test.cpp)
add_executable(MMxsmm.test MMxsmm.test.cpp)
add_executable(vanilla.test vanilla.test.cpp)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    target_link_libraries(vanilla.test LINK_PUBLIC Catch2::Catch2WithMain bfloat vanilla OpenMP::OpenMP_CXX)
    target_link_libraries(MMxsmm.test LINK_PUBLIC Catch2::Catch2WithMain MMxsmm vanilla OpenMP::OpenMP_CXX)
else()
    target_link_libraries(vanilla.test LINK_PUBLIC Catch2::Catch2WithMain bfloat vanilla)
    target_link_libraries(MMxsmm.test LINK_PUBLIC Catch2::Catch2WithMain MMxsmm vanilla)
endif()

target_link_libraries(bfloat.test LINK_PUBLIC Catch2::Catch2WithMain bfloat)