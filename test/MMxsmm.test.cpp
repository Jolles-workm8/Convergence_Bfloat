#include "MMxsmm.hpp"

#include <omp.h>

#include <catch2/catch_all.hpp>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <vector>

#include "vanilla.hpp"

TEST_CASE("VNNI SWAP", "[vnni]") {
  SECTION("check vnni format") {
    float a[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    float b[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float c[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    float a_swapped[9];

    vnni_swap<3, 3>(a, a_swapped);
    MMxsmm_bfloat(a, b, c, 3, 3, 3, 3, 3, 3);

    for (size_t i = 0; i < 9; i++) {
      REQUIRE(c[i] == a[i]);
    }
  }
  SECTION("random") {
    std::vector<float> a(9, 1);
    float b[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float c[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    // First create an instance of an engine.
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{rnd_device()};  // Generates random floats
    std::uniform_real_distribution<float> dist{1, 100};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    generate(a.begin(), a.end(), gen);

    MMxsmm_bfloat(a.data(), b, c, 3, 3, 3, 3, 3, 3);

    for (size_t i = 0; i < 9; i++) {
      CHECK(c[i] == a[i]);
    }
  }
  SECTION("even m, even k") {
    const size_t k = 4;
    const size_t m = 4;
    /** rowmajor
     * 0, 1, 2, 3,
     * 4, 5, 6, 7,
     * 8, 9, 10, 11,
     * 12, 13, 14, 15
     **/
    /** colmajor
     * 0, 4, 8, 12
     * 1, 5, 9, 13
     * 2, 6, 10, 14
     * 3, 7, 11, 15
     **/
    // so the alogrithm doenst mind row or colmajor
    libxsmm_bfloat16 input[16] = {0, 1, 2,  3,  4,  5,  6,  7,
                                  8, 9, 10, 11, 12, 13, 14, 15};
    libxsmm_bfloat16 output_expected[16] = {0, 4,  1, 5,  2,  6,  3,  7,
                                            8, 12, 9, 13, 10, 14, 11, 15};

    libxsmm_bfloat16 output[16];

    vnni_swap<k, m>(input, output);

    for (size_t i = 0; i < 16; i++) {
      REQUIRE(output[i] == output_expected[i]);
    }
  }
  SECTION("uneven m, uneven k") {
    const size_t k = 3;
    const size_t m = 3;

    int input[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    int output[9];

    int output_expected[9] = {0, 3, 1, 4, 2, 5, 6, 7, 8};

    vnni_swap<k, m>(input, output);
    for (size_t i = 0; i < 9; i++) {
      REQUIRE(output[i] == output_expected[i]);
    }
  }
}

TEST_CASE("GeMM with libxsmm", "[libxsmm]") {
  std::cout.precision(17);
  int constexpr n = 2;
  int constexpr m = 2;
  int constexpr k = 20;

  int lda = m;
  int ldb = n;
  int ldc = k;
  std::vector<float> a(m * k, 1);
  std::vector<float> a_vnni(m * k, 1);
  std::vector<float> b(k * n, 1);
  std::vector<float> c_ref(m * n, 0);
  std::vector<double> c_dref(m * n, 0);
  std::vector<float> c_bf(m * n, 0);
  std::vector<float> c_xsmm(m * n, 0);
  std::vector<float> c_xsmm_bf(m * n, 0);
  SECTION("integer numbers", "int") {
    gemms_ref(a.data(), b.data(), c_ref.data(), m, n, k, lda, ldb, ldc);

    MMxsmm_svanilla(a.data(), b.data(), c_xsmm.data(), m, n, k, lda, ldb, ldc);
    //  MMxsmm_bfloat(a.data(), b.data(), c_xsmm_bf.data(), m, n, k, lda, ldb,
    //  ldc);

    for (size_t i = 0; i < c_ref.size(); i++) {
      // REQUIRE(c_ref[i] == c_xsmm[i]);
      // REQUIRE(c_xsmm[i] == c_xsmm_bf[i]);
    }
  }

  SECTION("random numbers", "real") {
    // First create an instance of an engine.
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{rnd_device()};  // Generates random floats
    std::uniform_real_distribution<float> dist{1, 100};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    generate(a.begin(), a.end(), gen);
    generate(b.begin(), b.end(), gen);

    vnni_swap<k, m>(a.data(), a_vnni.data());

    gemms_ref(a.data(), b.data(), c_ref.data(), m, n, k, lda, ldb, ldc);
    gemmd_ref(a.data(), b.data(), c_dref.data(), m, n, k, lda, ldb, ldc);
    gemm_bfloat(a.data(), b.data(), c_bf.data(), m, n, k, lda, ldb, ldc, 6);
    MMxsmm_svanilla(a.data(), b.data(), c_xsmm.data(), m, n, k, lda, ldb, ldc);
    MMxsmm_bfloat(a.data(), b.data(), c_xsmm_bf.data(), m, n, k, lda, ldb, ldc);

    for (size_t i = 0; i < c_ref.size(); i++) {
      std::cout << "i = " << '\t' << i << '\t' << "c_double" << '\t'
                << c_dref[i] << std::endl;
      std::cout << "i = " << '\t' << i << '\t' << "c_reference" << '\t'
                << c_ref[i] << std::endl;
      std::cout << "i = " << '\t' << i << '\t' << "c_bfloat" << '\t' << c_bf[i]
                << std::endl;
      std::cout << "i = " << '\t' << i << '\t' << "c_xsmm_ref" << '\t'
                << c_xsmm[i] << std::endl;
      std::cout << "i = " << '\t' << i << '\t' << "c_xsmm_bf" << '\t'
                << c_xsmm_bf[i] << std::endl;

      // CHECK(c_ref[i] == c_bf[i]);
      // CHECK(c_xsmm[i] == c_ref[i]);
    }
  }
}
