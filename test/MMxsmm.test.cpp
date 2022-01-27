#define CATCH_CONFIG_MAIN

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

TEST_CASE("VNNI") {}

TEST_CASE("VNNI SWAP", "[vnni], [swap]") {
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

    vnni_swap(input, output, k, m);

    for (size_t i = 0; i < 16; i++) {
      REQUIRE(output[i] == output_expected[i]);
    }
  }
  SECTION("uneven m, even k") {
    /** colmajor
     * 0, 3, 6, 9
     * 1, 4, 7, 10
     * 2, 5, 8, 11
     **/
    const size_t k = 4;
    const size_t m = 3;

    int input[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    int output[12];

    int output_expected[12] = {0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11};

    vnni_swap(input, output, k, m);
    for (size_t i = 0; i < 12; i++) {
      REQUIRE(output[i] == output_expected[i]);
    }
  }
}

TEST_CASE("Small int GeMM with libxsmm", "[small][int]") {
  float a[4] = {0, 1, 2, 3};
  float b[4] = {1, 1, 1, 1};
  float c_bf[4] = {0, 0, 0, 0};
  float c_ref[4] = {0, 0, 0, 0};

  MMxsmm_bfloat(a, b, c_bf, 2, 2, 2, 2, 2, 2, 4);
  MMxsmm_svanilla(a, b, c_ref, 2, 2, 2, 2, 2, 2);

  for (size_t i = 0; i < 3; i++) {
    CHECK(c_bf[i] == c_ref[i]);
  }
}
TEST_CASE("Small real GeMM with libxsmm", "[small][real]") {
  float a[4] = {0.5, 1.5, 2.5, 3.5};
  float b[4] = {1, 1, 1, 1};
  float c_bf[4] = {0, 0, 0, 0};
  float c_van[4] = {0, 0, 0, 0};
  float c_ref[4] = {3, 5, 3, 5};

  MMxsmm_bfloat(a, b, c_bf, 2, 2, 2, 2, 2, 2, 4);
  MMxsmm_svanilla(a, b, c_van, 2, 2, 2, 2, 2, 2);
  for (size_t i = 0; i < 4; i++) {
    CHECK(c_ref[i] == c_van[i]);
    CHECK(c_bf[i] == c_ref[i]);
  }
}
TEST_CASE("Big int GeMM with libxsmm", "[big][int]") {
  std::cout.precision(17);

  int m, n, k, lda, ldb, ldc;
  int count = 32;

  for (n = 1; n < count; n++) {
    for (k = 2; k < count; k++) {
      for (m = 1; m < count; m++) {
        SECTION("n = " + std::to_string(n) + ", m = " + std::to_string(m) +
                ", k = " + std::to_string(k)) {
          lda = m;
          ldb = k;
          ldc = m;

          std::vector<float> a(m * k, 0);
          std::vector<float> b(k * n, 0);
          std::vector<float> c_ref(m * n, 0);
          // std::vector<double> c_dref(m * n, 0);
          // std::vector<float> c_bf(m * n, 0);
          std::vector<float> c_xsmm(m * n, 0);
          std::vector<float> c_xsmm_bf(m * n, 0);

          std::iota(a.begin(), a.end(), 1);
          std::iota(b.begin(), b.end(), 1);

          gemms_ref(a.data(), b.data(), c_ref.data(), m, n, k, lda, ldb, ldc);

          MMxsmm_svanilla(a.data(), b.data(), c_xsmm.data(), m, n, k, lda, ldb,
                          ldc);

          MMxsmm_bfloat(a.data(), b.data(), c_xsmm_bf.data(), m, n, k, lda, ldb,
                        ldc, 2);

          for (size_t i = 0; i < c_ref.size(); i++) {
            CHECK(c_ref[i] == c_xsmm[i]);
            CHECK(c_xsmm[i] == c_xsmm_bf[i]);
          }
        }
      }
    }
  }
}

TEST_CASE("Big real GeMM with libxsmm", "[big][real]") {
  int m, n, k, lda, ldb, ldc;
  int count = 32;
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()};  // Generates random floats
  std::uniform_real_distribution<float> dist{0, 100};

  for (n = 1; n < count; n++) {
    for (k = 2; k < count; k++) {
      for (m = 1; m < count; m++) {
        SECTION("n = " + std::to_string(n) + ", m = " + std::to_string(m) +
                ", k = " + std::to_string(k)) {
          lda = m;
          ldb = k;
          ldc = m;

          std::vector<float> a(m * k, 0);
          std::vector<float> a_v(m * k, 0);
          std::vector<float> b(k * n, 0);
          std::vector<float> c_ref(m * n, 0);
          std::vector<double> c_dref(m * n, 0);
          std::vector<float> c_bf(m * n, 0);
          std::vector<float> c_xsmm(m * n, 0);
          std::vector<float> c_xsmm_bf(m * n, 0);

          auto gen = [&dist, &mersenne_engine]() {
            return dist(mersenne_engine);
          };

          generate(a.begin(), a.end(), gen);
          generate(b.begin(), b.end(), gen);

          gemms_ref(a.data(), b.data(), c_ref.data(), m, n, k, lda, ldb, ldc);

          MMxsmm_svanilla(a.data(), b.data(), c_xsmm.data(), m, n, k, lda, ldb,
                          ldc);

          MMxsmm_bfloat(a.data(), b.data(), c_xsmm_bf.data(), m, n, k, lda, ldb,
                        ldc, 2);

          for (size_t i = 0; i < c_ref.size(); i++) {
            CHECK(c_ref[i] == Catch::Approx(c_xsmm[i]).epsilon(0.1));
            CHECK(c_xsmm[i] == Catch::Approx(c_xsmm_bf[i]).epsilon(0.1));
          }
        }
      }
    }
  }
}
