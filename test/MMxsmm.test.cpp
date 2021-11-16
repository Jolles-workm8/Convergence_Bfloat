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

TEST_CASE("VNNI SWAP", "[vnni]") {
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
  SECTION("uneven m, uneven k") {
    const size_t k = 3;
    const size_t m = 3;

    int input[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    int output[9];

    int output_expected[9] = {0, 3, 1, 4, 2, 5, 6, 7, 8};

    vnni_swap(input, output, k, m);
    for (size_t i = 0; i < 9; i++) {
      REQUIRE(output[i] == output_expected[i]);
    }
  }
}

TEST_CASE("Small int GeMM with libxsmm", "[small][int]") {
  std::cout << "simple int" << std::endl;
  float a[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  float a_v[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  float b[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float c_bf[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  float c_ref[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // vnni_swap(a, a_v, 3, 3);
  MMxsmm_bfloat(a, b, c_bf, 3, 3, 3, 3, 3, 3);
  MMxsmm_svanilla(a, b, c_ref, 3, 3, 3, 3, 3, 3);

  for (size_t i = 0; i < 9; i++) {
    CHECK(c_bf[i] == c_ref[i]);
  }

  std::cout << std::endl;
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      std::cout << c_ref[i + 3 * j] << '\t';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      std::cout << c_bf[i + 3 * j] << '\t';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
TEST_CASE("Small real GeMM with libxsmm", "[small][real]") {
  float a[9] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5};
  float a_v[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  float b[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  float c_bf[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  float c_van[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  float c_ref[9] = {10.5, 13.5, 16.5, 10.5, 13.5, 16.5, 10.5, 13.5, 16.5};

  MMxsmm_bfloat(a, b, c_bf, 3, 3, 3, 3, 3, 3);
  MMxsmm_svanilla(a, b, c_van, 3, 3, 3, 3, 3, 3);
  for (size_t i = 0; i < 9; i++) {
    CHECK(c_ref[i] == c_van[i]);
    CHECK(c_bf[i] == c_ref[i]);
  }
}
TEST_CASE("Big int GeMM with libxsmm", "[big][int]") {
  std::cout.precision(17);

  int m, n, k, lda, ldb, ldc;
  int count = 25;

  for (n = 1; n < count; n++) {
    for (k = 1; k < count; k++) {
      for (m = 1; m < count; m++) {
        SECTION("n = " + std::to_string(n) + ", m = " + std::to_string(m) +
                ", k = " + std::to_string(k)) {
          lda = m;
          ldb = k;
          ldc = m;

          std::vector<float> a(m * k, 0);
          std::vector<float> a_v(m * k, 1);
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
                        ldc);

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
  int count = 25;
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()};  // Generates random floats
  std::uniform_real_distribution<float> dist{0, 100};

  for (n = 1; n < count; n++) {
    for (k = 1; k < count; k++) {
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
                        ldc);

          for (size_t i = 0; i < c_ref.size(); i++) {
            CHECK(c_ref[i] == Catch::Approx(c_xsmm[i]).epsilon(0.1));
            CHECK(c_xsmm[i] == Catch::Approx(c_xsmm_bf[i]).epsilon(0.1));
          }
        }
      }
    }
  }
}

/*
  SECTION("random numbers", "real") {
    // First create an instance of an engine.
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{rnd_device()};  // Generates random floats
    std::uniform_real_distribution<float> dist{1, 100};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    generate(a.begin(), a.end(), gen);
    generate(b.begin(), b.end(), gen);

    // vnni_swap<k, m>(a.data(), a_vnni.data());

    gemms_ref(a.data(), b.data(), c_ref.data(), m, n, k, lda, ldb, ldc);
    gemmd_ref(a.data(), b.data(), c_dref.data(), m, n, k, lda, ldb, ldc);
    gemm_bfloat(a.data(), b.data(), c_bf.data(), m, n, k, lda, ldb, ldc, 6);
    MMxsmm_svanilla(a.data(), b.data(), c_xsmm.data(), m, n, k, lda, ldb,
  ldc); MMxsmm_bfloat(a.data(), b.data(), c_xsmm_bf.data(), m, n, k, lda, ldb,
  ldc);

    for (size_t i = 0; i < c_ref.size(); i++) {
      std::cout << "i = " << '\t' << i << '\t' << "c_double" << '\t'
                << c_dref[i] << std::endl;
      std::cout << "i = " << '\t' << i << '\t' << "c_reference" << '\t'
                << c_ref[i] << std::endl;
      std::cout << "i = " << '\t' << i << '\t' << "c_bfloat" << '\t' <<
   c_bf[i]
                << std::endl;
      std::cout << "i = " << '\t' << i << '\t' << "c_xsmm_ref" << '\t'
                << c_xsmm[i] << std::endl;
      std::cout << "i = " << '\t' << i << '\t' << "c_xsmm_bf" << '\t'
                << c_xsmm_bf[i] << std::endl;

// CHECK(c_ref[i] == c_bf[i]);
// CHECK(c_xsmm[i] == c_xsmm_bf[i]);
}
}
}
*/
