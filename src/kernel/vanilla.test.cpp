#include "vanilla.hpp"

#include <catch2/catch_all.hpp>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <vector>

TEST_CASE("GeMM reference kernel", "[ref]") {
  int constexpr n = 32;
  int constexpr m = 32;
  int constexpr k = 32;

  int lda = m;
  int ldb = n;
  int ldc = k;
  std::vector<float> a(n * m, 0);
  std::vector<float> b(n * k, 1);

  SECTION("Testing single precision", "single") {
    std::vector<float> c(m * k, 0);
    gemms_ref(a.data(), b.data(), c.data(), m, n, k, lda, ldb, ldc);

    for (size_t i = 1; i < c.size(); i++) {
      REQUIRE((c[i] == 0));
    }

    std::fill(a.begin(), a.end(), 1);
    std::fill(b.begin(), b.end(), 1);

    gemms_ref(a.data(), b.data(), c.data(), m, n, k, lda, ldb, ldc);

    for (size_t i = 1; i < c.size(); i++) {
      REQUIRE(c[i] == 32);
    }

    gemms_ref(a.data(), b.data(), c.data(), m, n, k, lda, ldb, ldc);

    for (size_t i = 1; i < c.size(); i++) {
      REQUIRE(c[i] == 64);
    }

    std::fill(a.begin(), a.end(), -1);

    gemms_ref(a.data(), b.data(), c.data(), m, n, k, lda, ldb, ldc);

    for (size_t i = 1; i < c.size(); i++) {
      REQUIRE(c[i] == 32);
    }
  }
  SECTION("Testing double precision", "double") {
    std::vector<double> c(m * k, 0);
    gemmd_ref(a.data(), b.data(), c.data(), m, n, k, lda, ldb, ldc);

    for (size_t i = 1; i < c.size(); i++) {
      REQUIRE((c[i] == 0));
    }

    std::fill(a.begin(), a.end(), 1);
    std::fill(b.begin(), b.end(), 1);

    gemmd_ref(a.data(), b.data(), c.data(), m, n, k, lda, ldb, ldc);

    for (size_t i = 1; i < c.size(); i++) {
      REQUIRE(c[i] == 32);
    }

    gemmd_ref(a.data(), b.data(), c.data(), m, n, k, lda, ldb, ldc);

    for (size_t i = 1; i < c.size(); i++) {
      REQUIRE(c[i] == 64);
    }

    std::fill(a.begin(), a.end(), -1);

    gemmd_ref(a.data(), b.data(), c.data(), m, n, k, lda, ldb, ldc);

    for (size_t i = 1; i < c.size(); i++) {
      REQUIRE(c[i] == 32);
    }
  }
}

TEST_CASE("Testing bfloat kernel", "[bfloat]") {
  int constexpr n = 32;
  int constexpr m = 32;
  int constexpr k = 32;

  int lda = m;
  int ldb = n;
  int ldc = k;

  std::vector<float> a(n * m, 0);
  std::vector<float> b(n * k, 1);
  std::vector<float> c(m * k, 0);
  std::vector<float> c_ref(m * k, 0);

  SECTION("Testing integer numbers") {
    gemm_bfloat(a.data(), b.data(), c.data(), m, n, k, lda, ldb, ldc, 6);

    for (size_t i = 1; i < c.size(); i++) {
      REQUIRE((c[i] == 0));
    }

    std::fill(a.begin(), a.end(), 1);
    std::fill(b.begin(), b.end(), 1);

    gemm_bfloat(a.data(), b.data(), c.data(), m, n, k, lda, ldb, ldc, 6);

    for (size_t i = 1; i < c.size(); i++) {
      REQUIRE(c[i] == 32);
    }

    gemm_bfloat(a.data(), b.data(), c.data(), m, n, k, lda, ldb, ldc, 6);

    for (size_t i = 1; i < c.size(); i++) {
      REQUIRE(c[i] == 64);
    }

    std::fill(a.begin(), a.end(), -1);

    gemm_bfloat(a.data(), b.data(), c.data(), m, n, k, lda, ldb, ldc, 6);

    for (size_t i = 1; i < c.size(); i++) {
      REQUIRE(c[i] == 32);
    }
  }
  SECTION("Compare with reference kernel") {
    // First create an instance of an engine.
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{rnd_device()};  // Generates random floats
    std::uniform_real_distribution<float> dist{0, 1};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    for (size_t i = 0; i < 1000; i++) {
      generate(a.begin(), a.end(), gen);
      generate(b.begin(), b.end(), gen);

      gemm_bfloat(a.data(), b.data(), c.data(), m, n, k, lda, ldb, ldc, 6);
      gemms_ref(a.data(), b.data(), c_ref.data(), m, n, k, lda, ldb, ldc);

      for (size_t i = 1; i < c.size(); i++) {
        REQUIRE(c[i] == c_ref[i]);
      }
    }
  }
}