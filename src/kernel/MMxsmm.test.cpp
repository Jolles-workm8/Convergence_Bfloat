#include "MMxsmm.hpp"

#include <catch2/catch_all.hpp>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <vector>

#include "vanilla.hpp"

TEST_CASE("GeMM with libxsmm") {
  int constexpr n = 10000;
  int constexpr m = 10000;
  int constexpr k = 10000;

  int lda = m;
  int ldb = n;
  int ldc = k;
  std::vector<float> a(n * m);
  std::vector<float> b(n * k);
  std::vector<float> c_ref(m * k, 0);
  std::vector<float> c_xsmm(m * k, 0);

  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()};  // Generates random floats
  std::uniform_real_distribution<float> dist{0, 1};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

  generate(a.begin(), a.end(), gen);
  generate(b.begin(), b.end(), gen);

  gemms_ref(a.data(), b.data(), c_ref.data(), m, n, k, lda, ldb, ldc);
  MMxsmm_vanilla(a.data(), b.data(), c_xsmm.data(), m, n, k, lda, ldb, ldc);

  for (size_t i = 0; i < c_ref.size(); i++) {
    REQUIRE(c_ref[i] == c_xsmm[i]);
  }
}
