#include "bfloat.hpp"

#include <bitset>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <vector>

TEST_CASE("Float to BFloat", "[number]") {
  float a;
  float b_trunc, b_round, b_int;

  std::cout.precision(17);

  SECTION("Testing integer numbers", "int") {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_int_distribution<int> dist{10000, 10000};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    for (size_t i = 0; i < 10000; i++) {
      a = (float)gen();

      b_trunc = float_to_bfloat_trunc(a);
      b_round = float_to_bfloat_round(a);
      b_int = float_to_bfloat_intr(a);

      REQUIRE(b_int == b_round);
    }
  }

  SECTION("Testing rational numbers", "float") {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_real_distribution<float> dist{-10000, 1000};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    for (size_t i = 0; i < 10000; i++) {
      a = gen();
      b_trunc = float_to_bfloat_trunc(a);
      b_round = float_to_bfloat_round(a);
      b_int = float_to_bfloat_intr(a);

      REQUIRE(b_int == b_round);
    }
  }

  SECTION("Testing edge cases", "edge") {
    a = 0;
    b_int = float_to_bfloat_intr(a);
    b_round = float_to_bfloat_round(a);
    b_trunc = float_to_bfloat_trunc(a);

    REQUIRE(b_round == b_int);

    a = std::numeric_limits<float>::lowest();
    b_int = float_to_bfloat_intr(a);
    b_round = float_to_bfloat_round(a);

    REQUIRE(b_round == b_int);

    a = std::numeric_limits<float>::min();
    b_int = float_to_bfloat_intr(a);
    b_round = float_to_bfloat_round(a);

    REQUIRE(b_round == b_int);

    a = std::numeric_limits<float>::max();
    b_int = float_to_bfloat_intr(a);
    b_round = float_to_bfloat_round(a);

    REQUIRE(b_round == b_int);
  }
}
TEST_CASE("Float to Bfloat Vector", "[vector]") {
  float a;
  float b_intr, b_round;
  std::vector<float> v_intr(3);
  std::vector<float> v_round(3);
  SECTION("Testing random numbers", "numbers") {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_real_distribution<float> dist{-10000, 1000};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    uint32_t b_as_int;

    for (size_t i = 0; i < 10000; i++) {
      a = gen();
      v_round = float_to_3xbfloat_vector(a);
      v_intr = float_to_3xbfloat_vector_intr(a);
      REQUIRE(v_round == v_intr);
    }
  }

  SECTION("Testing edge numbers", "edge") {
    a = 0;
    v_round = float_to_3xbfloat_vector(a);
    v_intr = float_to_3xbfloat_vector_intr(a);

    REQUIRE(v_intr.size() == v_round.size());

    for (size_t i = 0; i < 2; i++) {
      REQUIRE(v_round[i] == v_intr[i]);
    }

    a = std::numeric_limits<float>::lowest();
    v_round = float_to_3xbfloat_vector(a);
    v_intr = float_to_3xbfloat_vector_intr(a);

    REQUIRE(v_intr.size() == v_round.size());

    for (size_t i = 0; i < 2; i++) {
      REQUIRE(v_round[i] == v_intr[i]);
    }

    REQUIRE(v_intr.size() == v_round.size());

    for (size_t i = 0; i < 2; i++) {
      REQUIRE(v_round[i] == v_intr[i]);
    }

    a = std::numeric_limits<float>::min();
    v_round = float_to_3xbfloat_vector(a);
    v_intr = float_to_3xbfloat_vector_intr(a);

    REQUIRE(v_intr.size() == v_round.size());

    for (size_t i = 0; i < 2; i++) {
      REQUIRE(v_round[i] == v_intr[i]);
    }

    a = std::numeric_limits<float>::max();
    v_round = float_to_3xbfloat_vector(a);
    v_intr = float_to_3xbfloat_vector_intr(a);
  }
}
