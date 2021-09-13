#include "bfloat.hpp"

#include <catch2/catch_all.hpp>
#include <iterator>
#include <random>

TEST_CASE("Float to BFloat") {
  float a;
  float b, b_int;

  SECTION("Testing some basic numbers") {
    a = 0;

    for (size_t i; i < 20; i++) {
      a = i;
      b = float_to_bfloat(a);
      b_int = float_to_bfloat_intr(a);

      REQUIRE(b == b_int);
    }
  }

  SECTION("Testing some random numbers") {
    // First create an instance of an engine.
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{rnd_device()};  // Generates random floats
    std::uniform_real_distribution<float> dist{-1000, 1000};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    a = gen();
    a = 0.5f;
    b = float_to_bfloat(a);
    b_int = float_to_bfloat_intr(a);

        std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    std::cout << "b_int = " << b_int << std::endl;

    REQUIRE(b == b_int);
  }
}