#include "bfloat.hpp"

#include <bitset>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <vector>

TEST_CASE("Float to BFloat", "[fp_to_bf]") {
  float a;
  float b_trunc, b_round, b_int;

  std::cout.precision(17);

  SECTION("Testing integer numbers", "[int]") {
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

  SECTION("Testing rational numbers", "[float]") {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_real_distribution<float> dist{-10000, 1000};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    uint32_t b_as_int;

    for (size_t i = 0; i < 10000; i++) {
      a = gen();
      b_trunc = float_to_bfloat_trunc(a);
      b_round = float_to_bfloat_round(a);
      b_int = float_to_bfloat_intr(a);
      /*
            std::memcpy(&b_as_int, &b_trunc, sizeof(float));
            std::bitset<32> bit_b_trunc{b_as_int};

            std::memcpy(&b_as_int, &b_round, sizeof(float));
            std::bitset<32> bit_b_round{b_as_int};

            std::memcpy(&b_as_int, &b_int, sizeof(float));
            std::bitset<32> bit_b_int{b_as_int};
            /*
                  if (b_round != b_int) {

                    std::cout << "float = " << '\t' << a << std::endl;
                    std::cout << "bfloat_trunc = " << '\t' << b_trunc <<
         std::endl; std::cout << "bfloat_round = " << '\t' << b_round <<
         std::endl; std::cout << "bfloat_int= " << '\t' << b_int << std::endl;

                    std::cout << "bits_trunc = " << '\t' << bit_b_trunc <<
         std::endl; std::cout << "bits_round = " << '\t' << bit_b_round <<
         std::endl; std::cout << "bits_int = " << '\t' << bit_b_int <<
         std::endl;

                    std::cout << "lower dist = " << std::abs(a - b_round) <<
               std::endl; std::cout << "upper dist = " << std::abs(a - b_trunc)
         << std::endl;
                  }
                  */

      REQUIRE(b_int == b_round);
    }
  }
  /*
  undefined behaviour!!!!!!!
    SECTION("Testing edge cases", "[edge]") {
      a = 0;
      b_int = float_to_bfloat_intr(a);
      b_round = float_to_bfloat_round(a);

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
    */
}

TEST_CASE("Float to Bfloat Vector", "[vector]") {
  std::vector<float> b_round;
  std::vector<float> b_intr;
  float a;
  SECTION("Testing random numbers", "[numbers]") {

    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_real_distribution<float> dist{-10000, 1000};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    uint32_t b_as_int;

    for (size_t i = 0; i < 10000; i++) {
      a = gen();
      b_round = float_to_3xbfloat_vector(a);
      b_intr = float_to_3xbfloat_vector_intr(a);

      REQUIRE(b_round == b_intr);
    }
  }
}
