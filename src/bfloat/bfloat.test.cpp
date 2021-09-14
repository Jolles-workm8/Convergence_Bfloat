#include "bfloat.hpp"

#include <bitset>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <iostream>
#include <iterator>
#include <random>

TEST_CASE("Float to BFloat") {
  float a;
  float b, b_int;

  std::cout.precision(17);

  SECTION("Testing integer numbers with truncation") {
    for (size_t i = 0; i < 10; i++) {
      a = i;
      b = float_to_bfloat(a);
      b_int = float_to_bfloat_intr(a);

      REQUIRE(b == b_int);
    }
  }

  SECTION("Testing random rational numbers with truncation") {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_real_distribution<float> dist{-1000, 1000};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    uint32_t b_as_int;
    uint32_t b_int_as_int;

    std::memcpy(&b_as_int, &b, sizeof(float));
    std::memcpy(&b_int_as_int, &b_int, sizeof(float));

    for (size_t i = 0; i < 10000; i++) {
      a = gen();
      // a = 0.5f;

      b = float_to_bfloat_round(a);
      b_int = float_to_bfloat_intr(a);

      std::memcpy(&b_as_int, &b, sizeof(float));
      std::memcpy(&b_int_as_int, &b_int, sizeof(float));

      std::bitset<32> bit_b{b_as_int};
      std::bitset<32> bit_b_int{b_int_as_int};

      if (b != b_int) {
        std::cout << "float = " << '\t' << a << std::endl;
        std::cout << "next float = " << '\t'
                  << nextafter(std::abs(a), std::abs(a + 1)) << std::endl;
        std::cout << "bfloat_trunc = " << '\t' << b << std::endl;
        std::cout << "bfloat_rne = " << '\t' << b_int << std::endl;

        std::cout << "bits_trunc = " << '\t' << bit_b << std::endl;
        std::cout << "bits_rne = " << '\t' << bit_b_int << std::endl;
      }

      if (a > 0) {
        REQUIRE(b <= a);
      } else {
        REQUIRE(b >= a);
      }
    }
  }

  SECTION("Testing random rational number with truncation against RNE") {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_real_distribution<float> dist{-1000, 1000};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    /*
        for (size_t i = 0; i < 10000; i++) {
          a = gen();
          // a = 0.5f;

          b = float_to_bfloat_round(a);
          b_int = float_to_bfloat_intr(a);

          // calclaute hlaf rounding error 0.00391 * 0.5 = 0.001955
          // TODO calc biggest e between 2 bf16 use numeric limits
          if (a > 0) {
            if (std::abs(a) > std::abs(b) + 0.00391) {
              REQUIRE(b != b_int);
            }
          }
          if (a < 0) {
            if (std::abs(a) < std::abs(b) + 0.00391) {
              REQUIRE(b != b_int);
            }
          }

          // b_int is more accurate, why?
          // intel conversion operator not just cuts the end of the mantissa.
       its
          // rounding somehow

          // REQUIRE(b == b_int);
        }
        */
  }
}