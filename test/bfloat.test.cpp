#include "bfloat.hpp"
#include "vector_bfloat.hpp"
#include "MMxsmm.hpp"

#include <bitset>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <vector>
#include <libxsmm.h>


TEST_CASE("Float to BFloat", "[number]") {
  float a;
  float b_round, b_int;

  std::cout.precision(17);

  SECTION("Testing integer numbers", "int") {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_int_distribution<int> dist{-10000, 10000};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    for (size_t i = 0; i < 10000; i++) {
      a = (float)gen();

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

      b_round = float_to_bfloat_round(a);
      b_int = float_to_bfloat_intr(a);

      REQUIRE(b_int == b_round);
    }
  }

  SECTION("Testing edge cases", "edge") {
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
}
TEST_CASE("Float to Bfloat Vector", "[vector]") {
  float a;
  std::vector<float> v_intr(3);
  std::vector<float> v_round(3);

  SECTION("Testing random numbers", "numbers") {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_real_distribution<float> dist{-10000, 1000};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

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

TEST_CASE("Decomposition with vector instructions", "[vector_instr]"){
   
    float a[2050];
    //generate some odd flaoting point values without randomization
    for (size_t i = 0; i < 2050; i++)
    {
      a[i] =i*0.212;
    }
    
    libxsmm_bfloat16 bf0[2][2050];
    libxsmm_bfloat16 bf1[2][2050];
    libxsmm_bfloat16 bf2[2][2050];
    float fp_bf0[2][2050];
    float fp_bf1[2][2050];
    float fp_bf2[2][2050];
    

    split_compress(a, bf0[0], bf1[0], bf2[0], 2050);
    gen_bf_matrices(a, bf0[1], bf1[1], bf2[1], 2050);

    for (size_t i = 0; i < 2; i++)
    {
    libxsmm_convert_bf16_f32(bf0[i], fp_bf0[i], 2050);
    libxsmm_convert_bf16_f32(bf1[i], fp_bf1[i], 2050);
    libxsmm_convert_bf16_f32(bf2[i], fp_bf2[i], 2050);    
    }
    
    


    for (size_t i = 0; i < 2050; i++)
    {
      CHECK(fp_bf0[0][i]==fp_bf0[1][i]);
      CHECK(fp_bf1[0][i]==fp_bf1[1][i]);
      CHECK(fp_bf2[0][i]==fp_bf2[1][i]);
    }
}