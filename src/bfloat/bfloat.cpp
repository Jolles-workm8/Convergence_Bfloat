#include "bfloat.hpp"

float float_to_bfloat_trunc(float i_fp32) {
  // transpose the FP32 to BF16 virtually by cutting the mantissa.

  uint32_t float_as_int;
  float o_bf16;

  std::memcpy(&float_as_int, &i_fp32, sizeof(float));

  std::bitset<32> l_a{float_as_int};
  std::bitset<32> l_b = 0xffff0000;

  l_a = l_a & l_b;

  float_as_int = l_a.to_ulong();

  std::memcpy(&o_bf16, &float_as_int, sizeof(float));

  return o_bf16;
}

float float_to_bfloat_round(float i_fp32) {
  if (std::isnan(i_fp32) || std::isinf(i_fp32)) {
    return i_fp32;
  }
  float o_bf16, o_bf16n;
  bool negative = false;
  uint32_t float_as_int;

  std::memcpy(&float_as_int, &i_fp32, sizeof(float));

  std::bitset<32> l_a{float_as_int};
  std::bitset<32> l_b = 0xffff0000;

  l_a = l_a & l_b;
  l_b = l_a;

  for (size_t i = 16; i < 32; i++) {
    if (l_a.test(i)) {
      l_b.reset(i);
    } else {
      l_b.set(i);
      break;
    }
  }

  float_as_int = l_a.to_ulong();
  std::memcpy(&o_bf16, &float_as_int, sizeof(float));

  float_as_int = l_b.to_ulong();
  std::memcpy(&o_bf16n, &float_as_int, sizeof(float));

  if (std::isinf(o_bf16n)) {
    return o_bf16n;
  }

  if (std::abs(i_fp32 - o_bf16n) < std::abs(i_fp32 - o_bf16)) {
    return o_bf16n;
  } else if (std ::abs(i_fp32 - o_bf16n) > std::abs(i_fp32 - o_bf16)) {
    return o_bf16;
  } else if (std::abs(i_fp32 - o_bf16n) == std::abs(i_fp32 - o_bf16)) {
    if (l_a.test(16)) {
      return o_bf16n;
    } else {
      return o_bf16;
    }
  }
}

float float_to_bfloat_intr(float i_fp32) {
  float l_v[4];
  float l_w[4];
  float l_y[4];

  l_y[0] = 0.0f;
  l_v[0] = i_fp32;
  l_w[0] = 1.0f;

  for (size_t i = 1; i < 3; i++) {
    l_v[i] = 0.0f;
    l_w[i] = 0.0f;
    l_y[i] = 0.0f;
  }

  __m128 l_a = _mm_loadu_ps(l_v);
  __m128 l_b = _mm_loadu_ps(l_w);
  __m128 l_c = _mm_loadu_ps(l_y);

  __m128bh l_bfa = _mm_cvtneps_pbh(l_a);
  __m128bh l_bfb = _mm_cvtneps_pbh(l_b);

  l_c = _mm_dpbf16_ps(l_c, l_bfa, l_bfb);

  _mm_store_ps(l_w, l_c);
  /*
    for (size_t i = 0; i < 4; i++) {
      std::cout << "l_w" << i << " = " << l_w[i] << std::endl;
      std::cout << "l_v" << i << " = " << l_v[i] << std::endl;
    }
    */

  return l_w[0];
}

std::vector<float> float_to_3xbfloat_vector(float i_fp32) {
  std::vector<float> bfloat = {0.0f, 0.0f, 0.0f};

  // compute first bfloat
  bfloat[0] = float_to_bfloat_round(i_fp32);

  // compute second bfloat
  float intermediate = i_fp32 - bfloat[0];
  bfloat[1] = float_to_bfloat_round(intermediate);

  // compute third bfloat
  intermediate -= bfloat[1];
  bfloat[2] = float_to_bfloat_round(intermediate);
  /*
    std::cout << "floating point value:" << '\t' << i_a << '\n';

    std::cout << "bfloat_vector = { ";
    for (float n : bfloat) {
      std::cout << n << ", ";
    }
    std::cout << "}; \n";
  */
  return bfloat;
}

std::vector<float> float_to_3xbfloat_vector_intr(float i_fp32) {
  std::vector<float> bfloat = {0.0f, 0.0f, 0.0f};

  // compute first bfloat
  bfloat[0] = float_to_bfloat_intr(i_fp32);

  // compute second bfloat
  float intermediate = i_fp32 - bfloat[0];
  /*
  std::cout << "bf_0 = " << bfloat[0] << std::endl;
  std::cout << "intermediate = " << intermediate << std::endl;
  */
  bfloat[1] = float_to_bfloat_intr(intermediate);
  /*
  std::cout << "bf_1 = " << bfloat[1] << std::endl;
  std::cout << "intermediate = " << intermediate << std::endl;
  */
  // compute third bfloat
  intermediate -= bfloat[1];
  bfloat[2] = float_to_bfloat_intr(intermediate);
  /*
  std::cout << "bf_2 = " << bfloat[2] << std::endl;
  std::cout << "intermediate = " << intermediate << std::endl;
  */
  /*
    std::cout << "floating point value:" << '\t' << i_a << '\n';

    std::cout << "bfloat_vector = { ";
    for (float n : bfloat) {
      std::cout << n << ", ";
    }
    std::cout << "}; \n";
  */
  return bfloat;
};
