#include "bfloat.hpp"

float float_to_bfloat(float i_fp32) {
  // transpose the FP32 to BF16 virtually by cutting the mantissa.

  uint32_t float_as_int;
  float o_bf16;

  std::memcpy(&float_as_int, &i_fp32, sizeof(float));

  std::bitset<32> l_bf_bit{float_as_int};

  for (size_t i = 0; i < 16; i++) {
    l_bf_bit.reset(i);
  }

  float_as_int = l_bf_bit.to_ulong();

  std::memcpy(&o_bf16, &float_as_int, sizeof(float));

  return o_bf16;
}
float float_to_bfloat_round(float i_fp32) {
  // TODO: Locate the upper and lower bfloat to the fp_32 input. Round up/down
  // to earest neighbour. Rounding down is like above, so implement switch case
  // to evaluate first and then set bits.

  // i_fp32 *= 1.001957f;

  uint32_t float_as_int;
  float o_bf16;

  std::memcpy(&float_as_int, &i_fp32, sizeof(float));

  std::bitset<32> l_bf_bit{float_as_int};
  std::bitset<32> l_bf_bit_next;

  // substract the last 16 bits, so first 8 bits of mantissa are left.

  for (size_t i = 0; i < 16; i++) {
    l_bf_bit.reset(i);
  }

  l_bf_bit_next = l_bf_bit;

  for (size_t i = 16; i < 31; i++) {
    l_bf_bit_next.flip(i);
    if (l_bf_bit_next.test(i)) {
      break;
    }
  }

  float_as_int = l_bf_bit.to_ulong();

  std::memcpy(&o_bf16, &float_as_int, sizeof(float));

  return o_bf16;
}

float float_to_bfloat_intr(float i_fp32) {
  float l_v[4];
  float l_w[4];

  l_v[0] = i_fp32;
  l_w[0] = 1;

  for (size_t i = 1; i < 3; i++) {
    l_v[i] = 0;
    l_w[i] = 0;
  }

  __m128 l_a = _mm_loadu_ps(l_v);
  __m128 l_b = _mm_loadu_ps(l_w);

  __m128bh l_bfa = _mm_cvtneps_pbh(l_a);
  __m128bh l_bfb = _mm_cvtneps_pbh(l_b);

  __m128 l_c = _mm_dpbf16_ps(l_c, l_bfa, l_bfb);

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
  bfloat[0] = float_to_bfloat(i_fp32);

  // compute second bfloat
  float intermediate = i_fp32 - bfloat[0];
  bfloat[1] = float_to_bfloat(intermediate);

  // compute third bfloat
  intermediate -= bfloat[1];
  bfloat[2] = float_to_bfloat(intermediate);
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
  std::vector<float> v;

  v.push_back(i_fp32);
  return v;
};
