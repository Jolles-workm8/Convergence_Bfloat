#ifndef BFLOAT_HPP_
#define BFLOAT_HPP_

#include <immintrin.h>

#include <bitset>
#include <cmath>
#include <cstring>
#include <iostream>
#include <ostream>
#include <vector>

float float_to_bfloat_trunc(float i_fp32);
float float_to_bfloat_round(float i_fp32);
float float_to_bfloat_intr(float i_fp32);

std::vector<float> float_to_3xbfloat_vector(float i_a);
std::vector<float> float_to_3xbfloat_vector_intr(float i_fp32);

template <typename T>
void multiplication_bfloat(std::vector<float> &bf_a, std::vector<float> &bf_b,
                           int i_approx_level, T &result) {
  // cast as typename
  
  if(i_approx_level < 1) {
  result += static_cast<T>(bf_a.at(0)) * static_cast<T>(bf_b.at(0));
  }
  if(i_approx_level < 2) {
  result += static_cast<T>(bf_a.at(0)) * static_cast<T>(bf_b.at(1));
  result += static_cast<T>(bf_a.at(1)) * static_cast<T>(bf_b.at(0));
  }
  if (i_approx_level < 3) {
    result += static_cast<T>(bf_a.at(0)) * static_cast<T>(bf_b.at(2));
    result += static_cast<T>(bf_a.at(2)) * static_cast<T>(bf_b.at(0));
    result += static_cast<T>(bf_a.at(1)) * static_cast<T>(bf_b.at(1));
  }
  if (i_approx_level < 4) {
    result += static_cast<T>(bf_a.at(1)) * static_cast<T>(bf_b.at(2));
    result += static_cast<T>(bf_a.at(2)) * static_cast<T>(bf_b.at(1));
  }
  if (i_approx_level < 5) {
    result += static_cast<T>(bf_a.at(2)) * static_cast<T>(bf_b.at(2));
  }
}

#endif /* end of include guard: BFLOAT_HPP_ */
