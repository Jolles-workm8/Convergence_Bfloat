#ifndef BFLOAT_HPP_
#define BFLOAT_HPP_

#include <bitset>
#include <cstring>
#include <iostream>
#include <vector>

float float_to_bfloat(float i_fp32);

std::vector<float> float_to_3xbfloat_vector(float i_a);

template <typename T>
void multiplication_bfloat(std::vector<float> &bf_a, std::vector<float> &bf_b,
                           int i_operations, T &result) {

  // cast as typename
  result += static_cast<T>(bf_a.at(0)) * static_cast<T>(bf_b.at(0));
  result += static_cast<T>(bf_a.at(0)) * static_cast<T>(bf_b.at(1));
  result += static_cast<T>(bf_a.at(1)) * static_cast<T>(bf_b.at(0));

  if (i_operations == 6) {
    result += static_cast<T>(bf_a.at(0)) * static_cast<T>(bf_b.at(2));
    result += static_cast<T>(bf_a.at(2)) * static_cast<T>(bf_b.at(0));
    result += static_cast<T>(bf_a.at(1)) * static_cast<T>(bf_b.at(1));
  }
  if (i_operations == 9) {
    result += static_cast<T>(bf_a.at(1)) * static_cast<T>(bf_b.at(2));
    result += static_cast<T>(bf_a.at(2)) * static_cast<T>(bf_b.at(1));
    result += static_cast<T>(bf_a.at(2)) * static_cast<T>(bf_b.at(2));
  }
}

#endif /* end of include guard: BFLOAT_HPP_ */
