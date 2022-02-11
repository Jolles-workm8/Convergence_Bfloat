#ifndef _MEASUREMENT_HPP_
#define _MEASUREMENT_HPP_
#include "io.hpp"
#include "setup.hpp"
#include <cmath>
#include <iostream>
#include <vector>

std::vector<double> measure_total(Setup *classname);

std::vector<double> measure_frobenius(Setup *classname);

template <typename T1, typename T2>
double static total_error(std::vector<T1> &v1, std::vector<T2> &v2) {
  double total_error = 0.0;
  for (size_t i = 0; i < v1.size(); i++) {
    total_error += std::abs((double)v1.at(i) - (double)v2.at(i));
  }
  return total_error;
}

template <typename T1, typename T2>
double static max_error(std::vector<T1> &v1, std::vector<T2> &v2) {
  double max_error = 0.0;
  for (size_t i = 0; i < v1.size(); i++) {
    if (std::abs(v1.at(i) - v2.at(i)) > max_error) {
      max_error = std::abs((double)v1.at(i) - (double)v2.at(i));
    }
  }
  return max_error;
}

template <typename T>
std::vector<double> static average_error(std::vector<T> &v, size_t const size) {
  std::vector<double> result(v.size());

#pragma omp parallel for
  for (size_t i = 0; i < v.size(); i++) {
    result.at(i) = ((double)v.at(i) / (double)size);
  }

  return result;
}

template <typename T> double frobenius_norm(std::vector<T> &v) {
  double result = 0.0;

#pragma omp simd
  for (size_t i = 0; i < v.size(); i++) {
    result += (double)v.at(i) * (double)v.at(i);
  }

  return std::sqrt(result);
}

template <typename T1, typename T2>
std::vector<double> substract(std::vector<T1> &v1, std::vector<T2> &v2) {
  if (v1.size() != v2.size()) {
    std::cerr << "invalid vector sizes" << '\n';
  }
  std::vector<double> result(v1.size(), 0);

#pragma omp parallel for
  for (size_t i = 0; i < v1.size(); i++) {
    result.at(i) = (double)v1.at(i) - (double)v2.at(i);
  }

  return result;
}

template <typename T1, typename T2>
double frobenius_error(std::vector<T1> &ref, std::vector<T2> &measured) {
  std::vector<double> v = substract(ref, measured);

  return frobenius_norm(v) / frobenius_norm(measured);
}

#endif /* end of include guard: _MEASUREMENT_HPP_ */
