#ifndef _SETUP_HPP_
#define _SETUP_HPP_

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <vector>

#include "bfloat.hpp"
#include "vanilla.hpp"
#include "xsmm.hpp"

class Setup {
 private:
 public:
  size_t l_m = 0;
  size_t l_n = 0;
  size_t l_k = 0;
  size_t l_ka = 0;
  size_t l_kb = 0;
  size_t l_lda = 0;
  size_t l_ldb = 0;
  size_t l_ldc = 0;

  size_t size = 0;

  // pointer to init random numbers
  // void (*rnd_fnc_pntr)(){&nullptr};

  std::vector<float> l_a;
  std::vector<float> l_b;

  std::vector<float> l_c_ref_fp32;
  std::vector<double> l_c_ref_fp64;
  std::vector<float> l_c_bf_3;
  std::vector<double> l_c_bf_3d;
  std::vector<float> l_c_bf_6;
  std::vector<double> l_c_bf_6d;
  std::vector<float> l_c_bf_9;
  std::vector<double> l_c_bf_9d;

  Setup(size_t i_m, size_t i_n, size_t i_k);

  ~Setup();

  void random_expo(float lambda);

  void random_uniform(float i_min, float i_max);

  void random_normal(float i_mean, float i_var);

  void GEMM();

  template <typename T>
  void gemm_ref(std::vector<T> &io_c) {
#pragma omp parallel for
    for (size_t l_mi = 0; l_mi < l_m; l_mi++) {
      for (size_t l_ni = 0; l_ni < l_n; l_ni++) {
        for (size_t l_ki = 0; l_ki < l_k; l_ki++) {
          T result = 0;
          result = static_cast<T>(l_a.at(l_mi + l_ki * l_lda)) *
                   static_cast<T>(l_b.at(l_ki + l_ni * l_ldb));

#pragma omp critical
          io_c.at(l_mi + l_ni * l_ldc) += result;
        }
      }
    }
  }

  template <typename T>
  void gemm_bfloat(std::vector<T> &io_c, int i_operations) {
#pragma omp parallel for
    for (size_t l_mi = 0; l_mi < l_m; l_mi++) {
      for (size_t l_ni = 0; l_ni < l_n; l_ni++) {
        for (size_t l_ki = 0; l_ki < l_k; l_ki++) {
          T result = 0;
          std::vector<float> l_abf =
              float_to_3xbfloat_vector(l_a.at(l_mi + l_ki * l_lda));
          std::vector<float> l_bbf =
              float_to_3xbfloat_vector(l_b.at(l_ki + l_ni * l_ldb));
          multiplication_bfloat(l_abf, l_bbf, i_operations, result);

#pragma omp critical
          io_c.at(l_mi + l_ni * l_ldc) += result;
        }
      }
    }
  };
};

#endif /* end of include guard: _SETUP_HPP_ */
