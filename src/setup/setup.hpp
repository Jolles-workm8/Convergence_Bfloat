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

#include "MMxsmm.hpp"
#include "bfloat.hpp"
#include "vanilla.hpp"

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

  std::vector<double> l_c_ref_fp64;
  std::vector<float> l_c_ref_fp32;
  std::vector<float> l_c_bf_3;
  std::vector<float> l_c_bf_6;
  std::vector<float> l_c_bf_9;

  Setup(size_t i_m, size_t i_n, size_t i_k);

  ~Setup();

  void random_expo(float lambda);

  void random_uniform(float i_min, float i_max);

  void random_normal(float i_mean, float i_var);

  void GEMM();

  void XSMM();
};

#endif /* end of include guard: _SETUP_HPP_ */
