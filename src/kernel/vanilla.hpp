#ifndef VANILLA_HPP_
#define VANILLA_HPP_

#include <immintrin.h>
#include <omp.h>

#include <vector>

#include "bfloat.hpp"

void gemms_ref(float const *i_a, float const *i_b, float *io_c,
               unsigned int i_m, unsigned int i_n, unsigned int i_k,
               unsigned int i_lda, unsigned int i_ldb, unsigned int i_ldc);

void gemmd_ref(float const *i_a, float const *i_b, double *io_c,
               unsigned int i_m, unsigned int i_n, unsigned int i_k,
               unsigned int i_lda, unsigned int i_ldb, unsigned int i_ldc);
template <typename T>
void gemm_bfloat(float const *i_a, float const *i_b, T *io_c, unsigned int i_m,
                 unsigned int i_n, unsigned int i_k, unsigned int i_lda,
                 unsigned int i_ldb, unsigned int i_ldc, int i_approx_lvl) {
#pragma omp parallel for
  for (size_t l_m = 0; l_m < i_m; l_m++) {
    for (size_t l_n = 0; l_n < i_n; l_n++) {
      for (size_t l_k = 0; l_k < i_k; l_k++) {
        T result = 0;
        std::vector<float> l_abf =
            float_to_3xbfloat_vector(i_a[l_m + l_k * i_m]);
        std::vector<float> l_bbf =
            float_to_3xbfloat_vector(i_b[l_k + l_n * i_k]);
        multiplication_bfloat(l_abf, l_bbf, i_approx_lvl, result);

#pragma omp critical
        io_c[l_m + l_n * i_m] += result;
      }
    }
  }
}

#endif
