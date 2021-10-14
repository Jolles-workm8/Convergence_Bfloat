#ifndef VANILLA_HPP_
#define VANILLA_HPP_

#include <immintrin.h>
#include <omp.h>

#include <bfloat.hpp>
#include <vector>

void gemm_ref(float const *i_a, float const *i_b, float *io_c, unsigned int i_m,
              unsigned int i_n, unsigned int i_k, unsigned int i_lda,
              unsigned int i_ldb, unsigned int i_ldc) {
#pragma omp parallel for
  for (unsigned int l_m = 0; l_m < i_m; l_m++) {
    for (unsigned int l_n = 0; l_n < i_n; l_n++) {
      for (unsigned int l_k = 0; l_k < i_k; l_k++) {
        io_c[l_m + l_n * i_ldc] +=
            (i_a[l_m + l_k * i_lda] * i_b[l_k + l_n * i_ldb]);
      }
    }
  }
};

void gemm_bfloat(float const *i_a, float const *i_b, float *io_c,
                 unsigned int i_m, unsigned int i_n, unsigned int i_k,
                 unsigned int i_lda, unsigned int i_ldb, unsigned int i_ldc,
                 int i_operations) {
#pragma omp parallel for
  for (size_t l_m = 0; l_m < i_m; l_m++) {
    for (size_t l_n = 0; l_n < i_n; l_n++) {
      for (size_t l_k = 0; l_k < i_k; l_k++) {
        float result = 0;
        std::vector<float> l_abf =
            float_to_3xbfloat_vector(i_a[l_m + l_k * i_lda]);
        std::vector<float> l_bbf =
            float_to_3xbfloat_vector(i_b[l_k + l_n * i_ldb]);
        multiplication_bfloat(l_abf, l_bbf, i_operations, result);

#pragma omp critical
        io_c[l_m + l_n * i_ldc] += result;
      }
    }
  }
};