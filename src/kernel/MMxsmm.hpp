#ifndef XSMM_HPP_
#define XSMM_HPP_

#include <libxsmm.h>
#include <omp.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

void MMxsmm_svanilla(float const *i_a, float const *i_b, float *io_c,
                     unsigned int i_m, unsigned int i_n, unsigned int i_k,
                     unsigned int i_lda, unsigned int i_ldb,
                     unsigned int i_ldc);

void MMxsmm_bfloat(float const *i_a, float const *i_b, float *io_c,
                   unsigned int i_m, unsigned int i_n, unsigned int i_k,
                   unsigned int i_lda, unsigned int i_ldb, unsigned int i_ldc);

void gen_bf_matrices(float *src, libxsmm_bfloat16 *bf_0, libxsmm_bfloat16 *bf_1,
                     libxsmm_bfloat16 *bf_2, unsigned int size);

template <typename T>
void vnni_swap(T const *src, T *dest, size_t K, size_t M) {
  for (size_t l_k = 0; l_k < K; l_k += 2) {
    for (size_t l_m = 0; l_m < M; l_m++) {
      dest[2 * l_m + l_k * M] = src[l_m + l_k * M];
      dest[2 * l_m + l_k * M + 1] = src[l_m + (l_k + 1) * M];
    }
  }

  if (K % 2 == 1) {
    for (size_t l_m = 0; l_m < M; l_m++) {
      dest[2 * l_m + (K - 1) * M] = src[l_m + (K - 1) * M];
      dest[2 * l_m + (K - 1) * M + 1] = 0;
    }
  }
};
#endif