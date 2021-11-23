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

void MMxsmm_bfloat(float *i_a, float *i_b, float *io_c, unsigned int i_m,
                   unsigned int i_n, unsigned int i_k, unsigned int i_lda,
                   unsigned int i_ldb, unsigned int i_ldc);

void gen_bf_matrices(float *src, libxsmm_bfloat16 *bf_0, libxsmm_bfloat16 *bf_1,
                     libxsmm_bfloat16 *bf_2, unsigned int size);

void swap_pointer(float *i_a, float *i_b);

// Assume that src is a matrix of the form src[m][k] in storaged in column
// major;
void vnni(float *a, float *b, unsigned int i_m, unsigned int i_n,
          unsigned int i_k);

template <typename T>
void vnni_swap(T *src, T *dest, size_t K, size_t M) {
  /*
  for (size_t l_k = 0; l_k < K - (K % 2); l_k += 2) {
    for (size_t l_m = 0; l_m < M; l_m++) {
      dest[l_k / 2][l_m][0] = src[l_k][l_m];
      dest[l_k / 2][l_m][1] = src[l_k + 1][l_m];
    }
  }
  */

  for (size_t l_k = 0; l_k < K; l_k += 2) {
    for (size_t l_m = 0; l_m < M; l_m++) {
      dest[2 * l_m + l_k * M] = src[l_m + l_k * M];
      dest[2 * l_m + l_k * M + 1] = src[l_m + (l_k + 1) * M];
    }
  }
};
#endif