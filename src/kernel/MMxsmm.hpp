#ifndef XSMM_HPP_
#define XSMM_HPP_

#include <libxsmm.h>
#include <omp.h>

#include <algorithm>
#include <iostream>
#include <vector>

void MMxsmm_svanilla(float const *i_a, float const *i_b, float *io_c,
                     unsigned int i_m, unsigned int i_n, unsigned int i_k,
                     unsigned int i_lda, unsigned int i_ldb,
                     unsigned int i_ldc);

void MMxsmm_bfloat(float const *i_a, float const *i_b, float *io_c,
                   unsigned int i_m, unsigned int i_n, unsigned int i_k,
                   unsigned int i_lda, unsigned int i_ldb, unsigned int i_ldc);

void gen_bf_matrices(const float *src, libxsmm_bfloat16 *bf_0,
                     libxsmm_bfloat16 *bf_1, libxsmm_bfloat16 *bf_2,
                     unsigned int size);

template <std::size_t K, std::size_t M>
void vnni_swap(libxsmm_bfloat16 src[K][M], libxsmm_bfloat16 dest[K / 2][M][2]) {
  for (size_t i = 0; i < K; i += 2) {
    for (size_t j = 0; j < M; j++) {
      dest[i / 2][j][0] = src[i][j];
      dest[i / 2][j][1] = src[i + 1][j];
    }
  }
};
#endif