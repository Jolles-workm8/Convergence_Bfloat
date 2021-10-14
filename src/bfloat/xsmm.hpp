#ifndef XSMM_HPP_
#define XSMM_HPP_

#include <libxsmm.h>
#include <omp.h>

#include <vector>

void xsmm_vanilla(float const *i_a, float const *i_b, float *io_c,
                  unsigned int i_m, unsigned int i_n, unsigned int i_k,
                  unsigned int i_lda, unsigned int i_ldb, unsigned int i_ldc) {
  int flags = LIBXSMM_GEMM_FLAG_NONE;
  double *alpha = (double)1;
  double *beta = (double)1;
  /* generates and dispatches a matrix multiplication kernel */
  libxsmm_smmfunction kernel = libxsmm_dmmdispatch(
      i_m, i_n, i_k, NULL /*lda*/, NULL /*ldb*/, NULL /*ldc*/, &alpha, &beta,
      &flags, NULL /*prefetch*/);
  assert(kernel);
  /* kernel multiplies and accumulates matrices: C += Ai * Bi */
  kernel(i_a + i_m * i_k, i_b + i_k * i_n, io_c);
};

#endif