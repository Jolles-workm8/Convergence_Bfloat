#include "MMxsmm.hpp"

void MMxsmm_vanilla(float const *i_a, float const *i_b, float *io_c,
                    unsigned int i_m, unsigned int i_n, unsigned int i_k,
                    unsigned int i_lda, unsigned int i_ldb,
                    unsigned int i_ldc) {
  int flags = LIBXSMM_GEMM_FLAG_NONE;
  const double alpha = 1;
  const double beta = 1;

  libxsmm_descriptor_blob l_xgemmBlob;
  const libxsmm_gemm_descriptor *l_desc = 0;

  libxsmm_mmfunction<float, float> kernel(flags, i_m, i_n, i_k, 1.0, 1.0);

  kernel(i_a, i_b, io_c);
  // l_desc = libxsmm_gemm_descriptor_dinit(
  //    &l_xgemmBlob, LIBXSMM_GEMM_PRECISION_F32, i_m, i_n, i_k, i_lda, i_ldb,
  //    i_ldc, alpha, beta, flags, NULL);

  // libxsmm_xmmdispatch( .smm);

  // assert(kernel);
  /* kernel multiplies and accumulates matrices: C += Ai * Bi */
  // kernel(i_a, i_b, io_c);
}