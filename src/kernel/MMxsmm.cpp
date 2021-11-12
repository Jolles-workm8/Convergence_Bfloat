#include "MMxsmm.hpp"

void MMxsmm_svanilla(float const *i_a, float const *i_b, float *io_c,
                     unsigned int i_m, unsigned int i_n, unsigned int i_k,
                     unsigned int i_lda, unsigned int i_ldb,
                     unsigned int i_ldc) {
  int flags = LIBXSMM_GEMM_FLAG_NONE;
  const float alpha = 1;
  const float beta = 1;

  libxsmm_smmfunction kernel = libxsmm_smmdispatch(
      i_m, i_n, i_k, NULL, NULL, NULL, &beta, &alpha, &flags, NULL);

  assert(kernel);
  // kernel multiplies and accumulates matrices: C += Ai * Bi
  kernel(i_a, i_b, io_c);
}

void MMxsmm_bfloat(float const *i_a, float const *i_b, float *io_c,
                   unsigned int i_m, unsigned int i_n, unsigned int i_k,
                   unsigned int i_lda, unsigned int i_ldb, unsigned int i_ldc) {
  // const int flags = 16384;
  int flags = LIBXSMM_GEMM_FLAG_NONE;
  const float alpha = 1;
  const float beta = 1;

  libxsmm_bfloat16 l_a0[i_m * i_k];
  libxsmm_bfloat16 l_a1[i_m * i_k];
  libxsmm_bfloat16 l_a2[i_m * i_k];
  libxsmm_bfloat16 l_b0[i_k * i_n];
  libxsmm_bfloat16 l_b1[i_k * i_n];
  libxsmm_bfloat16 l_b2[i_k * i_n];

  libxsmm_bsmmfunction kernel = libxsmm_bsmmdispatch(
      i_m, i_n, i_k, NULL, NULL, NULL, &beta, &alpha, &flags, NULL);

  gen_bf_matrices(i_a, l_a0, l_a1, l_a2, i_m * i_k);
  gen_bf_matrices(i_b, l_b0, l_b1, l_b2, i_k * i_n);

  kernel(l_a0, l_b0, io_c);
  kernel(l_a0, l_b1, io_c);
  kernel(l_a1, l_b0, io_c);
  kernel(l_a1, l_b1, io_c);
  kernel(l_a0, l_b2, io_c);
  kernel(l_a2, l_b0, io_c);
  /*
    kernel(l_a1, l_b2, io_c);
    kernel(l_a2, l_b1, io_c);
    kernel(l_a2, l_b2, io_c);
    */
}

void gen_bf_matrices(const float *src, libxsmm_bfloat16 *bf_0,
                     libxsmm_bfloat16 *bf_1, libxsmm_bfloat16 *bf_2,
                     unsigned int size) {
  std::vector<float> intermediate(size, 0);
  std::vector<float> copy(size, 0);

  libxsmm_rne_convert_fp32_bf16(src, bf_0, size);
  libxsmm_convert_bf16_f32(bf_0, copy.data(), size);

  for (size_t i = 0; i < size; i++) {
    intermediate[i] = src[i] - copy[i];
  }

  libxsmm_rne_convert_fp32_bf16(intermediate.data(), bf_1, size);
  libxsmm_convert_bf16_f32(bf_1, copy.data(), size);

  for (size_t i = 0; i < size; i++) {
    intermediate[i] -= copy[i];
  }
  libxsmm_rne_convert_fp32_bf16(intermediate.data(), bf_2, size);
}

/**
[m] *[k] colmajor;
[k0 k1];
[ a00, a01, a02, a03..];
[ a00, a10, a01, a11, a02, a12, ..];
**/