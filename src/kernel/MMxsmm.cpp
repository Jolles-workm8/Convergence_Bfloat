#include "MMxsmm.hpp"

void MMxsmm_svanilla(float const *i_a, float const *i_b, float *io_c,
                     unsigned int i_m, unsigned int i_n, unsigned int i_k,
                     unsigned int i_lda, unsigned int i_ldb,
                     unsigned int i_ldc) {
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const float alpha = 1;
  const float beta = 1;
  /*
    libxsmm_smmfunction kernel = libxsmm_smmdispatch(
        i_m, i_n, i_k, NULL, NULL, NULL, &beta, &alpha, &flags, NULL);
  */

  libxsmm_gemm_prefetch_type i_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

  // add description
  libxsmm_descriptor_blob l_xgemmBlob;
  const libxsmm_gemm_descriptor *l_desc = 0;
  // const int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  l_desc = libxsmm_gemm_descriptor_dinit(
      &l_xgemmBlob, LIBXSMM_GEMM_PRECISION_F32, i_m, i_n, i_k, i_lda, i_ldb,
      i_ldc, alpha, beta, flags, i_prefetch);

  libxsmm_smmfunction kernel = libxsmm_xmmdispatch(l_desc).smm;

  assert(kernel);
  // kernel multiplies and accumulates matrices: C += Ai * Bi
  kernel(i_a, i_b, io_c);
}

void MMxsmm_bfloat(float const *i_a, float const *i_b, float *io_c,
                   unsigned int i_m, unsigned int i_n, unsigned int i_k,
                   unsigned int i_lda, unsigned int i_ldb, unsigned int i_ldc) {
  const float alpha = 1;
  const float beta = 1;

  std::vector<float> l_a;
  std::vector<float> temp_a;
  std::vector<float> l_b;

  if (i_k % 2 == 1) {
    l_a.resize((i_k + 1) * i_m, 0);
    temp_a.resize((i_k + 1) * i_m, 0);
    l_b.resize((i_k + 1) * i_n, 0);
    std::memcpy(temp_a.data(), i_a, i_k * i_m * sizeof(float));
    vnni_swap(temp_a.data(), l_a.data(), i_k + 1, i_m);
    for (size_t l_n = 0; l_n < i_n; l_n++) {
      for (size_t l_k = 0; l_k < i_k; l_k++) {
        l_b[l_k + l_n * (i_k + 1)] = i_b[l_k + l_n * i_k];
      }
    }
    i_k++;
  } else {
    l_a.resize(i_k * i_m, 0);
    temp_a.resize(i_k * i_m, 0);
    l_b.resize(i_k * i_n, 0);
    std::memcpy(temp_a.data(), i_a, i_k * i_m * sizeof(float));
    vnni_swap(temp_a.data(), l_a.data(), i_k, i_m);
    std::memcpy(l_b.data(), i_b, i_k * i_n * sizeof(float));
  }

  libxsmm_bfloat16 l_a0[i_m * i_k];
  libxsmm_bfloat16 l_a1[i_m * i_k];
  libxsmm_bfloat16 l_a2[i_m * i_k];
  libxsmm_bfloat16 l_b0[i_k * i_n];
  libxsmm_bfloat16 l_b1[i_k * i_n];
  libxsmm_bfloat16 l_b2[i_k * i_n];

  libxsmm_gemm_prefetch_type i_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

  // add description
  libxsmm_descriptor_blob l_xgemmBlob;
  const libxsmm_gemm_descriptor *l_desc = 0;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N') | LIBXSMM_GEMM_FLAG_VNNI_A;

  l_desc = libxsmm_gemm_descriptor_dinit2(
      &l_xgemmBlob, LIBXSMM_GEMM_PRECISION_BF16, LIBXSMM_GEMM_PRECISION_F32,
      i_m, i_n, i_k, i_lda, i_ldb, i_ldc, alpha, beta, l_flags, i_prefetch);

  libxsmm_bsmmfunction kernel = libxsmm_xmmdispatch(l_desc).bsmm;

  // assert(kernel);

  gen_bf_matrices(l_a.data(), l_a0, l_a1, l_a2, i_m * i_k);
  gen_bf_matrices(l_b.data(), l_b0, l_b1, l_b2, i_n * i_k);

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

void gen_bf_matrices(float *src, libxsmm_bfloat16 *bf_0, libxsmm_bfloat16 *bf_1,
                     libxsmm_bfloat16 *bf_2, unsigned int size) {
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

void vnni_swap(float *src, float *dest, size_t K, size_t M) {
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

  if (K % 2) {
    for (size_t l_m = 0; l_m < M; l_m++) {
      dest[2 * l_m + (K - 1) * M] = src[l_m + (K - 1) * M];
      dest[2 * l_m + (K - 1) * M + 1] = 0;
    }
  }
}
