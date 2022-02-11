#include "MMxsmm.hpp"

void MMxsmm_svanilla(float const *i_a, float const *i_b, float *io_c,
                     unsigned int i_m, unsigned int i_n, unsigned int i_k,
                     unsigned int i_lda, unsigned int i_ldb,
                     unsigned int i_ldc) {
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const float alpha = 1;
  const float beta = 0;
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

void MMxsmm_dvanilla(float const *i_a, float const *i_b, double *io_c,
                     unsigned int i_m, unsigned int i_n, unsigned int i_k,
                     unsigned int i_lda, unsigned int i_ldb,
                     unsigned int i_ldc) {
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const double alpha = 1;
  const double beta = 0;

  double *l_a;
  double *l_b;
  l_a = (double *)std::malloc(sizeof(double) * i_m * i_k);
  l_b = (double *)std::malloc(sizeof(double) * i_k * i_n);

#pragma omp simd
  for (size_t i = 0; i < (i_m * i_k); i++) {
    l_a[i] = i_a[i];
  }
#pragma omp simd
  for (size_t i = 0; i < (i_m * i_k); i++) {
    l_b[i] = i_b[i];
  }

  libxsmm_gemm_prefetch_type i_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

  // add description
  libxsmm_descriptor_blob l_xgemmBlob;
  const libxsmm_gemm_descriptor *l_desc = 0;
  // const int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  l_desc = libxsmm_gemm_descriptor_dinit(
      &l_xgemmBlob, LIBXSMM_GEMM_PRECISION_F64, i_m, i_n, i_k, i_lda, i_ldb,
      i_ldc, alpha, beta, flags, i_prefetch);

  libxsmm_dmmfunction kernel = libxsmm_xmmdispatch(l_desc).dmm;

  assert(kernel);
  // kernel multiplies and accumulates matrices: C += Ai * Bi
  kernel(l_a, l_b, io_c);
}

void MMxsmm_bfloat(float const *i_a, float const *i_b, float *io_c,
                   unsigned int i_m, unsigned int i_n, unsigned int i_k,
                   unsigned int i_lda, unsigned int i_ldb, unsigned int i_ldc,
                   unsigned int i_approx_lvl) {
  const float alpha = 1;
  const float beta = 1;

  std::vector<float> l_a;
  std::vector<float> temp_a;
  std::vector<float> l_b;

#pragma omp simd
  for (size_t i = 0; i < i_m * i_n; i++) {
    io_c[i] = 0;
  }

  if (i_k % 2 == 1) {
    l_a.resize((i_k + 1) * i_m, 0);
    l_b.resize((i_k + 1) * i_n, 0);

    vnni_swap(i_a, l_a.data(), i_k, i_m);
    for (size_t l_n = 0; l_n < i_n; l_n++) {
      for (size_t l_k = 0; l_k < i_k; l_k++) {
        l_b[l_k + l_n * (i_k + 1)] = i_b[l_k + l_n * i_k];
      }
    }
    i_k++;
    i_ldb++;
  } else {
    l_a.resize(i_k * i_m, 0);
    l_b.resize(i_k * i_n, 0);

    vnni_swap(i_a, l_a.data(), i_k, i_m);
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
  if (i_approx_lvl > 0) {
    kernel(l_a0, l_b1, io_c);
    kernel(l_a1, l_b0, io_c);
  }
  if (i_approx_lvl > 1) {
    kernel(l_a1, l_b1, io_c);
    kernel(l_a0, l_b2, io_c);
    kernel(l_a2, l_b0, io_c);
  }
  if (i_approx_lvl > 2) {
    kernel(l_a1, l_b2, io_c);
    kernel(l_a2, l_b1, io_c);
  }
  if (i_approx_lvl > 3) {
    kernel(l_a2, l_b2, io_c);
  }
}

void gen_bf_matrices(float *src, libxsmm_bfloat16 *bf_0, libxsmm_bfloat16 *bf_1,
                     libxsmm_bfloat16 *bf_2, unsigned int size) {
  std::vector<float> intermediate(size, 0);
  std::vector<float> copy(size, 0);

  libxsmm_rne_convert_fp32_bf16(src, bf_0, size);
  libxsmm_convert_bf16_f32(bf_0, copy.data(), size);

#pragma omp simd
  for (size_t i = 0; i < size; i++) {
    intermediate[i] = src[i] - copy[i];
  }

  libxsmm_rne_convert_fp32_bf16(intermediate.data(), bf_1, size);
  libxsmm_convert_bf16_f32(bf_1, copy.data(), size);
#pragma omp simd
  for (size_t i = 0; i < size; i++) {
    intermediate[i] -= copy[i];
  }
  libxsmm_rne_convert_fp32_bf16(intermediate.data(), bf_2, size);
}
