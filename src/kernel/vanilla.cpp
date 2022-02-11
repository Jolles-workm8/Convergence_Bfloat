#include "vanilla.hpp"

void gemms_ref(float const *i_a, float const *i_b, float *io_c,
               unsigned int i_m, unsigned int i_n, unsigned int i_k,
               unsigned int i_lda, unsigned int i_ldb, unsigned int i_ldc) {
  //#pragma omp parallel for
  for (unsigned int l_k = 0; l_k < i_k; l_k++) {
    for (unsigned int l_n = 0; l_n < i_n; l_n++) {
      for (unsigned int l_m = 0; l_m < i_m; l_m++) {
        io_c[l_m + l_n * i_m] += i_a[l_m + l_k * i_m] * i_b[l_k + l_n * i_k];
      }
    }
  }
}

void gemmd_ref(float const *i_a, float const *i_b, double *io_c,
               unsigned int i_m, unsigned int i_n, unsigned int i_k,
               unsigned int i_lda, unsigned int i_ldb, unsigned int i_ldc) {
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

  for (unsigned int l_m = 0; l_m < i_m; l_m++) {
    for (unsigned int l_n = 0; l_n < i_n; l_n++) {
      for (unsigned int l_k = 0; l_k < i_k; l_k++) {
        io_c[l_m + l_n * i_m] += l_a[l_m + l_k * i_m] * l_b[l_k + l_n * i_k];
      }
    }
  }
  free(l_a);
  free(l_b);
}
