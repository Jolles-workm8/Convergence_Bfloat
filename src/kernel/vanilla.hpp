#ifndef VANILLA_HPP_
#define VANILLA_HPP_

#include <immintrin.h>
#include <omp.h>

#include <vector>

#include "bfloat.hpp"

void gemms_ref(float const *i_a, float const *i_b, float *io_c,
               unsigned int i_m, unsigned int i_n, unsigned int i_k,
               unsigned int i_lda, unsigned int i_ldb, unsigned int i_ldc);

void gemmd_ref(float const *i_a, float const *i_b, double *io_c,
               unsigned int i_m, unsigned int i_n, unsigned int i_k,
               unsigned int i_lda, unsigned int i_ldb, unsigned int i_ldc);

void gemm_bfloat(float const *i_a, float const *i_b, float *io_c,
                 unsigned int i_m, unsigned int i_n, unsigned int i_k,
                 unsigned int i_lda, unsigned int i_ldb, unsigned int i_ldc,
                 int i_operations);
#endif