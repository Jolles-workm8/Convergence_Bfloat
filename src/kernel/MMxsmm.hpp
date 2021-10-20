#ifndef XSMM_HPP_
#define XSMM_HPP_

#include <libxsmm.h>
#include <omp.h>

#include <vector>

void MMxsmm_vanilla(float const *i_a, float const *i_b, float *io_c,
                    unsigned int i_m, unsigned int i_n, unsigned int i_k,
                    unsigned int i_lda, unsigned int i_ldb, unsigned int i_ldc);

#endif