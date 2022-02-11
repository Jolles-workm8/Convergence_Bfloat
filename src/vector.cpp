#include <omp.h>

#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "io.hpp"
#include "measurement.hpp"
#include "setup.hpp"

int main() {
  /*
   unsigned int i_m = 0;
   unsigned int i_n = 0;
   unsigned int i_k = 0;


   if (i_argc != 5) {
     std::cerr << "invalid number of arguments" << std::endl;
     return EXIT_FAILURE;
   } else {
     i_m = atoi(i_argv[1]);
     i_n = atoi(i_argv[2]);
     i_k = atoi(i_argv[3]);
     i_iter = atoi(i_argv[4]);
     if (i_m < 1 || i_n < 1 || i_k < 1) {
       std::cerr << "invalid parameters. m, n, k must be positive integers"
                 << std::endl;
       return EXIT_FAILURE;
     }
   }

   */

  std::vector<std::string> header_csv = {"fp32", "Z0", "Z1", "Z2", "Z3", "Z4"};

  size_t i_iter = 1000;

  std::cout << "###################################" << std::endl;
  std::cout << "### Bfloat-16 Error Bounds      ###" << std::endl;
  std::cout << "###                             ###" << std::endl;
  std::cout << "### Julius Isken                ###" << std::endl;
  std::cout << "###################################" << std::endl;

  // Compute the errorbounds for matrices
  std::cout << "Set up matrix measurement" << '\n';

  std::cout << "  Running calculation..." << '\n';

  for (unsigned int i = 2; i <= 64; i *= 2) {
    std::cout << "      i = " << i << '\n';
    auto s = std::to_string(i);
    Setup *matrix = new Setup(i, i, i);

    CSVWriter *matrix_u12_frobenius_writer =
        new CSVWriter(s + "x" + s + "_matrix_frobenius.csv");
    matrix_u12_frobenius_writer->addDatainRow(header_csv.begin(),
                                              header_csv.end());

    for (size_t j = 0; j < i_iter; j++) {
      matrix->random_uniform(-1, 1);
      matrix->XSMM();
      // matrix->GEMM();

      if (j % 100 == 0) {
        std::cout << "*" << std::flush;
      }

      std::vector<double> matrix_frobenius = measure_frobenius(matrix);

      matrix_u12_frobenius_writer->addDatainRow(matrix_frobenius.begin(),
                                                matrix_frobenius.end());
    }
    std::cout << '\n';
    delete matrix_u12_frobenius_writer;
    delete matrix;
  }

  // Compute the errorbounds for matrices
  return 0;
}
