#include <omp.h>

#include <iostream>
#include <limits>
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

  std::vector<std::string> header_csv = {"fp64", "fp32", "Z0", "Z1",
                                         "Z2",   "Z3",   "Z4"};

  size_t i_iter = 10000;

  std::cout << "###################################" << std::endl;
  std::cout << "### Bfloat-16 Error Bounds      ###" << std::endl;
  std::cout << "###                             ###" << std::endl;
  std::cout << "### Julius Isken                ###" << std::endl;
  std::cout << "###################################" << std::endl;

  // Init CSVWriter class for scalar

  std::cout << "Set up scalar measurement" << '\n';

  CSVWriter *scalar_avg_writer = new CSVWriter("scalar_average.csv");
  CSVWriter *scalar_total_writer = new CSVWriter("scalar_total.csv");
  CSVWriter *scalar_frobenius_writer = new CSVWriter("scalar_frobenius.csv");

  scalar_total_writer->addDatainRow(header_csv.begin(), header_csv.end());
  scalar_avg_writer->addDatainRow(header_csv.begin(), header_csv.end());
  scalar_frobenius_writer->addDatainRow(header_csv.begin(), header_csv.end());

  std::cout << "  Running calculation..." << '\n';

  // Compute the Errorbounds for Scalars
  Setup *scalar = new Setup(1, 1, 1);

  for (size_t j = 0; j < i_iter; j++) {
    scalar->random_normal(0, 1);
    if (j % 100 == 0) {
      std::cout << "*" << std::flush;
    }
    scalar->GEMM();
    scalar->XSMM();
  }

  std::cout << '\n';

  // measure results

  std::vector<double> scalar_total = measure_total(scalar);
  std::vector<double> scalar_average = average_error(scalar_total, 1);
  std::vector<double> scalar_frobenius = measure_frobenius(scalar);

  scalar_avg_writer->addDatainRow(scalar_average.begin(), scalar_average.end());
  scalar_total_writer->addDatainRow(scalar_total.begin(), scalar_total.end());
  scalar_frobenius_writer->addDatainRow(scalar_frobenius.begin(),
                                        scalar_frobenius.end());

  delete scalar;

  // Compute the errorbounds for vectors
  std::cout << "Set up vector measurement" << '\n';

  CSVWriter *vector_avg_writer = new CSVWriter("vector_average.csv");
  CSVWriter *vector_total_writer = new CSVWriter("vector_total.csv");
  CSVWriter *vector_frobenius_writer = new CSVWriter("vector_frobenius.csv");

  vector_total_writer->addDatainRow(header_csv.begin(), header_csv.end());
  vector_avg_writer->addDatainRow(header_csv.begin(), header_csv.end());
  vector_frobenius_writer->addDatainRow(header_csv.begin(), header_csv.end());

  std::cout << "  Running calculation..." << '\n';

  for (unsigned int i = 2; i <= 16; i *= 2) {
    std::cout << "      i = " << i << '\n';
    Setup *vector = new Setup(1, i, 1);
    vector->random_uniform(1, 2);
    for (size_t j = 0; j < i_iter; j++) {
      if (j % 100 == 0) {
        std::cout << "*" << std::flush;
      }
      vector->GEMM();
      // vector->XSMM();
    }

    std::cout << '\n';

    std::vector<double> vector_total = measure_total(vector);
    std::vector<double> vector_average = average_error(vector_total, i * i);
    std::vector<double> vector_frobenius = measure_frobenius(vector);

    vector_avg_writer->addDatainRow(vector_average.begin(),
                                    vector_average.end());
    vector_total_writer->addDatainRow(vector_total.begin(), vector_total.end());
    vector_frobenius_writer->addDatainRow(vector_frobenius.begin(),
                                          vector_frobenius.end());

    delete vector;
  }

  // Compute the errorbounds for matrices
  std::cout << "Set up matrix measurement" << '\n';

  CSVWriter *matrix_avg_writer = new CSVWriter("matrix_average.csv");
  CSVWriter *matrix_total_writer = new CSVWriter("matrix_total.csv");
  CSVWriter *matrix_frobenius_writer = new CSVWriter("matrix_frobenius.csv");

  matrix_total_writer->addDatainRow(header_csv.begin(), header_csv.end());
  matrix_avg_writer->addDatainRow(header_csv.begin(), header_csv.end());
  matrix_frobenius_writer->addDatainRow(header_csv.begin(), header_csv.end());

  std::cout << "  Running calculation..." << '\n';

  for (unsigned int i = 2; i <= 16; i *= 2) {
    std::cout << "      i = " << i << '\n';
    Setup *matrix = new Setup(i, i, i);
    matrix->random_normal(0, 1);

    for (size_t j = 0; j < i_iter; j++) {
      if (j % 100 == 0) {
        std::cout << "*" << std::flush;
      }
      matrix->GEMM();
      matrix->XSMM();
    }

    std::cout << '\n';

    // erst fehler rechenn dann average nehmen!!!

    std::vector<double> matrix_total = measure_total(matrix);
    std::vector<double> matrix_average = average_error(matrix_total, 1);
    std::vector<double> matrix_frobenius = measure_frobenius(matrix);

    matrix_avg_writer->addDatainRow(matrix_average.begin(),
                                    matrix_average.end());
    matrix_total_writer->addDatainRow(matrix_total.begin(), matrix_total.end());
    matrix_frobenius_writer->addDatainRow(matrix_frobenius.begin(),
                                          matrix_frobenius.end());

    delete matrix;
  }

  return 0;
}
