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

  std::vector<std::string> header_csv = {"fp32", "Z0", "Z1", "Z2", "Z3", "Z4"};

  size_t i_iter = 1;

  std::cout << "###################################" << std::endl;
  std::cout << "### Bfloat-16 Error Bounds      ###" << std::endl;
  std::cout << "###                             ###" << std::endl;
  std::cout << "### Julius Isken                ###" << std::endl;
  std::cout << "###################################" << std::endl;

  // Init CSVWriter class for scalar
  /*
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
      //scalar->GEMM();
      alar->XSMM();
    }

    std::cout << '\n';

    // measure results

    std::vector<double> scalar_total = measure_total(scalar);
    std::vector<double> scalar_average = average_error(scalar_total, 1);
    std::vector<double> scalar_frobenius = measure_frobenius(scalar);

    scalar_avg_writer->addDatainRow(scalar_average.begin(),
    scalar_average.end());
    scalar_total_writer->addDatainRow(scalar_total.begin(), scalar_total.end());
    scalar_frobenius_writer->addDatainRow(scalar_frobenius.begin(),
                                          scalar_frobenius.end());

    delete scalar;
  */
  // Compute the errorbounds for matrices
  std::cout << "Set up matrix measurement" << '\n';

  CSVWriter *matrix_u12_avg_writer = new CSVWriter("matrix_u[1,2]_average.csv");
  CSVWriter *matrix_u12_total_writer = new CSVWriter("matrix_u[1,2]_total.csv");
  CSVWriter *matrix_u12_frobenius_writer =
      new CSVWriter("matrix_u[1,2]_frobenius.csv");

  matrix_u12_total_writer->addDatainRow(header_csv.begin(), header_csv.end());
  matrix_u12_avg_writer->addDatainRow(header_csv.begin(), header_csv.end());
  matrix_u12_frobenius_writer->addDatainRow(header_csv.begin(),
                                            header_csv.end());

  std::cout << "  Running calculation..." << '\n';

  for (unsigned int i = 2; i <= 64; i *= 2) {
    std::cout << "      i = " << i << '\n';
    Setup *matrix = new Setup(i, i, i);
    matrix->random_uniform(1, 2);

    for (size_t j = 0; j < i_iter; j++) {
      if (j % 100 == 0) {
        std::cout << "*" << std::flush;
        matrix->XSMM();
      }
      // matrix->GEMM();
    }

    std::cout << '\n';

    // erst fehler rechenn dann average nehmen!!!

    std::vector<double> matrix_total = measure_total(matrix);
    std::vector<double> matrix_average = average_error(matrix_total, i * i);
    std::vector<double> matrix_frobenius = measure_frobenius(matrix);

    matrix_u12_avg_writer->addDatainRow(matrix_average.begin(),
                                        matrix_average.end());
    matrix_u12_total_writer->addDatainRow(matrix_total.begin(),
                                          matrix_total.end());
    matrix_u12_frobenius_writer->addDatainRow(matrix_frobenius.begin(),
                                              matrix_frobenius.end());

    delete matrix;
  }
  delete matrix_u12_avg_writer;
  delete matrix_u12_total_writer;
  delete matrix_u12_frobenius_writer;
  // Compute the errorbounds for matrices
  std::cout << "Set up matrix measurement" << '\n';

  CSVWriter *matrix_range_avg_writer =
      new CSVWriter("matrix_range_average.csv");
  CSVWriter *matrix_range_total_writer =
      new CSVWriter("matrix_range_total.csv");
  CSVWriter *matrix_range_frobenius_writer =
      new CSVWriter("matrix_range_frobenius.csv");

  matrix_range_total_writer->addDatainRow(header_csv.begin(), header_csv.end());
  matrix_range_avg_writer->addDatainRow(header_csv.begin(), header_csv.end());
  matrix_range_frobenius_writer->addDatainRow(header_csv.begin(),
                                              header_csv.end());

  std::cout << "  Running calculation..." << '\n';

  for (unsigned int i = 2; i <= 64; i *= 2) {
    std::cout << "      i = " << i << '\n';
    Setup *matrix = new Setup(i, i, i);
    matrix->random_uniform(0, 2147483647);

    for (size_t j = 0; j < i_iter; j++) {
      if (j % 100 == 0) {
        std::cout << "*" << std::flush;
        matrix->XSMM();
      }
      // matrix->GEMM();
    }

    std::cout << '\n';

    // erst fehler rechenn dann average nehmen!!!

    std::vector<double> matrix_total = measure_total(matrix);
    std::vector<double> matrix_average = average_error(matrix_total, i * i);
    std::vector<double> matrix_frobenius = measure_frobenius(matrix);

    matrix_range_avg_writer->addDatainRow(matrix_average.begin(),
                                          matrix_average.end());
    matrix_range_total_writer->addDatainRow(matrix_total.begin(),
                                            matrix_total.end());
    matrix_range_frobenius_writer->addDatainRow(matrix_frobenius.begin(),
                                                matrix_frobenius.end());

    delete matrix;
  }
  delete matrix_range_avg_writer;
  delete matrix_range_total_writer;
  delete matrix_range_frobenius_writer;
  // Compute the errorbounds for matrices
  std::cout << "Set up matrix measurement" << '\n';

  CSVWriter *matrix_repeat_avg_writer =
      new CSVWriter("matrix_repeat_average.csv");
  CSVWriter *matrix_repeat_total_writer =
      new CSVWriter("matrix_repeat_total.csv");
  CSVWriter *matrix_repeat_frobenius_writer =
      new CSVWriter("matrix_repeat_frobenius.csv");

  matrix_repeat_total_writer->addDatainRow(header_csv.begin(),
                                           header_csv.end());
  matrix_repeat_avg_writer->addDatainRow(header_csv.begin(), header_csv.end());
  matrix_repeat_frobenius_writer->addDatainRow(header_csv.begin(),
                                               header_csv.end());

  std::cout << "  Running calculation..." << '\n';

  for (unsigned int i = 2; i <= 64; i *= 2) {
    std::cout << "      i = " << i << '\n';
    Setup *matrix = new Setup(i, i, i);
    matrix->random_uniform(-1, 1);

    for (size_t j = 0; j < i_iter; j++) {
      if (j % 100 == 0) {
        std::cout << "*" << std::flush;
        matrix->XSMM();
      }
      // matrix->GEMM();
    }

    std::cout << '\n';

    // erst fehler rechenn dann average nehmen!!!

    std::vector<double> matrix_total = measure_total(matrix);
    std::vector<double> matrix_average = average_error(matrix_total, i * i);
    std::vector<double> matrix_frobenius = measure_frobenius(matrix);

    matrix_repeat_avg_writer->addDatainRow(matrix_average.begin(),
                                           matrix_average.end());
    matrix_repeat_total_writer->addDatainRow(matrix_total.begin(),
                                             matrix_total.end());
    matrix_repeat_frobenius_writer->addDatainRow(matrix_frobenius.begin(),
                                                 matrix_frobenius.end());

    delete matrix;
  }
  delete matrix_repeat_avg_writer;
  delete matrix_repeat_total_writer;
  delete matrix_repeat_frobenius_writer;

  return 0;
}
