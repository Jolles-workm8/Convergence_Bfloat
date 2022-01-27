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

  // Compute the errorbounds for vectors
  std::cout << "Set up vector measurement u[1,2]" << '\n';

  CSVWriter *u12_vector_avg_writer = new CSVWriter("u[1,2]_vector_average.csv");
  CSVWriter *u12_vector_total_writer = new CSVWriter("u[1,2]_vector_total.csv");
  CSVWriter *u12_vector_frobenius_writer =
      new CSVWriter("u[1,2]_vector_frobenius.csv");

  u12_vector_total_writer->addDatainRow(header_csv.begin(), header_csv.end());
  u12_vector_avg_writer->addDatainRow(header_csv.begin(), header_csv.end());
  u12_vector_frobenius_writer->addDatainRow(header_csv.begin(),
                                            header_csv.end());

  std::cout << "  Running calculation..." << '\n';

  for (unsigned int i = 2; i <= 16; i *= 2) {
    std::cout << "      i = " << i << '\n';
    Setup *u12vector = new Setup(1, i, 1);
    u12vector->random_uniform(1, 2);
    for (size_t j = 0; j < i_iter; j++) {
      if (j % 100 == 0) {
        std::cout << "*" << std::flush;
      }
      u12vector->GEMM();
      // u12vector->XSMM();
    }

    std::cout << '\n';

    std::vector<double> vector_total = measure_total(u12vector);
    std::vector<double> vector_average = average_error(vector_total, i * i);
    std::vector<double> vector_frobenius = measure_frobenius(u12vector);

    u12_vector_avg_writer->addDatainRow(vector_average.begin(),
                                        vector_average.end());
    u12_vector_total_writer->addDatainRow(vector_total.begin(),
                                          vector_total.end());
    u12_vector_frobenius_writer->addDatainRow(vector_frobenius.begin(),
                                              vector_frobenius.end());

    delete[] u12vector;
  }

  delete[] u12_vector_avg_writer;
  delete[] u12_vector_total_writer;
  delete[] u12_vector_frobenius_writer;

  // Compute the errorbounds for vectors
  std::cout << "Set up vector measurement large negative exponent" << '\n';

  CSVWriter *ln_vector_avg_writer =
      new CSVWriter("large_neg_vector_average.csv");
  CSVWriter *ln_vector_total_writer =
      new CSVWriter("large_neg_vector_total.csv");
  CSVWriter *ln_vector_frobenius_writer =
      new CSVWriter("large_neg_vector_frobenius.csv");

  ln_vector_total_writer->addDatainRow(header_csv.begin(), header_csv.end());
  ln_vector_avg_writer->addDatainRow(header_csv.begin(), header_csv.end());
  ln_vector_frobenius_writer->addDatainRow(header_csv.begin(),
                                           header_csv.end());

  std::cout << "  Running calculation..." << '\n';

  for (unsigned int i = 2; i <= 16; i *= 2) {
    std::cout << "      i = " << i << '\n';
    Setup *lnvector = new Setup(1, i, 1);
    lnvector->random_uniform(2.35098870164e-38, 1.17549435082e-38);
    for (size_t j = 0; j < i_iter; j++) {
      if (j % 100 == 0) {
        std::cout << "*" << std::flush;
      }
      lnvector->GEMM();
      // lnvector->XSMM();
    }

    std::cout << '\n';

    std::vector<double> vector_total = measure_total(lnvector);
    std::vector<double> vector_average = average_error(vector_total, i * i);
    std::vector<double> vector_frobenius = measure_frobenius(lnvector);

    ln_vector_avg_writer->addDatainRow(vector_average.begin(),
                                       vector_average.end());
    ln_vector_total_writer->addDatainRow(vector_total.begin(),
                                         vector_total.end());
    ln_vector_frobenius_writer->addDatainRow(vector_frobenius.begin(),
                                             vector_frobenius.end());

    delete[] lnvector;
  }
  delete[] ln_vector_avg_writer;
  delete[] ln_vector_total_writer;
  delete[] ln_vector_frobenius_writer;

  // Compute the errorbounds for vectors
  std::cout << "Set up vector measurement all fp range" << '\n';

  CSVWriter *range_vector_avg_writer =
      new CSVWriter("range_vector_average.csv");
  CSVWriter *range_vector_total_writer =
      new CSVWriter("range_vector_total.csv");
  CSVWriter *range_vector_frobenius_writer =
      new CSVWriter("range_vector_frobenius.csv");

  range_vector_total_writer->addDatainRow(header_csv.begin(), header_csv.end());
  range_vector_avg_writer->addDatainRow(header_csv.begin(), header_csv.end());
  range_vector_frobenius_writer->addDatainRow(header_csv.begin(),
                                              header_csv.end());

  std::cout << "  Running calculation..." << '\n';

  for (unsigned int i = 2; i <= 16; i *= 2) {
    std::cout << "      i = " << i << '\n';
    Setup *rgvector = new Setup(1, i, 1);
    rgvector->random_uniform(0, std::numeric_limits<float>::max());
    for (size_t j = 0; j < i_iter; j++) {
      if (j % 100 == 0) {
        std::cout << "*" << std::flush;
      }
      rgvector->GEMM();
      // rgvector->XSMM();
    }

    std::cout << '\n';

    std::vector<double> vector_total = measure_total(rgvector);
    std::vector<double> vector_average = average_error(vector_total, i * i);
    std::vector<double> vector_frobenius = measure_frobenius(rgvector);

    range_vector_avg_writer->addDatainRow(vector_average.begin(),
                                          vector_average.end());
    range_vector_total_writer->addDatainRow(vector_total.begin(),
                                            vector_total.end());
    range_vector_frobenius_writer->addDatainRow(vector_frobenius.begin(),
                                                vector_frobenius.end());

    delete[] vector;
  }
  delete[] range_vector_avg_writer;
  delete[] range_vector_total_writer;
  delete[] range_vector_frobenius_writer;

  return 0;
}
