#include "measurement.hpp"

std::vector<double> measure_total(Setup *classname) {
  std::vector<double> c_total = {
      total_error(classname->l_c_ref_fp64, classname->l_c_ref_fp32),
      total_error(classname->l_c_ref_fp64, classname->l_c_bf_3),
      total_error(classname->l_c_ref_fp64, classname->l_c_bf_6),
      total_error(classname->l_c_ref_fp64, classname->l_c_bf_9),
  };
  /*
    std::vector<double> c_avg =
        average_error(c_total, classname->l_n * classname->l_m);
  */

  return c_total;
}

std::vector<double> measure_frobenius(Setup *classname) {
  std::vector<double> c_frobenius = {
      frobenius_error(classname->l_c_ref_fp64, classname->l_c_ref_fp32),
      frobenius_error(classname->l_c_ref_fp64, classname->l_c_bf_3),
      frobenius_error(classname->l_c_ref_fp64, classname->l_c_bf_6),
      frobenius_error(classname->l_c_ref_fp64, classname->l_c_bf_9),
  };

  return c_frobenius;
}
