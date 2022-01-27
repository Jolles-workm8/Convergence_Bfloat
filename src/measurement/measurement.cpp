#include "measurement.hpp"

std::vector<double> measure_total(Setup *classname) {
  std::vector<double> c_total = {
      total_error(classname->l_c_ref_fp64, classname->l_c_ref_fp32),
      total_error(classname->l_c_ref_fp64, classname->l_c_bf_Z0),
      total_error(classname->l_c_ref_fp64, classname->l_c_bf_Z1),
      total_error(classname->l_c_ref_fp64, classname->l_c_bf_Z2),
      total_error(classname->l_c_ref_fp64, classname->l_c_bf_Z3),
      total_error(classname->l_c_ref_fp64, classname->l_c_bf_Z4)};
  /*
    std::vector<double> c_avg =
        average_error(c_total, classname->l_n * classname->l_m);
  */

  return c_total;
}

std::vector<double> measure_frobenius(Setup *classname) {
  std::vector<double> c_frobenius = {
      total_error(classname->l_c_ref_fp64, classname->l_c_ref_fp32),
      total_error(classname->l_c_ref_fp64, classname->l_c_bf_Z0),
      total_error(classname->l_c_ref_fp64, classname->l_c_bf_Z1),
      total_error(classname->l_c_ref_fp64, classname->l_c_bf_Z2),
      total_error(classname->l_c_ref_fp64, classname->l_c_bf_Z3),
      total_error(classname->l_c_ref_fp64, classname->l_c_bf_Z4)};

  return c_frobenius;
}
