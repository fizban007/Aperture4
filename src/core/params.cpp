#include "params.h"
#include "cpptoml.h"
#include "utils/logger.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace Aperture {

///  Function to convert a string to a bool variable.
///
///  @param str   The string to be converted.
///  \return      The bool corresponding to str.
bool
to_bool(std::string& str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  std::istringstream ist(str);
  bool b;
  ist >> std::boolalpha >> b;
  return b;
}

params_struct
parse_config(const std::string& filename) {
  params_struct result;
  parse_config(filename, result);
  return result;
}

void
parse_config(const std::string& filename, params_struct& params) {
  auto config = cpptoml::parse_file(filename);
  params_struct defaults;

  params.dt = config->get_as<double>("dt").value_or(defaults.dt);
  params.max_steps = config->get_as<uint64_t>("max_steps")
                         .value_or(defaults.max_steps);
  params.data_interval = config->get_as<int>("data_interval")
                             .value_or(defaults.data_interval);
}


}
