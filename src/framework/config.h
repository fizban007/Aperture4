#ifndef __CONFIG_H_
#define __CONFIG_H_

#include "core/enum_types.h"
#include "core/typedefs_and_constants.h"
#include "utils/index.hpp"
#include "utils/logger.h"

namespace Aperture {

class sim_environment;

template <int Dim,
          typename FloatT = Scalar,
          template <int> typename Index_t = idx_col_major_t,
          MemoryModel MemModel = default_memory_model>
class Config {
 public:
  static constexpr int dim = Dim;
  static constexpr MemoryModel default_mem_model = MemModel;
  typedef FloatT value_type;
  typedef Index_t<Dim> index_type;

  Config(const sim_environment& env) { Logger::print_info("default config constructor"); }
  // Config(const Config& other) { Logger::print_info("config copy constructor"); }
  Config(const Config& other) = delete;

  Config(Config&& other) { Logger::print_info("config move constructor"); }

  Config& operator=(const Config& other) = delete;

  Config& operator=(Config&& other) {
    Logger::print_info("config move assignment");
    return *this;
  }
};

}

#endif
