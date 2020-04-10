#ifndef __PARAM_STORE_H_
#define __PARAM_STORE_H_

#include "core/params.h"
#include "utils/logger.h"
#include <string>

namespace Aperture {

class param_store {
 private:
  // Because we need to use std::variant which is not supported in CUDA code, we
  // have to use p_impl idiom to hide the implementation detail
  class param_store_impl;
  param_store_impl* p_impl;

 public:
  param_store();
  ~param_store();
 
  void parse(const std::string& filename);
  const sim_params& params() const;

  template <typename T>
  T get(const std::string& name, T default_value) const;

  template <typename T>
  T get(const std::string& name) const;

  template <typename T>
  void add(const std::string& name, const T& value);

};

}

#endif // __PARAM_STORE_H_
