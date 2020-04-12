#ifndef __PARAMS_STORE_H_
#define __PARAMS_STORE_H_

#include "core/params.h"
#include "utils/logger.h"
#include <string>
#include "visit_struct/visit_struct.hpp"

namespace Aperture {

class params_store {
 private:
  // Because we need to use std::variant which is not supported in CUDA code, we
  // have to use p_impl idiom to hide the implementation detail
  class params_store_impl;
  params_store_impl* p_impl;

 public:
  params_store();
  ~params_store();
 
  void parse(const std::string& filename);
  // const params_struct& params() const;

  template <typename T>
  T get(const std::string& name, T default_value) const;

  template <typename T>
  T get(const std::string& name) const;

  template <typename T>
  void add(const std::string& name, const T& value);

};

}

#endif // __PARAMS_STORE_H_
