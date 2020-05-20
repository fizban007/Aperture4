#ifndef __MULTI_ARRAY_DATA_HPP_
#define __MULTI_ARRAY_DATA_HPP_

#include "framework/data.h"
#include "core/multi_array.hpp"

namespace Aperture {

////////////////////////////////////////////////////////////////////////////////
///  Thin wrapper around a multi_array for the purpose of unified data
///  management.
////////////////////////////////////////////////////////////////////////////////
template <typename T, int Rank>
class multi_array_data : public data_t, public multi_array<T, Rank> {
 public:
  multi_array_data(MemType model = default_mem_type) :
      multi_array<T, Rank>(model) {}
  multi_array_data(const extent_t<Rank>& ext, MemType model = default_mem_type) :
      multi_array<T, Rank>(ext, model) {}

  void init() override {
    this->assign(0.0);
  }
};

}

#endif // __MULTI_ARRAY_DATA_HPP_
