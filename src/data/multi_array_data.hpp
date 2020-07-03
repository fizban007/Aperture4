/*
 * Copyright (c) 2020 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

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
