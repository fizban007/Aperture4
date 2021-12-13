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
///  management, so that the data_exporter can automatically output every entry
///  of this type. This only works on one rank!!
////////////////////////////////////////////////////////////////////////////////
template <typename T, int Rank>
class multi_array_data : public data_t, public multi_array<T, Rank> {
 public:
  using multi_array<T, Rank>::multi_array;

  void init() override {
    this->assign(0.0);
  }
};

}

#endif // __MULTI_ARRAY_DATA_HPP_
