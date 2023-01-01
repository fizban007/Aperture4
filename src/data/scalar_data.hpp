/*
 * Copyright (c) 2021 Alex Chen.
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

#pragma once

#include "core/buffer.hpp"
#include "framework/data.h"

namespace Aperture {

////////////////////////////////////////////////////////////////////////////////
///  Thin wrapper of a single scalar variable that is to be written to the
///  output.
////////////////////////////////////////////////////////////////////////////////
template <typename T>
class scalar_data : public data_t {
 public:
  scalar_data(MemType type = default_mem_type) : m_data(1, type) {}

  void init() override { m_data.assign(T(0.0)); }

  void set_gather(bool b) { m_do_gather = b; }

  bool do_gather() const { return m_do_gather; }

  void copy_to_host() { m_data.copy_to_host(); }

  buffer<T>& data() { return m_data; }

  const buffer<T>& data() const { return m_data; }

 private:
  buffer<T> m_data;
  bool m_do_gather = true;
};

}  // namespace Aperture
