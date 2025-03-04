/*
 * Copyright (c) 2022 Alex Chen.
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
#include "core/particle_structs.h"
#include "core/typedefs_and_constants.h"
#include "data/particle_data.h"
#include "framework/data.h"

namespace Aperture {

template <typename BufferType>
class tracked_ptc : public data_t {
 public:
  tracked_ptc(size_t max_size, MemType type = default_mem_type);
  ~tracked_ptc();

  size_t size() { return m_size; }
  size_t number() { return m_number; }
  void set_number(size_t n) { m_number = n; }

  buffer<output_type> x1, x2, x3;
  buffer<output_type> p1, p2, p3, E;
#ifdef PARA_PERP
  buffer<output_type> work_para, work_perp;
#endif
  buffer<output_type> weight;
  buffer<uint32_t> flag;
  buffer<uint64_t> id;

 private:
  size_t m_size;
  size_t m_number;
};

using tracked_particles_t = tracked_ptc<ptc_buffer>;
using tracked_photons_t = tracked_ptc<ph_buffer>;

}  // namespace Aperture
