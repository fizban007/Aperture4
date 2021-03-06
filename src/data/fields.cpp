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

#include "fields.h"
#include "core/detail/multi_array_helpers.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/grid.h"
#include <exception>

namespace Aperture {

template <int N, typename Conf>
field_t<N, Conf>::field_t(const Grid_t& grid, MemType memtype)
    : m_memtype(memtype) {
  set_memtype(memtype);
  resize(grid);
}

template <int N, typename Conf>
field_t<N, Conf>::field_t(const Grid_t& grid,
                          const vec_t<stagger_t, N> st, MemType memtype)
    : m_stagger(st), m_memtype(memtype) {
  set_memtype(memtype);
  resize(grid);
}

template <int N, typename Conf>
field_t<N, Conf>::field_t(const Grid_t& grid, field_type type,
                          MemType memtype)
    : m_memtype(memtype) {
  set_memtype(memtype);
  if (type == field_type::face_centered) {
    m_stagger[0] = stagger_t(0b001);
    m_stagger[1] = stagger_t(0b010);
    m_stagger[2] = stagger_t(0b100);
  } else if (type == field_type::edge_centered) {
    m_stagger[0] = stagger_t(0b110);
    m_stagger[1] = stagger_t(0b101);
    m_stagger[2] = stagger_t(0b011);
  } else if (type == field_type::cell_centered) {
    m_stagger[0] = m_stagger[1] = m_stagger[2] = stagger_t(0b000);
  } else if (type == field_type::vert_centered) {
    m_stagger[0] = m_stagger[1] = m_stagger[2] = stagger_t(0b111);
  }
  resize(grid);
}

template <int N, typename Conf>
void
field_t<N, Conf>::init() {
  // Logger::print_debug("field init, memtype {}", (int)m_memtype);
  for (int i = 0; i < N; i++) {
    m_data[i].assign(0.0);
  }
}

template <int N, typename Conf>
void
field_t<N, Conf>::resize(const Grid_t& grid) {
  m_grid = &grid;
  for (int i = 0; i < N; i++) {
    m_data[i].resize(m_grid->extent());
  }
}

template <int N, typename Conf>
void
field_t<N, Conf>::set_memtype(MemType type) {
  m_memtype = type;
  for (int i = 0; i < N; i++) {
    m_data[i].set_memtype(type);
  }
}

template <int N, typename Conf>
void
field_t<N, Conf>::add_by(const field_t<N, Conf> &other, typename Conf::value_t scale) {
  for (int i = 0; i < N; i++) {
    if (m_memtype == MemType::host_only)
      add(m_data[i], other.m_data[i], index_t<Conf::dim>{}, index_t<Conf::dim>{},
          m_grid->extent());
    else
      add_dev(m_data[i], other.m_data[i], index_t<Conf::dim>{}, index_t<Conf::dim>{},
              m_grid->extent());
  }
}

///////////////////////////////////////////////////////
// Explicitly instantiate some fields
///////////////////////////////////////////////////////
template class field_t<3, Config<1, float>>;
template class field_t<3, Config<2, float>>;
template class field_t<3, Config<3, float>>;
template class field_t<3, Config<1, double>>;
template class field_t<3, Config<2, double>>;
template class field_t<3, Config<3, double>>;

template class field_t<1, Config<1, float>>;
template class field_t<1, Config<2, float>>;
template class field_t<1, Config<3, float>>;
template class field_t<1, Config<1, double>>;
template class field_t<1, Config<2, double>>;
template class field_t<1, Config<3, double>>;

}  // namespace Aperture
