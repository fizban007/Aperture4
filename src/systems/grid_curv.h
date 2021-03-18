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

#ifndef _GRID_CURV_H_
#define _GRID_CURV_H_

#include "core/multi_array.hpp"
#include "grid.h"

namespace Aperture {

template <typename ValueT, int Rank, typename Idx_t>
struct grid_ptrs {
  vec_t<ndptr_const<ValueT, Rank, Idx_t>, 3> le;
  vec_t<ndptr_const<ValueT, Rank, Idx_t>, 3> lb;
  vec_t<ndptr_const<ValueT, Rank, Idx_t>, 3> Ae;
  vec_t<ndptr_const<ValueT, Rank, Idx_t>, 3> Ab;
  ndptr_const<ValueT, Rank, Idx_t> dV;
};

////////////////////////////////////////////////////////////////////////////////
///  Curvilinear grid, keeps track of volume, area and length elements of every
///  cell. Spherical and other coordinate systems should derive from this class.
////////////////////////////////////////////////////////////////////////////////
template <typename Conf>
class grid_curv_t : public grid_t<Conf> {
 public:
  static std::string name() { return "grid"; }
  typedef typename Conf::value_t value_t;
  typedef grid_ptrs<value_t, Conf::dim, typename Conf::idx_t> grid_ptrs_t;

  grid_curv_t() : grid_t<Conf>() {
    resize_arrays();
  }
  grid_curv_t(const domain_comm<Conf>& comm) : grid_t<Conf>(comm) {
    resize_arrays();
  }
  grid_curv_t(const grid_curv_t<Conf>& grid) = default;
  virtual ~grid_curv_t() {}

  void resize_arrays() {
    for (int i = 0; i < 3; i++) {
      m_le[i].resize(this->extent());
      m_lb[i].resize(this->extent());
      m_Ae[i].resize(this->extent());
      m_Ab[i].resize(this->extent());
    }
    m_dV.resize(this->extent());
  }

  void init() {
    compute_coef();
  }

  virtual void compute_coef() = 0;
  grid_ptrs_t get_grid_ptrs() const {
    grid_ptrs_t result;

    for (int i = 0; i < 3; i++) {
      result.le[i] = m_le[i].dev_ndptr_const();
      result.lb[i] = m_lb[i].dev_ndptr_const();
      result.Ae[i] = m_Ae[i].dev_ndptr_const();
      result.Ab[i] = m_Ab[i].dev_ndptr_const();
    }
    result.dV = m_dV.dev_ndptr_const();

    return result;
  }

  std::array<multi_array<value_t, Conf::dim>, 3> m_le;
  std::array<multi_array<value_t, Conf::dim>, 3> m_lb;
  std::array<multi_array<value_t, Conf::dim>, 3> m_Ae;
  std::array<multi_array<value_t, Conf::dim>, 3> m_Ab;
  multi_array<value_t, Conf::dim> m_dV;
};

}  // namespace Aperture

#endif  // _GRID_CURV_H_
