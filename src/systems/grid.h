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

#ifndef __GRID_H_
#define __GRID_H_

#include "core/domain_info.h"
#include "core/grid.hpp"
#include "framework/system.h"

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
class domain_comm;

// The system that is responsible for setting up the computational grid
template <typename Conf>
class grid_t : public system_t, public Grid<Conf::dim, typename Conf::value_t> {
 public:
  using value_t = typename Conf::value_t;
  static std::string name() { return "grid"; }

  typedef Grid<Conf::dim, value_t> base_type;

  grid_t();
  template <template <class> class ExecPolicy>
  grid_t(const domain_comm<Conf, ExecPolicy>& comm) :
      grid_t(comm.domain_info()) {
    comm.resize_buffers(*this);
  }
  grid_t(const domain_info_t<Conf::dim>& domain_info);
  grid_t(const grid_t<Conf>& grid) = default;
  virtual ~grid_t();

  grid_t<Conf>& operator=(const grid_t<Conf>& grid) = default;

  // Coordinate for output position
  virtual vec_t<float, Conf::dim> cart_coord(
      const index_t<Conf::dim>& pos) const {
    vec_t<float, Conf::dim> result;
    for (int i = 0; i < Conf::dim; i++) result[i] = this->coord(i, pos[i], false);
    return result;
  }

  // Convert local coordinate position to global one
  vec_t<value_t, 3> x_global(const vec_t<value_t, 3>& rel_x,
                             uint32_t cell) {
    index_t<Conf::dim> pos = get_pos(Conf::idx(cell, m_ext), m_ext);
    return this->coord_global(pos, rel_x);
  }

  // Convert global coordinate position to local one
  void from_x_global(const vec_t<value_t, 3>& global_x,
                     vec_t<value_t, 3>& rel_x,
                     uint32_t& cell) {
    index_t<Conf::dim> pos;
    this->from_global(global_x, pos, rel_x);
    cell = Conf::idx(pos, m_ext).linear;
  }

  inline typename Conf::idx_t get_idx(const index_t<Conf::dim>& pos) const {
    return typename Conf::idx_t(pos, m_ext);
  }

  template <typename... Args>
  inline typename Conf::idx_t get_idx(Args... args) const {
    return typename Conf::idx_t(index_t<Conf::dim>(args...), m_ext);
  }

  inline typename Conf::idx_t idx_at(uint32_t lin) const {
    return typename Conf::idx_t(lin, m_ext);
  }

  inline typename Conf::idx_t begin() const { return idx_at(0); }

  inline typename Conf::idx_t end() const {
    return idx_at(m_ext.size());
  }

  inline size_t size() const { return m_ext.size(); }

 protected:
  extent_t<Conf::dim> m_ext;
};

}  // namespace Aperture

#endif  // __GRID_H_
