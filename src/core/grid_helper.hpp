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

#pragma once

#include "core/cuda_control.h"
#include "core/grid.hpp"
#include "utils/index.hpp"
#include "utils/vec.hpp"

namespace Aperture {

template <typename Conf>
HD_INLINE typename Conf::idx_t
grid_get_idx(typename Conf::grid_t& grid, uint32_t cell) {
  return Conf::idx(cell, grid.extent());
}

template <typename Conf>
HD_INLINE index_t<Conf::dim>
grid_get_pos(typename Conf::grid_t& grid, uint32_t cell) {
  auto idx = Conf::idx(cell, grid.extent());
  return idx.get_pos();
}

}  // namespace Aperture
