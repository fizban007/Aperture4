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

#include "core/grid.hpp"
#include <cstdint>

namespace Aperture {

struct params_struct;

void init_morton(const uint32_t m2dLUT[256], const uint32_t m3dLUT[256]);
// void init_dev_params(conparams_structams& params);
template <int Dim, typename value_t>
void init_dev_grid(const Grid<Dim, value_t>& grid);

void init_dev_charge_mass(const float charge[max_ptc_types],
                          const float mass[max_ptc_types]);

void init_dev_rank(int rank);

}  // namespace Aperture
