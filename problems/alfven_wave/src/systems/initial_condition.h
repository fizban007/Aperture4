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

#ifndef _INITIAL_CONDITION_H_
#define _INITIAL_CONDITION_H_

#include "framework/environment.h"
#include "systems/grid_sph.hpp"

namespace Aperture {

template <typename Conf>
void set_initial_condition(const grid_sph_t<Conf>& grid);

}

#endif  // _INITIAL_CONDITION_H_
