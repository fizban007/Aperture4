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

#include "grid_sph.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/domain_comm.h"
#include "utils/range.hpp"

namespace Aperture {

// template <typename Conf>
// grid_sph_t<Conf>::~grid_sph_t() {}

template class grid_sph_t<Config<2>>;
template class grid_sph_t<Config<3>>;

}  // namespace Aperture
