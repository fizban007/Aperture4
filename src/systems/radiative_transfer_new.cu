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

#include "framework/config.h"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/coord_policy_spherical.hpp"
#include "systems/policies/exec_policy_cuda.hpp"
#include "systems/radiation/default_radiation_scheme.hpp"
#include "systems/radiative_transfer_new_impl_cu.hpp"

namespace Aperture {

template class radiative_transfer<Config<1>, exec_policy_cuda,
                                  coord_policy_cartesian,
                                  default_radiation_scheme>;
template class radiative_transfer<Config<2>, exec_policy_cuda,
                                  coord_policy_cartesian,
                                  default_radiation_scheme>;
template class radiative_transfer<Config<3>, exec_policy_cuda,
                                  coord_policy_cartesian,
                                  default_radiation_scheme>;

template class radiative_transfer<Config<2>, exec_policy_cuda,
                                  coord_policy_spherical,
                                  default_radiation_scheme>;

}  // namespace Aperture
