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

#include "framework/config.h"
#include "systems/policies/ptc_physics_policy_empty.hpp"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/coord_policy_cartesian_gca.hpp"
#include "systems/policies/coord_policy_cartesian_impl_cooling.hpp"
#include "systems/policies/coord_policy_spherical.hpp"
#include "systems/policies/coord_policy_gr_ks_sph.hpp"
#include "systems/policies/exec_policy_cuda.hpp"
#include "systems/ptc_updater_base_impl.hpp"

namespace Aperture {

template class ptc_updater_new<Config<1>, exec_policy_cuda,
                               coord_policy_cartesian>;
template class ptc_updater_new<Config<2>, exec_policy_cuda,
                               coord_policy_cartesian>;
template class ptc_updater_new<Config<3>, exec_policy_cuda,
                               coord_policy_cartesian>;

template class ptc_updater_new<Config<2>, exec_policy_cuda,
                               coord_policy_cartesian_gca>;
template class ptc_updater_new<Config<2>, exec_policy_cuda,
                               coord_policy_cartesian_impl_cooling>;

template class ptc_updater_new<Config<2>, exec_policy_cuda,
                               coord_policy_spherical>;
template class ptc_updater_new<Config<3>, exec_policy_cuda,
                               coord_policy_spherical>;

template class ptc_updater_new<Config<2>, exec_policy_cuda,
                               coord_policy_gr_ks_sph>;

}  // namespace Aperture
