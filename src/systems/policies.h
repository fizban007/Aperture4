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

#ifndef __POLICIES_H_
#define __POLICIES_H_

namespace Aperture {

template <typename Conf>
class exec_policy_cuda;

template <typename Conf>
class exec_policy_host;

template <typename Conf>
class exec_policy_openmp;

template <typename Conf>
class exec_policy_openmp_simd;

template <typename Conf>
class coord_policy_cartesian;

template <typename Conf>
class coord_policy_spherical;

template <typename Conf>
class coord_policy_gr_ks_sph;

template <typename T>
class ptc_physics_policy_empty {};

}

#endif // __POLICIES_H_
