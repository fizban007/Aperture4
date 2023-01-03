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

#ifndef _PTC_UPDATER_SIMD_H_
#define _PTC_UPDATER_SIMD_H_

#include "systems/ptc_updater.h"

namespace Aperture {

template <typename Conf, template <class> class CoordPolicy,
          template <class> class PhysicsPolicy = ptc_physics_policy_empty>
class ptc_updater_simd : public ptc_updater<Conf, exec_policy_openmp,
                                                CoordPolicy, PhysicsPolicy> {
 public:
  // typedef ptc_updater<Conf, exec_policy_openmp, CoordPolicy, PhysicsPolicy>
  //     base_class;
  typedef typename Conf::value_t value_t;
  using base_class = ptc_updater<Conf, exec_policy_openmp, CoordPolicy, PhysicsPolicy>;
  static std::string name() { return "ptc_updater"; }

  using base_class::ptc_updater;

  void update_particles(value_t dt, uint32_t step);
};

}  // namespace Aperture

#endif  // _PTC_UPDATER_SIMD_H_
