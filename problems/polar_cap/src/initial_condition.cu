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

#include "core/math.hpp"
#include "core/random.h"
// #include "data/curand_states.h"
#include "data/fields.h"
#include "data/particle_data.h"
#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/policies/exec_policy_gpu.hpp"
#include "systems/ptc_injector_new.h"
#include "utils/kernel_helper.hpp"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace Aperture {

template <typename Conf>
void
polar_cap_initial_condition(vector_field<Conf> &B0, particle_data_t &ptc,
                            rng_states_t<exec_tags::device> &states) {
  using value_t = typename Conf::value_t;

  value_t Bp = sim_env().params().get_as<double>("Bp", 1.0e3);
  value_t Rpc = sim_env().params().get_as<double>("Rpc", 1.0);
  value_t R_star = sim_env().params().get_as<double>("R_star", 10.0);

  B0.set_values(0, [Bp, R_star](auto x, auto y, auto z) {
    z = z / R_star + 1.0;
    x /= R_star;
    y /= R_star;
    value_t r = math::sqrt(x * x + y * y + z * z);
    return 3.0f * Bp * x * z / (r * r * r * r * r);
  });
  B0.set_values(1, [Bp, R_star](auto x, auto y, auto z) {
    z = z / R_star + 1.0;
    x /= R_star;
    y /= R_star;
    value_t r = math::sqrt(x * x + y * y + z * z);
    return 3.0f * Bp * y * z / (r * r * r * r * r);
  });
  B0.set_values(2, [Bp, R_star](auto x, auto y, auto z) {
    z = z / R_star + 1.0;
    x /= R_star;
    y /= R_star;
    value_t r = math::sqrt(x * x + y * y + z * z);
    return 3.0f * Bp * z * z / (r * r * r * r * r) - Bp / (r * r * r);
  });

  // Initialize a background plasma?
}

template void polar_cap_initial_condition<Config<3>>(
    vector_field<Config<3>> &B0, particle_data_t &ptc, rng_states_t<exec_tags::device> &states);

}  // namespace Aperture
