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

#include "core/math.hpp"
#include "core/random.h"
#include "data/curand_states.h"
#include "data/fields.h"
#include "data/particle_data.h"
#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/policies/exec_policy_cuda.hpp"
#include "systems/ptc_injector_cuda.hpp"
#include "utils/kernel_helper.hpp"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace {

using namespace Aperture;

template <typename value_t>
struct fast_wave_solution {
  value_t sinth = 0.0;
  value_t costh;
  value_t lambda = 1.0;
  value_t omega = 1.0 / lambda;
  value_t a0 = 30.0;
  value_t B0 = 10.0 * omega;

  value_t x0 = 0.0;
  value_t length = 4.0f;
  value_t smooth_width = 0.1f;

  HD_INLINE value_t xi(value_t x, value_t y) const {
    return x * sinth + y * costh;
  }

  HD_INLINE value_t eta(value_t x, value_t y) const {
    return x * costh - y * sinth;
  }

  HOST_DEVICE fast_wave_solution(value_t sinth_, value_t lambda_,
                                 value_t x0_, value_t length_,
                                 value_t a0_)
      : sinth(sinth_),
        lambda(lambda_),
        x0(x0_),
        length(length_),
        a0(a0_) {
    costh = math::sqrt(1.0f - sinth * sinth);
    omega = 1.0 / lambda;
    // delta_eta = delta_y * sinth;
    // eta0 = eta(0.0, y0);
  }

  HD_INLINE value_t wave_arg(value_t t, value_t x, value_t y) const {
    return 2.0 * M_PI * (x - t) /
           lambda;
  }

  HD_INLINE value_t wave_arg_clamped(value_t t, value_t x, value_t y) const {
    return 2.0 * M_PI * clamp<value_t>((x - t) / lambda, 0.0f, length);
  }

  HD_INLINE value_t wave_profile(value_t x) const {
    // Convert x into a number between 0 and 1
    value_t arg = clamp<value_t>(x / (2.0 * M_PI) / length, 0.0f, 1.0f);
    value_t prof = 0.0f;
    if (arg < smooth_width) {
      prof = square(math::sin(arg * M_PI / (smooth_width * 2.0f)));
    } else if (arg > (1.0f - smooth_width)) {
      prof = square(
          math::sin((arg - 1.0f + smooth_width) * M_PI / (smooth_width * 2.0f) +
                    0.5f * M_PI));
    } else {
      prof = 1.0f;
    }
    return math::sin(x) * prof;
    // return math::sin(x) * square(math::sin(0.5 * x / length));
  }

  HD_INLINE value_t Bz(value_t t, value_t x, value_t y) const {
    return a0 * omega * wave_profile(wave_arg_clamped(t, x, y));
  }

  HD_INLINE value_t Ey(value_t t, value_t x, value_t y) const {
    return Bz(t, x, y);
  }

};

}  // namespace

namespace Aperture {

template <typename Conf>
void
initial_condition_wave(vector_field<Conf> &B,
                       vector_field<Conf> &E, vector_field<Conf> &B0,
                       particle_data_t &ptc, rng_states_t &states, int mult,
                       Scalar weight) {
  using value_t = typename Conf::value_t;
  value_t weight_enhance_factor = 1.0f;
  value_t sinth = sim_env().params().get_as<double>("theta_bg", 0.0);
  value_t rho_bg = sim_env().params().get_as<double>("rho_bg", 10000.0);
  value_t a0 = sim_env().params().get_as<double>("a0", 5000.0);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);
  q_e *= weight_enhance_factor;
  int mult_wave = 1;

  fast_wave_solution<value_t> wave(sinth, 1.0, 0.05, 4.0, a0);

  B0.set_values(
      2, [wave, sinth](value_t x, value_t y, value_t z) { return wave.B0 * sinth; });
  B0.set_values(
      0, [wave, sinth](value_t x, value_t y, value_t z) { return wave.B0 * wave.costh; });
  B.set_values(
      2, [wave](value_t x, value_t y, value_t z) { return wave.Bz(0.0, x, y); });
  E.set_values(
      1, [wave](value_t x, value_t y, value_t z) { return wave.Ey(0.0, x, y); });

  auto num = ptc.number();

  auto& grid = B.grid();
  auto injector = sim_env().register_system<ptc_injector<Conf, exec_policy_cuda>>(grid);
  injector->init();

  injector->inject(
      [] __device__(auto &pos, auto &grid, auto &ext) { return true; },
      [mult] __device__(auto &pos, auto &grid, auto &ext) {
        return 2 * mult;
      },
      [] __device__(auto &pos, auto &grid, auto &ext, rng_t &rng,
                    PtcType type) {
        return vec_t<value_t, 3>(0.0, 0.0, 0.0);
      },
      [mult, rho_bg, q_e] __device__(auto &x_global) {
        return rho_bg / q_e / mult;
      });

  Logger::print_info("After initial condition, there are {} particles", ptc.number());
}

template void initial_condition_wave<Config<2>>(
    vector_field<Config<2>> &B,
    vector_field<Config<2>> &E, vector_field<Config<2>> &B0,
    particle_data_t &ptc, rng_states_t &states, int mult, Scalar weight);

}  // namespace Aperture
