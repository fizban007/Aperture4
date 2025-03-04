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
#include "systems/physics/lorentz_transform.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "utils/kernel_helper.hpp"

namespace Aperture {

namespace {

HOST_DEVICE Scalar double_current_sheet_Bx(Scalar B0, Scalar y, Scalar ysize, Scalar delta) {
  if (y < 0.0f) {
    return B0 * tanh((y + 0.25f * ysize) / delta);
  } else {
    return -B0 * tanh((y - 0.25f * ysize) / delta);
  }
}

}

template <typename Conf>
void
harris_current_sheet(vector_field<Conf> &B, particle_data_t &ptc,
                     rng_states_t<exec_tags::dynamic> &states) {
  using value_t = typename Conf::value_t;
  // auto delta = sim_env().params().get_as<double>("current_sheet_delta", 5.0);
  value_t B_g = sim_env().params().get_as<double>("guide_field", 0.0);
  value_t sigma = sim_env().params().get_as<double>("sigma", 1.0e3);
  value_t kT_cs = sim_env().params().get_as<double>("current_sheet_kT", 1.0);
  value_t kT_upstream =
      sim_env().params().get_as<double>("upstream_kT", 1.0e-2);
  value_t beta_d =
      sim_env().params().get_as<double>("current_sheet_drift", 0.5);
  value_t gamma_d = 1.0f / math::sqrt(1.0f - beta_d * beta_d);

  value_t delta = 2.0f * kT_cs / (math::sqrt(sigma) * gamma_d * beta_d);
  value_t n_d = gamma_d * sigma / (4.0f * kT_cs);

  int n_cs = sim_env().params().get_as<int64_t>("current_sheet_n", 15);
  int n_upstream = sim_env().params().get_as<int64_t>("upstream_n", 5);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);

  // Our unit for length will be upstream c/\omega_p, therefore sigma determines
  // the upstream field strength
  value_t B0 = math::sqrt(sigma);
  auto &grid = B.grid();
  auto ext = grid.extent();
  value_t ysize = grid.sizes[1];

  // Initialize the magnetic field values
  B.set_values(0, [B0, delta, ysize](auto x, auto y, auto z) {
    return B0 * tanh(y / delta);
  });
  B.set_values(2, [B0, B_g](auto x, auto y, auto z) { return B0 * B_g; });

  // auto injector =
  //     sim_env().register_system<ptc_injector<Conf, exec_policy_gpu>>(grid);
  // ptc_injector<Conf, exec_policy_gpu> injector(grid);
  ptc_injector_dynamic<Conf> injector(grid);

  // Background (upstream) particles
  injector.inject_pairs(
      // Injection criterion
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      // Number injected
      [n_upstream] LAMBDA(auto &pos, auto &grid, auto &ext) {
        return 2 * n_upstream;
      },
      // Initialize particles
      [kT_upstream, B_g, delta] LAMBDA(auto &x_global,
                                           rand_state &state, PtcType type) {
        value_t y = x_global[1];
        value_t Bx = tanh(y / delta);
        value_t B = math::sqrt(Bx * Bx + B_g * B_g);
        vec_t<value_t, 3> u_d = rng_maxwell_juttner_3d(state, kT_upstream);

        // auto p1 = u_d[0];
        // auto p2 = u_d[1];
        // auto p3 = u_d[2];
        value_t pdotB = (u_d[0] * Bx + u_d[2] * B_g) / B;
        return vec_t<value_t, 3>(pdotB * Bx / B, 0.0, pdotB * B_g / B);
      },
      // Particle weight
      [n_upstream] LAMBDA(auto &x_global, PtcType type) { return 1.0 / n_upstream; });

  // Current sheet particles
  injector.inject_pairs(
      // Injection criterion
      [delta] LAMBDA(auto &pos, auto &grid, auto &ext) {
        value_t y = grid.template coord<1>(pos, 0.5f);
        value_t cs_y = 3.0f * delta;
        if (math::abs(y) < cs_y) {
          return true;
        } else {
          return false;
        }
      },
      // Number injected
      [n_cs] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2 * n_cs; },
      // Initialize particles
      [kT_cs, beta_d] LAMBDA(auto &x_global, rand_state &state,
                             PtcType type) {
        vec_t<value_t, 3> u_d = rng_maxwell_juttner_drifting(state, kT_cs, beta_d);
        value_t sign = 1.0f;
        if (type == PtcType::positron) sign *= -1.0f;

        auto p1 = u_d[1] * sign;
        auto p2 = u_d[2] * sign;
        auto p3 = u_d[0] * sign;
        return vec_t<value_t, 3>(p1, p2, p3);
        // return vec_t<value_t, 3>(gamma_shift * (p1 + beta_shift * gamma_d),
        // p2, p3);
      },
      // Particle weight
      [B0, n_cs, q_e, beta_d, delta] LAMBDA(auto &x_global, PtcType type) {
        auto y = x_global[1];
        value_t j = -B0 / delta / square(cosh(y / delta));
        value_t w = math::abs(j) / q_e / n_cs / (2.0f * beta_d);
        return w;
      },
      flag_or(PtcFlag::exclude_from_spectrum, PtcFlag::ignore_tracking));

  Logger::print_info("After initial condition, there are {} particles",
                     ptc.number());
}

template <typename Conf>
void
boosted_harris_sheet(vector_field<Conf> &B, particle_data_t &ptc,
                     rng_states_t<exec_tags::dynamic> &states) {
  using value_t = typename Conf::value_t;
  // auto delta = sim_env().params().get_as<double>("current_sheet_delta", 5.0);
  value_t B_g = sim_env().params().get_as<double>("guide_field", 0.0);
  value_t sigma = sim_env().params().get_as<double>("sigma", 1.0e3);
  value_t kT_cs = sim_env().params().get_as<double>("current_sheet_kT", 1.0);
  value_t kT_upstream =
      sim_env().params().get_as<double>("upstream_kT", 1.0e-2);
  value_t beta_d =
      sim_env().params().get_as<double>("current_sheet_drift", 0.5);
  value_t gamma_d = 1.0f / math::sqrt(1.0f - beta_d * beta_d);
  value_t boost_beta = sim_env().params().get_as<double>("boost_beta", 0.0);

  value_t delta = 2.0f * kT_cs / (math::sqrt(sigma) * gamma_d * beta_d);
  value_t n_d = gamma_d * sigma / (4.0f * kT_cs);

  int n_cs = sim_env().params().get_as<int64_t>("current_sheet_n", 15);
  int n_upstream = sim_env().params().get_as<int64_t>("upstream_n", 5);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);

  // Our unit for length will be upstream c/\omega_p, therefore sigma determines
  // the upstream field strength
  value_t B0 = math::sqrt(sigma);
  auto &grid = B.grid();
  auto ext = grid.extent();
  value_t ysize = grid.sizes[1];

  // Initialize the magnetic field values
  B.set_values(0, [B0, delta, ysize](auto x, auto y, auto z) {
    return B0 * tanh(y / delta);
  });
  B.set_values(2, [B0, B_g](auto x, auto y, auto z) { return B0 * B_g; });

  // auto injector =
  //     sim_env().register_system<ptc_injector<Conf, exec_policy_gpu>>(grid);
  // ptc_injector<Conf, exec_policy_gpu> injector(grid);
  ptc_injector_dynamic<Conf> injector(grid);

  // Background (upstream) particles
  injector.inject_pairs(
      // Injection criterion
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      // Number injected
      [n_upstream] LAMBDA(auto &pos, auto &grid, auto &ext) {
        return 2 * n_upstream;
      },
      // Initialize particles
      [kT_upstream, boost_beta] LAMBDA(auto &x_global,
                                           rand_state &state, PtcType type) {
        auto p1 = rng_gaussian<value_t>(state, 2.0f * kT_upstream);
        auto p2 = rng_gaussian<value_t>(state, 2.0f * kT_upstream);
        auto p3 = rng_gaussian<value_t>(state, 2.0f * kT_upstream);
        // value_t gamma = math::sqrt(1.0f + p1*p1 + p2*p2 + p3*p3);
        // value_t beta = p1 / gamma;
        // return vec_t<value_t, 3>(beta / math::sqrt(1.0f - beta*beta), 0.0f,
        // 0.0f);
        auto p = vec_t<value_t, 3>(p1, p2, p3);
        value_t gamma = math::sqrt(1.0f + p.dot(p));
        // return lorentz_transform_momentum(p, {boost_beta, 0.0, 0.0});
        vec_t<value_t, 4> p_prime =
            lorentz_transform_vector(gamma, p, {boost_beta, 0.0, 0.0});
        return p_prime.template subset<1, 4>();
      },
      // Particle weight
      [n_upstream] LAMBDA(auto &x_global, PtcType type) { return 1.0 / n_upstream; });

  // Current sheet particles
  injector.inject_pairs(
      // Injection criterion
      [delta] LAMBDA(auto &pos, auto &grid, auto &ext) {
        value_t y = grid.template coord<1>(pos, 0.5f);
        value_t cs_y = 3.0f * delta;
        if (math::abs(y) < cs_y) {
          return true;
        } else {
          return false;
        }
      },
      // Number injected
      [n_cs] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2 * n_cs; },
      // Initialize particles
      [kT_cs, beta_d, boost_beta] LAMBDA(auto &x_global,
                                             rand_state &state, PtcType type) {
        vec_t<value_t, 3> u_d = rng_maxwell_juttner_drifting(state, kT_cs, beta_d);
        value_t gamma_d = math::sqrt(1.0f + u_d.dot(u_d));
        value_t sign = 1.0f;
        if (type == PtcType::positron) sign *= -1.0f;

        value_t beta_shift = 0.995f;
        value_t gamma_shift = 1.0f / math::sqrt(1.0f - beta_shift * beta_shift);

        auto p1 = u_d[1] * sign;
        auto p2 = u_d[2] * sign;
        auto p3 = u_d[0] * sign;
        auto p = vec_t<value_t, 3>(p1, p2, p3);
        value_t gamma = math::sqrt(1.0f + p.dot(p));
        // return lorentz_transform_momentum(p, {boost_beta, 0.0, 0.0});
        vec_t<value_t, 4> p_prime =
            lorentz_transform_vector(gamma, p, {boost_beta, 0.0, 0.0});
        return p_prime.template subset<1, 4>();
      },
      // Particle weight
      [B0, n_cs, q_e, beta_d, delta] LAMBDA(auto &x_global, PtcType type) {
        auto y = x_global[1];
        value_t j = -B0 / delta / square(cosh(y / delta));
        value_t w = math::abs(j) / q_e / n_cs / (2.0f * beta_d);
        return w;
      },
      flag_or(PtcFlag::exclude_from_spectrum));

  Logger::print_info("After initial condition, there are {} particles",
                     ptc.number());
}

template <typename Conf>
void
double_harris_current_sheet(vector_field<Conf> &B, particle_data_t &ptc,
                            rng_states_t<exec_tags::dynamic> &states) {
  using value_t = typename Conf::value_t;
  // auto delta = sim_env().params().get_as<double>("current_sheet_delta", 5.0);
  value_t B_g = sim_env().params().get_as<double>("guide_field", 0.0);
  value_t sigma = sim_env().params().get_as<double>("sigma", 1.0e3);
  value_t kT_cs = sim_env().params().get_as<double>("current_sheet_kT", 1.0);
  value_t kT_upstream = sim_env().params().get_as<double>("upstream_kT", 0.01);
  value_t beta_d =
      sim_env().params().get_as<double>("current_sheet_drift", 0.5);
  value_t gamma_d = 1.0f / math::sqrt(1.0f - beta_d * beta_d);

  value_t delta = 2.0f * kT_cs / (math::sqrt(sigma) * gamma_d * beta_d);
  value_t n_d = gamma_d * sigma / (4.0f * kT_cs);

  value_t global_sizes[Conf::dim];
  value_t global_lower[Conf::dim];

  sim_env().params().get_array("size", global_sizes);
  sim_env().params().get_array("lower", global_lower);

  int n_cs = sim_env().params().get_as<int64_t>("current_sheet_n", 15);
  int n_upstream = sim_env().params().get_as<int64_t>("upstream_n", 5);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);

  // Our unit for length will be upstream c/\omega_p, therefore sigma determines
  // the upstream field strength
  value_t B0 = math::sqrt(sigma);
  auto &grid = B.grid();
  auto ext = grid.extent();
  value_t ysize = global_sizes[1];
  // value_t ylower = global_lower[1];

  // Initialize the magnetic field values. Note that the current sheet is in the
  // x-z plane, and the B field changes sign in the y direction. This should be
  // reflected in the grid setup as well.
  B.set_values(0, [B0, delta, ysize](auto x, auto y, auto z) {
    return double_current_sheet_Bx(B0, y, ysize, delta);
  });
  B.set_values(2, [B0, B_g](auto x, auto y, auto z) { return B0 * B_g; });

  // auto injector =
  //     sim_env().register_system<ptc_injector<Conf, exec_policy_gpu>>(grid);
  // ptc_injector<Conf, exec_policy_gpu> injector(grid);
  ptc_injector_dynamic<Conf> injector(grid);

  // Background (upstream) particles
  injector.inject_pairs(
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      [n_upstream] LAMBDA(auto &pos, auto &grid, auto &ext) {
        return 2 * n_upstream;
      },
      // [kT_upstream] LAMBDA(auto &pos, auto &grid, auto &ext, rng_t<exec_tags::dynamic> &rng,
      //                          PtcType type) {
      //   return rng.maxwell_juttner_3d(kT_upstream);
      [kT_upstream, ysize, B_g, delta] LAMBDA(auto &x_global,
                                       rand_state &state, PtcType type) {
        value_t y = x_global[1];
        value_t Bx = double_current_sheet_Bx(1.0, y, ysize, delta);
        value_t B = math::sqrt(Bx * Bx + B_g * B_g);
        vec_t<value_t, 3> u_d = rng_maxwell_juttner_3d(state, kT_upstream);

        value_t pdotB = (u_d[0] * Bx + u_d[2] * B_g) / B;
        return vec_t<value_t, 3>(pdotB * Bx / B, 0.0, pdotB * B_g / B);
      },
      // [n_upstream] LAMBDA(auto &pos, auto &grid, auto &ext) {
      [n_upstream, q_e] LAMBDA(auto &x_global, PtcType type) {
        return 1.0 / q_e / n_upstream;
      });

  // Current sheet particles
  injector.inject_pairs(
      [delta, ysize] LAMBDA(auto &pos, auto &grid, auto &ext) {
        value_t y = grid.template coord<1>(pos, 0.5f);
        value_t cs_y = 3.0f * delta;
        value_t y1 = 0.25 * ysize;
        value_t y2 = -0.25 * ysize;
        if (math::abs(y - y1) < cs_y || math::abs(y - y2) < cs_y) {
          return true;
        } else {
          return false;
        }
      },
      [n_cs] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2 * n_cs; },
      [kT_cs, beta_d] LAMBDA(auto &x_global, rand_state &state,
                                 PtcType type) {
        vec_t<value_t, 3> u_d = rng_maxwell_juttner_drifting(state, kT_cs, beta_d);
        value_t y = x_global[1];
        value_t sign = (y < 0 ? 1.0f : -1.0f);
        if (type == PtcType::positron) sign *= -1.0f;

        auto p1 = u_d[1] * sign;
        auto p2 = u_d[2] * sign;
        auto p3 = u_d[0] * sign;
        return vec_t<value_t, 3>(p1, p2, p3);
      },
      // [B0, n_cs, q_e, beta_d, delta, ysize] LAMBDA(auto &pos, auto &grid,
      // auto &ext) {
      [B0, n_cs, q_e, beta_d, delta, ysize] LAMBDA(auto &x_global, PtcType type) {
        // auto y = grid.coord(1, pos[1], 0.5f);
        auto y = x_global[1];
        value_t j = 0.0;
        if (y < 0.0f) {
          j = -B0 / delta / square(cosh((y + 0.25f * ysize) / delta));
        } else {
          j = B0 / delta / square(cosh((y - 0.25f * ysize) / delta));
        }
        value_t w = math::abs(j) / q_e / n_cs / (2.0f * beta_d);
        return w;
      },
      // flag_or(PtcFlag::exclude_from_spectrum));
      flag_or(PtcFlag::exclude_from_spectrum, PtcFlag::ignore_tracking));

  Logger::print_info("After initial condition, there are {} particles",
                     ptc.number());
}

template <typename Conf>
void
ffe_current_sheet(vector_field<Conf> &B,
                  particle_data_t &ptc,
                  rng_states_t<exec_tags::dynamic> &states) {
  using value_t = typename Conf::value_t;
  // auto delta = sim_env().params().get_as<double>("current_sheet_delta", 5.0);
  value_t B_g = sim_env().params().get_as<double>("guide_field", 0.0);
  value_t sigma = sim_env().params().get_as<double>("sigma", 1.0e3);
  value_t kT_cs = sim_env().params().get_as<double>("current_sheet_kT", 1.0);
  value_t kT_upstream = sim_env().params().get_as<double>("upstream_kT", 0.01);
  value_t beta_d =
      sim_env().params().get_as<double>("current_sheet_drift", 0.5);
  value_t gamma_d = 1.0f / math::sqrt(1.0f - beta_d * beta_d);

  value_t delta = 2.0f * kT_cs / (math::sqrt(sigma) * gamma_d * beta_d);
  value_t n_d = gamma_d * sigma / (4.0f * kT_cs);

  value_t global_sizes[Conf::dim];
  value_t global_lower[Conf::dim];

  sim_env().params().get_array("size", global_sizes);
  sim_env().params().get_array("lower", global_lower);

  int n_cs = sim_env().params().get_as<int64_t>("current_sheet_n", 15);
  int n_upstream = sim_env().params().get_as<int64_t>("upstream_n", 5);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);

  // Our unit for length will be upstream c/\omega_p, therefore sigma determines
  // the upstream field strength
  value_t B0 = math::sqrt(sigma);
  auto &grid = B.grid();
  auto ext = grid.extent();
  value_t ysize = global_sizes[1];
  // value_t ylower = global_lower[1];

  // Initialize the magnetic field values. Note that the current sheet is in the
  // x-z plane, and the B field changes sign in the y direction. This should be
  // reflected in the grid setup as well.
  B.set_values(0, [B0, delta, ysize](auto x, auto y, auto z) {
    return double_current_sheet_Bx(B0, y, ysize, delta);
  });
  B.set_values(2, [B0, B_g](auto x, auto y, auto z) { return B0 * B_g; });

  // auto injector =
  //     sim_env().register_system<ptc_injector<Conf, exec_policy_gpu>>(grid);
  // ptc_injector<Conf, exec_policy_gpu> injector(grid);
  ptc_injector_dynamic<Conf> injector(grid);

  // Background (upstream) particles
  injector.inject_pairs(
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      [n_upstream] LAMBDA(auto &pos, auto &grid, auto &ext) {
        return 2 * n_upstream;
      },
      // [kT_upstream] LAMBDA(auto &pos, auto &grid, auto &ext, rng_t<exec_tags::dynamic> &rng,
      //                          PtcType type) {
      //   return rng.maxwell_juttner_3d(kT_upstream);
      [kT_upstream, ysize, B_g, delta] LAMBDA(auto &x_global,
                                       rand_state &state, PtcType type) {
        value_t y = x_global[1];
        value_t Bx = double_current_sheet_Bx(1.0, y, ysize, delta);
        value_t B = math::sqrt(Bx * Bx + B_g * B_g);
        vec_t<value_t, 3> u_d = rng_maxwell_juttner_3d(state, kT_upstream);

        value_t pdotB = (u_d[0] * Bx + u_d[2] * B_g) / B;
        return vec_t<value_t, 3>(pdotB * Bx / B, 0.0, pdotB * B_g / B);
      },
      // [n_upstream] LAMBDA(auto &pos, auto &grid, auto &ext) {
      [n_upstream, q_e] LAMBDA(auto &x_global, PtcType type) {
        return 1.0 / q_e / n_upstream;
      });

  // Current sheet particles
  injector.inject_pairs(
      [delta, ysize] LAMBDA(auto &pos, auto &grid, auto &ext) {
        value_t y = grid.template coord<1>(pos, 0.5f);
        value_t cs_y = 3.0f * delta;
        value_t y1 = 0.25 * ysize;
        value_t y2 = -0.25 * ysize;
        if (math::abs(y - y1) < cs_y || math::abs(y - y2) < cs_y) {
          return true;
        } else {
          return false;
        }
      },
      [n_cs] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2 * n_cs; },
      [kT_cs, beta_d] LAMBDA(auto &x_global, rand_state &state,
                                 PtcType type) {
        vec_t<value_t, 3> u_d = rng_maxwell_juttner_drifting(state, kT_cs, beta_d);
        value_t y = x_global[1];
        value_t sign = (y < 0 ? 1.0f : -1.0f);
        if (type == PtcType::positron) sign *= -1.0f;

        auto p1 = u_d[1] * sign;
        auto p2 = u_d[2] * sign;
        auto p3 = u_d[0] * sign;
        return vec_t<value_t, 3>(p1, p2, p3);
      },
      // [B0, n_cs, q_e, beta_d, delta, ysize] LAMBDA(auto &pos, auto &grid,
      // auto &ext) {
      [B0, n_cs, q_e, beta_d, delta, ysize] LAMBDA(auto &x_global, PtcType type) {
        // auto y = grid.coord(1, pos[1], 0.5f);
        auto y = x_global[1];
        value_t j = 0.0;
        if (y < 0.0f) {
          j = -B0 / delta / square(cosh((y + 0.25f * ysize) / delta));
        } else {
          j = B0 / delta / square(cosh((y - 0.25f * ysize) / delta));
        }
        value_t w = math::abs(j) / q_e / n_cs / (2.0f * beta_d);
        return w;
      },
      // flag_or(PtcFlag::exclude_from_spectrum));
      flag_or(PtcFlag::exclude_from_spectrum, PtcFlag::ignore_tracking));

  Logger::print_info("After initial condition, there are {} particles",
                     ptc.number());
}

template void ffe_current_sheet<Config<2>>(vector_field<Config<2>> &B,
                                           particle_data_t &ptc,
                                           rng_states_t<exec_tags::dynamic> &states);

template void harris_current_sheet<Config<2>>(vector_field<Config<2>> &B,
                                              particle_data_t &ptc,
                                              rng_states_t<exec_tags::dynamic> &states);

template void boosted_harris_sheet<Config<2>>(vector_field<Config<2>> &B,
                                              particle_data_t &ptc,
                                              rng_states_t<exec_tags::dynamic> &states);

template void double_harris_current_sheet<Config<2>>(vector_field<Config<2>> &B,
                                                     particle_data_t &ptc,
                                                     rng_states_t<exec_tags::dynamic> &states);

template void double_harris_current_sheet<Config<3>>(vector_field<Config<3>> &B,
                                                     particle_data_t &ptc,
                                                     rng_states_t<exec_tags::dynamic> &states);

}  // namespace Aperture
