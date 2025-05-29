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

// Single current sheet centered at y=0
HOST_DEVICE Scalar current_sheet_Bx(Scalar B0, Scalar y, Scalar delta) {
    return B0 * tanh(y / delta);
}

HOST_DEVICE Scalar logcosh(Scalar x) {
    Scalar absx = std::abs(x);
    return ((absx > 18.5) ? absx - std::log(2.0) : std::log(std::cosh(x)));
}

HOST_DEVICE Scalar perturbed_Az(Scalar B0, Scalar x, Scalar y, Scalar ysize, Scalar delta, Scalar amp, Scalar lambda, Scalar phase) {
    Scalar result;
    result  = B0 * delta;
    result *= (1 - amp * std::sin(2 * M_PI * (x - phase) / lambda) * square(std::cos(2 * M_PI * y / ysize)));
    result *= (logcosh(y / delta) - logcosh(ysize / (2 * delta)));
    return result;
}

// Single perturbed current sheet centered at y=0
HOST_DEVICE Scalar perturbed_current_sheet_Bx(Scalar B0, Scalar x, Scalar y, Scalar ysize, Scalar delta, Scalar amp, Scalar lambda, Scalar phase) {
    Scalar eps = delta/1e4;
    Scalar result;
    result  = perturbed_Az(B0, x, y + eps, ysize, delta, amp, lambda, phase);
    result -= perturbed_Az(B0, x, y - eps, ysize, delta, amp, lambda, phase);
    result /= (2 * eps);
    return result;
}

// Single perturbed current sheet centered at y=0
HOST_DEVICE Scalar perturbed_current_sheet_By(Scalar B0, Scalar x, Scalar y, Scalar ysize, Scalar delta, Scalar amp, Scalar lambda, Scalar phase) {
    Scalar eps = delta/1e4;
    Scalar result;
    result  = perturbed_Az(B0, x + eps, y, ysize, delta, amp, lambda, phase);
    result -= perturbed_Az(B0, x - eps, y, ysize, delta, amp, lambda, phase);
    result /= (-2 * eps);
    return result;
}

HOST_DEVICE Scalar sech(Scalar x) {
    return 1.0f / std::cosh(x);
}

HOST_DEVICE Scalar double_ffe_sheet_Bz(Scalar B0, Scalar y, Scalar ysize, Scalar delta, Scalar b_g) {
  if (y < 0.0f) {
    return B0 * math::sqrt(square(sech((y + 0.25f * ysize) / delta)) + b_g * b_g);
  } else {
    return B0 * math::sqrt(square(sech((y - 0.25f * ysize) / delta)) + b_g * b_g);
  }
}

// Single ffe sheet centered at y=0
HOST_DEVICE Scalar ffe_sheet_Bz(Scalar B0, Scalar y, Scalar delta, Scalar b_g) {
    return B0 * math::sqrt(square(sech(y / delta)) + b_g * b_g);
}

HOST_DEVICE Scalar beta_dx(Scalar B0, Scalar b_g, Scalar yarg, Scalar beta_d) {
  return -beta_d / math::sqrt(1.0f + b_g*b_g) * std::tanh(yarg);
}

HOST_DEVICE Scalar beta_dz(Scalar B0, Scalar b_g, Scalar yarg, Scalar beta_d) {
  return beta_d / math::sqrt(1.0f + b_g*b_g) * math::sqrt(square(sech(yarg)) + b_g*b_g);
}

HOST_DEVICE Scalar drift_frac(Scalar yarg, Scalar b_g) {
    return sech(yarg) * math::sqrt((1.0f + b_g * b_g) / (1.0f + square(b_g * std::cosh(yarg))));
}

}

template <typename Conf>
void
harris_current_sheet(vector_field<Conf> &B, 
                     vector_field<Conf> &Bdelta,
                     vector_field<Conf> &J0,
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
  value_t rho_upstream = sim_env().params().get_as<double>("upstream_rho", 1.0);

  // The "sigma_cs" parameter triggers using current sheet density for normalization
  if (sim_env().params().has("sigma_cs")) {
    sigma = sim_env().params().get_as<double>("sigma_cs", sigma);
    value_t B0 = math::sqrt(sigma);
    delta = B0 / 2.0 / beta_d;
    n_d = 1.0; // n_d is actually not used in our setup
    kT_cs = sigma / 4.0;
  }
  // Logger::print_info("CS delta = {}", delta);
  // Logger::print_info("sigma = {}", sigma);

  int n_cs = sim_env().params().get_as<int64_t>("current_sheet_n", 15);
  int n_upstream = sim_env().params().get_as<int64_t>("upstream_n", 5);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);

  // Our unit for length will be upstream c/\omega_p, therefore sigma determines
  // the upstream field strength
  value_t B0 = math::sqrt(sigma);
  auto &grid = B.grid();
  auto ext = grid.extent();
  value_t ysize = global_sizes[1];
  value_t xsize = global_sizes[0];
  value_t ylower = global_lower[1];
  value_t xlower = global_lower[0];
  value_t xmid = xlower + 0.5 * xsize;

  value_t perturb_amp = sim_env().params().get_as<double>("perturbation_amp", 0.0);
  value_t perturb_lambda = sim_env().params().get_as<double>("perturbation_wavelength", 2*xsize);
  value_t perturb_phase = sim_env().params().get_as<double>("perturbation_phase", 0.0);

  // Don't remove pressure support from the initial current sheet unless asked
  // to explicitly
  value_t depressurize_x = sim_env().params().get_as<double>("depressurize_at_x", xlower - xsize);
  value_t depressurize_cs_widths = sim_env().params().get_as<double>("depressurize_cs_widths", 0.0);

  // Initialize the magnetic field values. Note that the current sheet is in the
  // x-z plane, and the B field changes sign in the y direction. This should be
  // reflected in the grid setup as well.
  B.set_values(0, [B0, delta, ysize](auto x, auto y, auto z) {
    return B0 * tanh(y / delta);
  });
  B.set_values(2, [B0, B_g](auto x, auto y, auto z) { return B0 * B_g; });
  J0.set_values(2, [B0, delta, ysize](auto x, auto y, auto z) {
    value_t j = B0 / delta / square(cosh(y / delta));
    return j;
  });
  // Bdelta.set_values(0, [B0, ysize, delta, perturb_amp, perturb_lambda, perturb_phase](auto x, auto y, auto z) {
  //   return perturbed_current_sheet_Bx(B0, x, y, ysize, delta, perturb_amp, perturb_lambda, perturb_phase) - B0 * tanh(y / delta);
  // });
  // Bdelta.set_values(1, [B0, ysize, delta, perturb_amp, perturb_lambda, perturb_phase](auto x, auto y, auto z) {
  //   return perturbed_current_sheet_By(B0, x, y, ysize, delta, perturb_amp, perturb_lambda, perturb_phase);
  // });

  // Logger::print_info(
  //     "B0 = {}; delta = {}; J0 = B0 / delta = {}; beta_drift = {}",
  //     B0, delta, B0 / delta, beta_d
  // );
  // Logger::print_info(
  //     "square(cosh(0)) = {}",
  //     square(cosh(0.0))
  // );

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
      [n_upstream, rho_upstream, q_e] LAMBDA(auto &x_global, PtcType type) {
        return rho_upstream / q_e / n_upstream;
      });

  // Current sheet particles
  injector.inject_pairs(
      [delta, ysize] LAMBDA(auto &pos, auto &grid, auto &ext) {
        value_t y = grid.template coord<1>(pos, 0.5f);
        value_t cs_y = 3.0f * delta;
        if (math::abs(y) < cs_y) {
          return true;
        } else {
          return false;
        }
      },
      [n_cs] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2 * n_cs; },
      [kT_cs, beta_d, xmid, delta, depressurize_x, depressurize_cs_widths] 
          LAMBDA(auto &x_global, rand_state &state, PtcType type) {

        value_t y = x_global[1];
        value_t x = x_global[0];
        vec_t<value_t, 3> u_d = rng_maxwell_juttner_drifting(state, kT_cs, beta_d);
        // value_t sign = (y < 0 ? 1.0f : -1.0f);
        value_t sign = 1.0f;
        if (type == PtcType::positron) sign *= -1.0f;

        // Remove pressure support in middle of CS by making plasma cold there
        if (math::abs(x - depressurize_x) < depressurize_cs_widths * delta) {
            u_d[0] = beta_d;
            u_d[1] = 0.0f;
            u_d[2] = 0.0f;
        }

        auto p1 = u_d[1] * sign;
        auto p2 = u_d[2] * sign;
        auto p3 = u_d[0] * sign;

        if (p1 != p1 || p2 != p2 || p3 != p3) {
          printf("NaN in current sheet! y = %f\n", y);
        }
        return vec_t<value_t, 3>(p1, p2, p3);
      },
      // [B0, n_cs, q_e, beta_d, delta, ysize] LAMBDA(auto &pos, auto &grid,
      // auto &ext) {
      [B0, n_cs, q_e, beta_d, delta, ysize] LAMBDA(auto &x_global, PtcType type) {
        // auto y = grid.coord(1, pos[1], 0.5f);
        auto y = x_global[1];
        value_t j = -B0 / delta / square(cosh(y / delta));
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
double_harris_current_sheet(vector_field<Conf> &B,
                            vector_field<Conf> &J0,
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
  J0.set_values(2, [B0, delta, ysize](auto x, auto y, auto z) {
    value_t j = 0.0f;
    if (y < 0.0f) {
      j = B0 / delta / square(cosh((y + 0.25f * ysize) / delta));
    } else {
      j = -B0 / delta / square(cosh((y - 0.25f * ysize) / delta));
    }
    return j;
  });

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
        if (p1 != p1 || p2 != p2 || p3 != p3) {
          printf("NaN in current sheet! y = %f\n", y);
        }
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

template <typename Conf> void ffe_current_sheet(
        vector_field<Conf> &B,
        vector_field<Conf> &Bbg,
        vector_field<Conf> &J0,
        particle_data_t &ptc,
        rng_states_t<exec_tags::device> &states
    )
{

  using value_t = typename Conf::value_t;
  value_t B_g = sim_env().params().get_as<double>("guide_field", 0.0);
  value_t sigma = sim_env().params().get_as<double>("sigma", 1.0e3);
  value_t kT_cs = sim_env().params().get_as<double>("current_sheet_kT", 1.0);
  value_t kT_upstream = sim_env().params().get_as<double>("upstream_kT", 1.0);
  value_t beta_d =
      sim_env().params().get_as<double>("current_sheet_drift", 0.5);
  value_t gamma_d = 1.0f / math::sqrt(1.0f - beta_d * beta_d);

  // value_t delta = 2.0f * kT_cs / (math::sqrt(sigma) * gamma_d * beta_d);
  // value_t n_d = gamma_d * sigma / (4.0f * kT_cs);

  value_t global_sizes[Conf::dim];
  value_t global_lower[Conf::dim];

  sim_env().params().get_array("size", global_sizes);
  sim_env().params().get_array("lower", global_lower);
  value_t rho_upstream = sim_env().params().get_as<double>("upstream_rho", 1.0);
  value_t n_d;

  // Our unit for length will be upstream c/\omega_p, therefore sigma determines
  // the upstream field strength
  value_t B0 = math::sqrt(sigma);
  //
  // implicitly assuming n_cs = 1.0 for each species
  value_t delta = B0 / (2.0f * beta_d);

  // The "sigma_cs" parameter triggers using current sheet density for normalization
  if (sim_env().params().has("sigma_cs")) {
    sigma = sim_env().params().get_as<double>("sigma_cs", sigma);
    value_t B0 = math::sqrt(sigma);
    delta = B0 / 2.0 / beta_d;
    n_d = 1.0; // n_d is actually not used in our setup
    kT_cs = sigma / 4.0;
  }

  int n_cs = sim_env().params().get_as<int64_t>("current_sheet_n", 15);
  int n_upstream = sim_env().params().get_as<int64_t>("upstream_n", 5);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);

  auto &grid = B.grid();
  auto ext = grid.extent();
  value_t ysize = global_sizes[1];
  value_t xsize = global_sizes[0];
  // value_t ylower = global_lower[1];

  value_t perturb_amp = sim_env().params().get_as<double>("perturbation_amp", 0.0);
  value_t perturb_lambda = sim_env().params().get_as<double>("perturbation_wavelength", 2*xsize);
  value_t perturb_phase = sim_env().params().get_as<double>("perturbation_phase", 0.0);

  // Initialize the magnetic field values. Note that the current sheet is in the
  // x-z plane, and the B field changes sign in the y direction. This should be
  // reflected in the grid setup as well.
  Bbg.set_values(0, [B0, ysize, delta, perturb_amp, perturb_lambda, perturb_phase](auto x, auto y, auto z) {
    return B0 * tanh(y / delta);
  });
  B.set_values(0, [B0, ysize, delta, perturb_amp, perturb_lambda, perturb_phase](auto x, auto y, auto z) {
    return perturbed_current_sheet_Bx(B0, x, y, ysize, delta, perturb_amp, perturb_lambda, perturb_phase) - B0 * tanh(y / delta);
  });
  B.set_values(1, [B0, ysize, delta, perturb_amp, perturb_lambda, perturb_phase](auto x, auto y, auto z) {
    return perturbed_current_sheet_By(B0, x, y, ysize, delta, perturb_amp, perturb_lambda, perturb_phase);
  });
  B.set_values(2, [B0, delta, B_g](auto x, auto y, auto z) {
    return ffe_sheet_Bz(B0, y, delta, B_g);
  });
  // J0 is set to be the negative of curl B
  // Refer to formulae in Mehlhaff+ 2024, MNRAS 527 11587
  // Note since we set to the negative of curl B, these J0 differ by an
  // overall minus sign from Mehlhaff+ 2024.
  // J0.set_values(0, [B0, delta, beta_d, B_g](auto x, auto y, auto z) {
  //   Scalar yarg = y / delta;
  //   return -2.0 * beta_dx(B0, B_g, yarg, beta_d) * drift_frac(yarg, B_g);
  // });
  J0.set_values(2, [B0, delta, beta_d, B_g](auto x, auto y, auto z) {
    Scalar yarg = y / delta;
    return 2.0 * beta_dz(B0, B_g, yarg, beta_d) * drift_frac(yarg, B_g);
  });

  ptc_injector_dynamic<Conf> injector(grid);

  // Background (upstream) particles
  injector.inject_pairs(
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },

      [n_upstream] LAMBDA(auto &pos, auto &grid, auto &ext) {
        return 2 * n_upstream;
      },

      [kT_upstream] LAMBDA(
          auto &x_global, rand_state &state, PtcType type) {
        return rng_maxwell_juttner_3d(state, kT_upstream);
      },

      [n_upstream, q_e, delta, B_g] LAMBDA(
          auto &x_global, PtcType type) {
        auto y = x_global[1];
        value_t yarg = y / delta;
        return 1.0 / q_e / n_upstream * std::max(0.0, (1.0f - drift_frac(yarg, B_g)));
      });

  // Current sheet particles
  injector.inject_pairs(
      [delta] LAMBDA(auto &pos, auto &grid, auto &ext) {
        value_t y = grid.template coord<1>(pos, 0.5f);
        value_t cs_y = 5.0f * delta;
        // value_t y1 = 0.25 * ysize;
        // value_t y2 = -0.25 * ysize;
        // if (math::abs(y - y1) < cs_y || math::abs(y - y2) < cs_y) {
        //   return true;
        // } else {
        //   return false;
        // }
        value_t y1 = 0.0f;
        if (math::abs(y - y1) < cs_y) {
          return true;
        } else {
          return false;
        }
      },

      [n_cs] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2 * n_cs; },

      [kT_cs, beta_d, delta, B_g] LAMBDA(
          auto &x_global, rand_state &state, PtcType type) {
        // Note that particle drift velocity, betad, is along u_d[0]
        vec_t<value_t, 3> u_d = rng_maxwell_juttner_drifting(state, kT_cs, beta_d);
        value_t y = x_global[1];
        value_t sign;  // = (y < 0 ? 1.0f : -1.0f);
        // drift_dir_[x|z]: Unit vectors pointing in the particle drift
        // direction. These rotate through the current layer(s) to rotate
        // the resulting current density J
        value_t drift_dir_x;
        value_t drift_dir_z;
        value_t yarg = y / delta;
        drift_dir_x = -tanh(yarg) / sqrt(1.0f + B_g * B_g);
        drift_dir_z = -sqrt(square(sech(yarg)) + B_g * B_g) / sqrt(1.0f + B_g * B_g);

        // At this point, the drift velocity is exactly what it needs to be for
        // positrons. It needs to be changed by an overall sign if we are
        // dealing with electrons
        // if (type == PtcType::electron) {
        //     sign = -1.0f;
        // } else {
        //     sign = 1.0f;
        // }
        sign = (type == PtcType::electron ? -1.0f : 1.0f);

        // Rotate particle drift velocity so that drift direction is drift_dir
        // Rotation is in the x/z plane
        auto p1 = drift_dir_x * u_d[0] + 0.0 * u_d[1] - drift_dir_z * u_d[2];
        auto p2 =         0.0 * u_d[0] + 1.0 * u_d[1] +         0.0 * u_d[2];
        auto p3 = drift_dir_z * u_d[0] + 0.0 * u_d[1] + drift_dir_x * u_d[2];

        p1 = p1 * sign;
        p2 = p2 * sign;
        p3 = p3 * sign;

        return vec_t<value_t, 3>(p1, p2, p3);
      },

      [B0, n_cs, q_e, beta_d, delta, B_g] LAMBDA(
            auto &x_global, PtcType type) {
        auto y = x_global[1];
        value_t j = 0.0;
        value_t yarg = y / delta;
        j = (B0 / delta) * (1.0f / (2.0f * beta_d));
        j = j * drift_frac(yarg, B_g);
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
double_ffe_current_sheet(vector_field<Conf> &B, vector_field<Conf> &J0, particle_data_t &ptc,
                            rng_states_t<exec_tags::device> &states) {
  using value_t = typename Conf::value_t;
  // auto delta = sim_env().params().get_as<double>("current_sheet_delta", 5.0);
  value_t B_g = sim_env().params().get_as<double>("guide_field", 0.0);
  value_t sigma = sim_env().params().get_as<double>("sigma", 1.0e3);
  value_t kT_cs = sim_env().params().get_as<double>("current_sheet_kT", 1.0);
  value_t kT_upstream = sim_env().params().get_as<double>("upstream_kT", 0.01);
  value_t beta_d =
      sim_env().params().get_as<double>("current_sheet_drift", 0.5);
  value_t gamma_d = 1.0f / math::sqrt(1.0f - beta_d * beta_d);

  // value_t delta = 2.0f * kT_cs / (math::sqrt(sigma) * gamma_d * beta_d);
  // value_t n_d = gamma_d * sigma / (4.0f * kT_cs);

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

  // implicitly assuming n_cs = 1.0 for each species
  value_t delta = B0 / (2.0f * beta_d);

  // Initialize the magnetic field values. Note that the current sheet is in the
  // x-z plane, and the B field changes sign in the y direction. This should be
  // reflected in the grid setup as well.
  B.set_values(0, [B0, delta, ysize](auto x, auto y, auto z) {
    return double_current_sheet_Bx(B0, y, ysize, delta);
  });
  // B.set_values(2, [B0, B_g](auto x, auto y, auto z) { return B0 * B_g; });
  B.set_values(2, [B0, delta, ysize, B_g](auto x, auto y, auto z) {
    return double_ffe_sheet_Bz(B0, y, ysize, delta, B_g);
  });
  // J0 is set to be the negative of curl B
  J0.set_values(0, [B0, delta, ysize, beta_d, B_g](auto x, auto y, auto z) {
    Scalar yarg = 0.0;
    if (y < 0.0f) {
      yarg = (y + 0.25f * ysize) / delta;
    } else {
      yarg = (y - 0.25f * ysize) / delta;
    }

    return -2.0 * beta_dx(B0, B_g, yarg, beta_d) * drift_frac(yarg, B_g);
  });
  J0.set_values(2, [B0, delta, ysize, beta_d, B_g](auto x, auto y, auto z) {
    Scalar yarg = 0.0;
    if (y < 0.0f) {
      yarg = (y + 0.25f * ysize) / delta;
    } else {
      yarg = (y - 0.25f * ysize) / delta;
    }
    if (y < 0.0f) {
      return 2.0 * beta_dz(B0, B_g, yarg, beta_d) * drift_frac(yarg, B_g);
    } else {
      return -2.0 * beta_dz(B0, B_g, yarg, beta_d) * drift_frac(yarg, B_g);
    }
  });

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
      // [kT_upstream] LAMBDA(auto &pos, auto &grid, auto &ext, rng_t<exec_tags::device> &rng,
      //                          PtcType type) {
      [kT_upstream] LAMBDA(auto &x_global, rand_state &state,
                                 PtcType type) {
        // return rng.maxwell_juttner_3d(kT_upstream);
        return rng_maxwell_juttner_3d(state, kT_upstream);
      },
      // [kT_upstream, ysize, B_g, delta] LAMBDA(auto &x_global,
      //                                  rand_state &state, PtcType type) {
      //   value_t y = x_global[1];
      //   value_t Bx = double_current_sheet_Bx(1.0, y, ysize, delta);
      //   value_t B = math::sqrt(Bx * Bx + B_g * B_g);
      //   vec_t<value_t, 3> u_d = rng_maxwell_juttner_3d(state, kT_upstream);

      //   value_t pdotB = (u_d[0] * Bx + u_d[2] * B_g) / B;
      //   return vec_t<value_t, 3>(pdotB * Bx / B, 0.0, pdotB * B_g / B);
      // },
      // [n_upstream] LAMBDA(auto &pos, auto &grid, auto &ext) {
      // [n_upstream, q_e] LAMBDA(auto &x_global, PtcType type) {
      //   return 1.0 / q_e / n_upstream;
      // });
      [n_upstream, q_e, delta, ysize, B_g] LAMBDA(auto &x_global, PtcType type) {
        auto y = x_global[1];
        value_t yarg = 0.0;
        if (y < 0.0f) {
          yarg = (y + 0.25f * ysize) / delta;
        } else {
          yarg = (y - 0.25f * ysize) / delta;
        }
        return 1.0 / q_e / n_upstream * std::max(0.0, (1.0f - drift_frac(yarg, B_g)));
        // return 1.0 / q_e / n_upstream;
      });

  // Current sheet particles
  injector.inject_pairs(
      [delta, ysize] LAMBDA(auto &pos, auto &grid, auto &ext) {
        value_t y = grid.template coord<1>(pos, 0.5f);
        value_t cs_y = 5.0f * delta;
        value_t y1 = 0.25 * ysize;
        value_t y2 = -0.25 * ysize;
        if (math::abs(y - y1) < cs_y || math::abs(y - y2) < cs_y) {
          return true;
        } else {
          return false;
        }
      },
      [n_cs] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2 * n_cs; },
      [kT_cs, beta_d, delta, ysize, B_g] LAMBDA(auto &x_global, rand_state &state,
                                 PtcType type) {
        // Note that particle drift velocity, betad, is along u_d[0]
        vec_t<value_t, 3> u_d = rng_maxwell_juttner_drifting(state, kT_cs, beta_d);
        value_t y = x_global[1];
        value_t sign = (y < 0 ? 1.0f : -1.0f);
        // drift_dir_[x|z]: Unit vectors pointing in the particle drift
        // direction. These rotate through the current layer(s) to rotate
        // the resulting current density J
        value_t drift_dir_x;
        value_t drift_dir_z;
        value_t yarg = 0.0;
        if (y < 0.0f) {
          yarg = (y + 0.25f * ysize) / delta;
        } else {
          yarg = (y - 0.25f * ysize) / delta;
        }
        drift_dir_x = -tanh(yarg) / sqrt(1.0f + B_g * B_g);
        drift_dir_z = -sqrt(square(sech(yarg)) + B_g * B_g) / sqrt(1.0f + B_g * B_g);
        // drift_dir_x does not need to be corrected here because x-drift is
        // the same in both the top and bottom layers
        drift_dir_z *= sign;

        // if (type == PtcType::positron) sign *= -1.0f;
        // At this point, the drift velocity is exactly what it needs to be for
        // positrons. It needs to be changed by an overall sign if we are
        // dealing with electrons
        // if (type == PtcType::electron) {
        //     sign = -1.0f;
        // } else {
        //     sign = 1.0f;
        // }
        sign = (type == PtcType::electron ? -1.0f : 1.0f);

        // Rotate particle drift velocity so that drift direction is drift_dir
        // Rotation is in the x/z plane
        auto p1 = drift_dir_x * u_d[0] + 0.0 * u_d[1] - drift_dir_z * u_d[2];
        auto p2 =         0.0 * u_d[0] + 1.0 * u_d[1] +         0.0 * u_d[2];
        auto p3 = drift_dir_z * u_d[0] + 0.0 * u_d[1] + drift_dir_x * u_d[2];

        p1 = p1 * sign;
        p2 = p2 * sign;
        p3 = p3 * sign;

        // auto p1 = u_d[1] * sign;
        // auto p2 = u_d[2] * sign;
        // auto p3 = u_d[0] * sign;
        return vec_t<value_t, 3>(p1, p2, p3);
      },
      // [B0, n_cs, q_e, beta_d, delta, ysize] LAMBDA(auto &pos, auto &grid,
      // auto &ext) {
      [B0, n_cs, q_e, beta_d, delta, ysize, B_g] LAMBDA(auto &x_global, PtcType type) {
        // auto y = grid.coord(1, pos[1], 0.5f);
        auto y = x_global[1];
        value_t j = 0.0;
        value_t yarg = 0.0;
        j = (B0 / delta) * (1.0f / (2.0f * beta_d));
        if (y < 0.0f) {
          // j = -B0 / delta / square(cosh((y + 0.25f * ysize) / delta));
          yarg = (y + 0.25f * ysize) / delta;
        } else {
          // j = B0 / delta / square(cosh((y - 0.25f * ysize) / delta));
          yarg = (y - 0.25f * ysize) / delta;
        }
        j = j * drift_frac(yarg, B_g);
        value_t w = math::abs(j) / q_e / n_cs / (2.0f * beta_d);
        return w;
      },
      // flag_or(PtcFlag::exclude_from_spectrum));
      flag_or(PtcFlag::exclude_from_spectrum, PtcFlag::ignore_tracking));

  Logger::print_info("After initial condition, there are {} particles",
                     ptc.number());
}

template void harris_current_sheet<Config<2>>(vector_field<Config<2>> &B,
                                              vector_field<Config<2>> &Bdelta, 
                                              vector_field<Config<2>> &J0,
                                              particle_data_t &ptc,
                                              rng_states_t<exec_tags::dynamic> &states);

// template void boosted_harris_sheet<Config<2>>(vector_field<Config<2>> &B,
//                                               particle_data_t &ptc,
//                                               rng_states_t<exec_tags::dynamic> &states);

template void double_harris_current_sheet<Config<2>>(vector_field<Config<2>> &B,
                                                     vector_field<Config<2>> &J0,
                                                     particle_data_t &ptc,
                                                     rng_states_t<exec_tags::dynamic> &states);

template void double_harris_current_sheet<Config<3>>(vector_field<Config<3>> &B,
                                                     vector_field<Config<3>> &J0,
                                                     particle_data_t &ptc,
                                                     rng_states_t<exec_tags::dynamic> &states);

template void double_ffe_current_sheet<Config<2>>(vector_field<Config<2>> &B,
                                                  vector_field<Config<2>> &J0,
                                                  particle_data_t &ptc,
                                                  rng_states_t<exec_tags::device> &states);

template void double_ffe_current_sheet<Config<3>>(vector_field<Config<3>> &B,
                                                  vector_field<Config<3>> &J0,
                                                  particle_data_t &ptc,
                                                  rng_states_t<exec_tags::device> &states);

template void ffe_current_sheet<Config<2>>(vector_field<Config<2>> &B,
                                                  vector_field<Config<2>> &B0,
                                                  vector_field<Config<2>> &J0,
                                                  particle_data_t &ptc,
                                                  rng_states_t<exec_tags::device> &states);

template void ffe_current_sheet<Config<3>>(vector_field<Config<3>> &B,
                                                  vector_field<Config<3>> &B0,
                                                  vector_field<Config<3>> &J0,
                                                  particle_data_t &ptc,
                                                  rng_states_t<exec_tags::device> &states);

}  // namespace Aperture
