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
#include "data/curand_states.h"
#include "data/fields.h"
#include "data/particle_data.h"
#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/policies/exec_policy_cuda.hpp"
#include "utils/kernel_helper.hpp"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace {}  // namespace

namespace Aperture {

template <typename Conf>
void
harris_current_sheet(vector_field<Conf> &B, particle_data_t &ptc,
                     rng_states_t &states, int mult) {
  using value_t = typename Conf::value_t;
  auto delta = sim_env().params().get_as<double>("current_sheet_delta", 5.0);
  auto kT_cs = sim_env().params().get_as<double>("current_sheet_kT", 1.0);
  auto beta_d = sim_env().params().get_as<double>("current_sheet_drift", 0.5);
  int n_cs = sim_env().params().get_as<int64_t>("current_sheet_n", 5);
  int n_upstream = sim_env().params().get_as<int64_t>("upstream_n", 5);
  auto sigma = sim_env().params().get_as<double>("sigma", 1.0e3);
  auto q_e = sim_env().params().get_as<double>("q_e", 1.0);

  // Our unit for length will be upstream c/\omega_p, therefore sigma determines
  // the upstream field strength
  value_t B0 = math::sqrt(sigma);

  // Initialize the magnetic field values
  B.set_values(
      0, [B0, delta](auto x, auto y, auto z) { return B0 * tanh(y / delta); });

  // Initialize the particles
  auto num = ptc.number();
  // Define a variable to hold the moving position in the photon array where we
  // insert new photons
  buffer<unsigned long long int> ptc_pos(1);
  ptc_pos[0] = num;
  ptc_pos.copy_to_device();
  // auto policy = exec_policy_cuda<Conf>{};
  using policy = exec_policy_cuda<Conf>;
  policy::launch(
      [delta, kT_cs, beta_d, n_cs, n_upstream, B0, q_e] __device__(
          auto ptc, auto states, auto ptc_pos) {
        auto &grid = policy::grid();
        auto ext = grid.extent();
        rng_t rng(states);

        policy::loop(
            Conf::begin(ext), Conf::end(ext),
            [delta, kT_cs, beta_d, n_cs, n_upstream, B0, q_e, &grid, &ext,
             &rng] __device__(auto idx, auto &ptc, auto &ptc_pos) {
              auto pos = get_pos(idx, ext);
              if (!grid.is_in_bound(pos)) return;

              // grid center position in y
              auto y = grid.pos(1, pos[1], 0.5f);
              value_t j = -B0 / delta / square(cosh(y / delta));
              value_t w = j / q_e / n_cs / beta_d;

              value_t cs_n = 5.0f;
              if (y < -cs_n || y > cs_n) {
                // Background plasma
                for (int i = 0; i < n_upstream; i++) {
                  auto offset = atomic_add(ptc_pos, 2);
                  auto offset_e = offset;
                  auto offset_p = offset + 1;

                  ptc.x1[offset_e] = ptc.x1[offset_p] = rng.uniform<value_t>();
                  ptc.x2[offset_e] = ptc.x2[offset_p] = rng.uniform<value_t>();
                  ptc.x3[offset_e] = ptc.x3[offset_p] = rng.uniform<value_t>();

                  auto u = rng.maxwell_juttner_3d(1.0e-3);
                  ptc.p1[offset_e] = ptc.p1[offset_p] = u[0];
                  ptc.p2[offset_e] = ptc.p2[offset_p] = u[1];
                  ptc.p3[offset_e] = ptc.p3[offset_p] = u[2];
                  ptc.E[offset_e] = ptc.E[offset_p] =
                      math::sqrt(1.0f + u.dot(u));
                  ptc.weight[offset_e] = ptc.weight[offset_p] = 1.0f;
                  ptc.cell[offset_e] = ptc.cell[offset_p] = idx.linear;
                  ptc.flag[offset_e] = set_ptc_type_flag(
                      flag_or(PtcFlag::initial), PtcType::electron);
                  ptc.flag[offset_p] = set_ptc_type_flag(
                      flag_or(PtcFlag::initial), PtcType::positron);
                }
              } else {
                // Current sheet plasma, only one sign
                for (int i = 0; i < n_cs; i++) {
                  auto offset = atomic_add(ptc_pos, 2);
                  auto offset_e = offset;
                  auto offset_p = offset + 1;

                  ptc.x1[offset_e] = ptc.x1[offset_p] = rng.uniform<value_t>();
                  ptc.x2[offset_e] = ptc.x2[offset_p] = rng.uniform<value_t>();
                  ptc.x3[offset_e] = ptc.x3[offset_p] = rng.uniform<value_t>();

                  auto u_d = rng.maxwell_juttner_drifting(kT_cs, beta_d);
                  // ptc.p1[offset_e] = ptc.p1[offset_p] = u[2];
                  // ptc.p2[offset_e] = ptc.p2[offset_p] = u[1];
                  // ptc.p3[offset_e] = ptc.p3[offset_p] = u[0]; // z direction
                  // is the drift direction ptc.E[offset_e] = ptc.E[offset_p] =
                  //     math::sqrt(1.0f + u.dot(u));
                  ptc.p1[offset_e] = u_d[1];
                  ptc.p2[offset_e] = u_d[2];
                  ptc.p3[offset_e] = u_d[0];
                  ptc.E[offset_e] = math::sqrt(1.0f + u_d.dot(u_d));

                  u_d = rng.maxwell_juttner_drifting(kT_cs, beta_d);
                  ptc.p1[offset_p] = -u_d[1];
                  ptc.p2[offset_p] = -u_d[2];
                  ptc.p3[offset_p] = -u_d[0];
                  ptc.E[offset_p] = math::sqrt(1.0f + u_d.dot(u_d));

                  ptc.weight[offset_e] = ptc.weight[offset_p] = w;
                  ptc.cell[offset_e] = ptc.cell[offset_p] = idx.linear;
                  ptc.flag[offset_e] = set_ptc_type_flag(
                      flag_or(PtcFlag::initial, PtcFlag::exclude_from_spectrum),
                      PtcType::electron);
                  ptc.flag[offset_p] = set_ptc_type_flag(
                      flag_or(PtcFlag::initial, PtcFlag::exclude_from_spectrum),
                      PtcType::positron);
                }
              }
            },
            ptc, ptc_pos);
      },
      ptc, states, ptc_pos);
  policy::sync();
}

template void harris_current_sheet<Config<2>>(vector_field<Config<2>> &B,
                                              particle_data_t &ptc,
                                              rng_states_t &states, int mult);

}  // namespace Aperture
