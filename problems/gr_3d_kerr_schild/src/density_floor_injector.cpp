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

#include "core/random.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "injector.h"
#include "systems/physics/metric_kerr_schild.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "utils/interpolation.hpp"
// #include <thrust/device_ptr.h>
// #include <thrust/scan.h>

namespace Aperture {

template <typename Conf>
void
bh_density_floor_injector<Conf>::register_data_components() {}

template <typename Conf>
void
bh_density_floor_injector<Conf>::init() {
  // sim_env().get_data("E", D);
  // sim_env().get_data("B", B);
  sim_env().get_data("particles", ptc);
  sim_env().get_data("rng_states", m_rng_states);
  // sim_env().get_data("DdotB", DdotB);
  // sim_env().get_data("Bmag", Bmag);

  auto ext = m_grid.extent();
  m_num_per_cell.set_memtype(MemType::host_device);
  m_num_per_cell.resize(ext);
  m_num_per_cell.assign_dev(0);
  // m_cum_num_per_cell.set_memtype(MemType::host_device);
  // m_cum_num_per_cell.resize(ext);
  // m_cum_num_per_cell.assign_dev(0);

  // This block calculates the density floor target value to enforce.
  m_sigma = 1e-1;
  sim_env().params().get_value("sigma", m_sigma);
  m_bp = 1.0f;
  sim_env().params().get_value("bp", m_bp);
  sim_env().params().get_value("q_e", m_qe);
  m_target_dens = m_bp * m_bp / m_sigma;

  // This block calculates the pml/damping layer starting radius.
  // Eventually, we will inject pairs within a certain distaince interior of
  // this radius.
  value_t nr = 1024;
  sim_env().params().get_value("Nr", nr);
  value_t size_log_r = 3.00;
  sim_env().params().get_value("size", size_log_r);
  value_t log_r_min = 0.588;
  sim_env().params().get_value("lower", log_r_min);
  value_t damping_length = 64;
  sim_env().params().get_value("damping_length", damping_length);
  value_t log_r_max = log_r_min + size_log_r;
  value_t d_log_r = size_log_r / nr;
  m_r_pml = math::exp(log_r_max - damping_length * d_log_r);

  // Get the temperature with which particles are injected.
  m_kT = 2.0 / m_r_pml;
  sim_env().params().get_value("kT", m_kT);

  m_ppc = 20;
  sim_env().params().get_value("ppc", m_ppc);

  int num_species = 2;
  sim_env().params().get_value("num_species", num_species);
  Rho.resize(num_species);
  for (int i = 0; i < num_species; i++) {
    sim_env().get_data(std::string("Rho_") + ptc_type_name(i), Rho[i]);
  }
  Rho.copy_to_device();

  pair_injected = sim_env().register_data<scalar_field<Conf>>(
      std::string("pair_injected"), m_grid, field_type::cell_centered,
      default_mem_type);
  pair_injected->reset_after_output(true);
}

template <typename Conf>
void
bh_density_floor_injector<Conf>::update(double dt, uint32_t step) {
  value_t a = m_grid.a;
  // value_t inj_thr = m_inj_thr;
  // value_t sigma_thr = m_sigma_thr;
  value_t target_dens = m_target_dens;
  value_t kT = m_kT;
  value_t r_pml = m_r_pml;
  value_t qe = m_qe;
  value_t ppc = m_ppc;
  int num_species = Rho.size();

  auto time = sim_env().get_time();
  if (time < 10.0) {
    return;
  }

  // m_num_per_cell.assign_dev(0);
  m_num_per_cell.assign(0);

  using exec_policy = exec_policy_dynamic<Conf>;

  // Measure how many pairs to inject per cell
  exec_policy::launch(
      [a, target_dens, r_pml, num_species] LAMBDA(
          auto rho, auto num_per_cell, auto states, auto pair_injected) {
        auto &grid = exec_policy::grid();
        auto ext = grid.extent();
        // auto interp = lerp<Conf::dim>{};
        rng_t<typename exec_policy::exec_tag> rng(states);

        // for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext)))
        // {
        exec_policy::loop(
            Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
              auto pos = get_pos(idx, ext);
              // printf("idx.linear is %ld\n", idx.linear);

              if (grid.is_in_bound(pos)) {
                value_t r = grid_ks_t<Conf>::radius(
                    grid.template coord<0>(pos[0], true));
                value_t th = grid_ks_t<Conf>::theta(
                    grid.template coord<1>(pos[1], true));

                if (r <= 1.02f * Metric_KS::rH(a) || r > 3.5f ||
                    th < grid.delta[1] || th > M_PI - grid.delta[1])
                  // if (r <= 1.1f * Metric_KS::rH(a) || r > 3.5f)
                  return;

                value_t n = 0.0f;
                for (int sp = 0; sp < num_species; sp++) {
                  n += math::abs(rho[sp][idx]);
                }

                auto u = rng.template uniform<float>();
                // // printf("u is %f\n", u);
                // if (pos[1] == 512 && pos[0] == 60) {
                // printf("sigma is %f, sigma_thr is %f, n is %f, B_sqr is
                // %f\n", sigma, sigma_thr, n, B_sqr);
                // }
                if (n < target_dens && r < r_pml && r > r_pml - 1.0) {
                  num_per_cell[idx] = 1;
                  pair_injected[idx] += 2.0;
                } else {
                  // printf("idx.linear is %ld\n", idx.linear);
                  num_per_cell[idx] = 0;
                }
                // if (u < 0.1f) {
                //   num_per_cell[idx] = 1;
                // }
              }
            });
      },
      Rho, m_num_per_cell, m_rng_states, pair_injected);
  exec_policy::sync();

  auto num_per_cell = adapt(typename exec_policy::exec_tag{}, m_num_per_cell);
  // auto ddotb_ptr = adapt(typename exec_policy::exec_tag{}, DdotB);
  // auto bmag = adapt(typename exec_policy::exec_tag{}, Bmag);
  // auto injector = ptc_injector_dynamic<Conf>(m_grid);
  if (step * dt > 1.0) {
    ptc_injector_dynamic<Conf> injector(m_grid);
    injector.inject_pairs(
        // First function is the injection criterion for each cell. pos is an
        // index_t<Dim> object marking the cell in the grid. Returns true for
        // cells that inject and false for cells that do nothing.
        [num_per_cell] LAMBDA(auto &pos, auto &grid, auto &ext) {
          auto idx = Conf::idx(pos, ext);
          return num_per_cell[idx] > 0;
        },
        // Second function returns the number of particles injected in each
        // cell. This includes all species
        [num_per_cell] LAMBDA(auto &pos, auto &grid, auto &ext) {
          auto idx = Conf::idx(pos, ext);
          return num_per_cell[idx] * 2;
        },
        // Third function is the momentum distribution of the injected
        // particles. Returns a vec_t<value_t, 3> object encoding the 3D
        // momentum of this particular particle
        [a, kT] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
          value_t r = grid_ks_t<Conf>::radius(x_global[0]);
          value_t theta = grid_ks_t<Conf>::theta(x_global[1]);

          vec_t<value_t, 3> u_d = rng_maxwell_juttner_3d(state, kT);
          // Now transform this momentum from the local fluid frame to the
          // global coordinate. Use the tetrads given in Benjamin Crinquand's
          // PhD thesis:
          // https://theses.hal.science/tel-03406333v1/file/Thesis_Benjamin_Crinquand_final.pdf
          // The relevant expressions are given on pages 164-165
          // Note also that in the locally flat fluid rest frame, u^i = u_i
          value_t h_11 = Metric_KS::g_11(r, theta, a);
          value_t h_22 = Metric_KS::g_22(r, theta, a);
          value_t h_33 = Metric_KS::g_33(r, theta, a);
          value_t h_13 = Metric_KS::g_13(r, theta, a);
          value_t scriptA = math::sqrt(h_33 / (h_11 * h_33 - h_13 * h_13));

          value_t u_3 = math::sqrt(h_33) * u_d[2];
          value_t u_2 = math::sqrt(h_22) * u_d[1];
          value_t u_1 = u_d[0] / scriptA + (h_13 / h_33) * u_3;

          return vec_t<Scalar, 3>{u_1, u_2, u_3};
        },
        // Fourth function is the particle weight, which can depend on the
        // global coordinate.
        [a, target_dens, ppc] LAMBDA(auto &x_global, PtcType type) {
          index_t<Conf::dim> pos;
          vec_t<value_t, 3> x;
          // auto &grid = exec_policy::grid();
          // grid.from_global(x_global, pos, x);
          // auto idx = Conf::idx(pos, grid.extent());
          value_t r = grid_ks_t<Conf>::radius(x_global[0]);
          value_t th = grid_ks_t<Conf>::theta(x_global[1]);
          value_t sqrt_gamma = Metric_KS::sqrt_gamma(a, r, th);
          value_t w = (target_dens * r * sqrt_gamma) / ppc;
          return w;
        });
  }
}

template class bh_density_floor_injector<Config<3>>;

}  // namespace Aperture
