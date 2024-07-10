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

template <typename Conf> void bh_injector<Conf>::register_data_components() {}

template <typename Conf> void bh_injector<Conf>::init() {
  sim_env().get_data("E", D);
  sim_env().get_data("B", B);
  sim_env().get_data("particles", ptc);
  sim_env().get_data("rng_states", m_rng_states);

  auto ext = m_grid.extent();
  m_num_per_cell.set_memtype(MemType::host_device);
  m_num_per_cell.resize(ext);
  m_num_per_cell.assign_dev(0);
  m_cum_num_per_cell.set_memtype(MemType::host_device);
  m_cum_num_per_cell.resize(ext);
  m_cum_num_per_cell.assign_dev(0);

  m_inj_thr = 1e-3;
  sim_env().params().get_value("inj_threshold", m_inj_thr);
  m_sigma_thr = 20.0f;
  sim_env().params().get_value("sigma_threshold", m_sigma_thr);
  sim_env().params().get_value("q_e", m_qe);

  int num_species = 2;
  sim_env().params().get_value("num_species", num_species);
  Rho.resize(num_species);
  for (int i = 0; i < num_species; i++) {
    sim_env().get_data(std::string("Rho_") + ptc_type_name(i), Rho[i]);
  }
  Rho.copy_to_device();
}

template <typename Conf>
void bh_injector<Conf>::update(double dt, uint32_t step) {
  value_t a = m_grid.a;
  value_t inj_thr = m_inj_thr;
  value_t sigma_thr = m_sigma_thr;
  value_t qe = m_qe;
  int num_species = Rho.size();

  // m_num_per_cell.assign_dev(0);
  m_num_per_cell.assign(0);

  using exec_policy = exec_policy_dynamic<Conf>;

  // Measure how many pairs to inject per cell
  exec_policy::launch(
      [a, inj_thr, sigma_thr, num_species] LAMBDA(
          auto B, auto D, auto rho, auto num_per_cell, auto states) {
        auto &grid = exec_policy::grid();
        auto ext = grid.extent();
        auto interp = lerp<Conf::dim>{};
        rng_t<typename exec_policy::exec_tag> rng(states);

        // for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext)))
        // {
        exec_policy::loop(Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
          auto pos = get_pos(idx, ext);
          // printf("idx.linear is %ld\n", idx.linear);

          if (grid.is_in_bound(pos)) {
            value_t r =
                grid_ks_t<Conf>::radius(grid.template coord<0>(pos[0], true));
            value_t th =
                grid_ks_t<Conf>::theta(grid.template coord<1>(pos[1], true));

            if (r <= 1.1f * Metric_KS::rH(a) || r > 3.5f ||
                th < grid.delta[1] || th > M_PI - grid.delta[1])
            // if (r <= 1.1f * Metric_KS::rH(a) || r > 3.5f)
              return;

            value_t D1 = interp(D[0], idx, stagger_t(0b110), stagger_t(0b111));
            value_t D2 = interp(D[1], idx, stagger_t(0b101), stagger_t(0b111));
            value_t D3 = interp(D[2], idx, stagger_t(0b011), stagger_t(0b111));
            value_t B1 = interp(B[0], idx, stagger_t(0b001), stagger_t(0b111));
            value_t B2 = interp(B[1], idx, stagger_t(0b010), stagger_t(0b111));
            value_t B3 = interp(B[2], idx, stagger_t(0b100), stagger_t(0b111));

            value_t sth = math::sin(th);
            value_t cth = math::cos(th);

            value_t DdotB = Metric_KS::dot_product_u({D1, D2, D3}, {B1, B2, B3},
                                                     a, r, sth, cth);
            value_t B_sqr = Metric_KS::dot_product_u({B1, B2, B3}, {B1, B2, B3},
                                                     a, r, sth, cth);

            value_t n = 1.0e-8f;
            for (int i = 0; i < num_species; i++) {
              n += math::abs(rho[i][idx]);
            }
            value_t sigma = B_sqr / n;

            auto u = rng.template uniform<float>();
            // printf("u is %f\n", u);
            if (sigma > sigma_thr && math::abs(DdotB) / B_sqr > inj_thr &&
                u < 0.001f) {
              // if (sigma > sigma_thr && math::abs(DdotB) / B_sqr > inj_thr) {
              num_per_cell[idx] = 1;
            } else {
              // printf("idx.linear is %ld\n", idx.linear);
              num_per_cell[idx] = 0;
            }
            // if (u < 0.01f) {
            //   num_per_cell[idx] = 1;
            // }
          }
        });
      },
      B, D, Rho, m_num_per_cell, m_rng_states);
  exec_policy::sync();

  auto num_per_cell = adapt(typename exec_policy::exec_tag{}, m_num_per_cell);
  // auto injector = ptc_injector_dynamic<Conf>(m_grid);
  ptc_injector_dynamic<Conf> injector(m_grid);
  injector.inject_pairs(
      // First function is the injection criterion for each cell. pos is an
      // index_t<Dim> object marking the cell in the grid. Returns true for
      // cells that inject and false for cells that do nothing.
      [] LAMBDA(auto &pos, auto &grid, auto &ext) {
        return true;
      },
      // Second function returns the number of particles injected in each cell.
      // This includes all species
      [num_per_cell] LAMBDA(auto &pos, auto &grid, auto &ext) {
        auto idx = Conf::idx(pos, ext);
        return num_per_cell[idx] * 2;
      },
      // Third function is the momentum distribution of the injected particles.
      // Returns a vec_t<value_t, 3> object encoding the 3D momentum of this
      // particular particle
      [] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        return vec_t<value_t, 3>(0.0, 0.0, 0.0);
      },
      // Fourth function is the particle weight, which can depend on the global
      // coordinate.
      [] LAMBDA(auto &x_global, PtcType type) {
        return math::sin(x_global[1]);
      });
}

template class bh_injector<Config<2>>;

} // namespace Aperture
