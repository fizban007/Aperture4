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

#ifndef _GATHER_MOMENTUM_SPACE_IMPL_H_
#define _GATHER_MOMENTUM_SPACE_IMPL_H_

#include "core/math.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "framework/params_store.h"
#include "gather_momentum_space.h"
#include "utils/interpolation.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
void gather_momentum_space<Conf, ExecPolicy>::register_data_components() {
  // int downsample =
  //     sim_env().params().template get_as<int64_t>("momentum_downsample", 16);
  sim_env().params().get_value("momentum_downsample", m_downsample);
  sim_env().params().get_array("momentum_num_bins", m_num_bins);
  sim_env().params().get_array("momentum_lower", m_lim_lower);
  sim_env().params().get_array("momentum_upper", m_lim_upper);
  sim_env().params().get_value("momentum_use_log_scale", m_use_log_scale);
  sim_env().params().get_value("pitch_angle_bins", m_pitch_bins);

  if (m_use_log_scale) {
    for (int i = 0; i < 3; i++) {
      m_lim_lower[i] = symlog(m_lim_lower[i]);
      m_lim_upper[i] = symlog(m_lim_upper[i]);
    }
  }
  // Energy doesn't need symlog
  m_lim_lower[3] = math::log(m_lim_lower[3]);
  m_lim_upper[3] = math::log(m_lim_upper[3]);

  sim_env().params().get_value("fld_output_interval", m_data_interval);

  // momentum = sim_env().register_data<momentum_space<Conf>>(
  //     "momentum", m_grid, downsample, num_bins, lim_lower, lim_upper,
  //     use_log_scale, ExecPolicy<Conf>::data_mem_type());
  sim_env().params().get_value("num_species", m_num_species);
  if (m_num_species > max_ptc_types) {
    Logger::print_err("Too many species of particles requested! Aborting");
    throw std::runtime_error("too many species");
  }
  momenta.resize(m_num_species);
  energies.resize(m_num_species);
  int energy_bins[2] = {m_num_bins[3], m_pitch_bins};
  float energy_lim_lower[2] = {m_lim_lower[3], -1.0f};
  float energy_lim_upper[2] = {m_lim_upper[3], 1.0f};

  for (int i = 0; i < m_num_species; i++) {
    momenta.set(i, sim_env().register_data<phase_space<Conf, 3>>(
                       std::string("momentum_") + ptc_type_name(i), m_grid,
                       m_downsample, &m_num_bins[0], &m_lim_lower[0],
                       &m_lim_upper[0], m_use_log_scale,
                       ExecPolicy<Conf>::data_mem_type()));
    energies.set(i, sim_env().register_data<phase_space<Conf, 2>>(
                        std::string("energy_") + ptc_type_name(i), m_grid,
                        m_downsample, &energy_bins[0], &energy_lim_lower[0],
                        &energy_lim_upper[0], m_use_log_scale,
                        ExecPolicy<Conf>::data_mem_type()));
    momenta[i]->reset_after_output(true);
    energies[i]->reset_after_output(true);
  }
  momenta.copy_to_device();
  energies.copy_to_device();
}

template <typename Conf, template <class> class ExecPolicy>
void gather_momentum_space<Conf, ExecPolicy>::init() {
  sim_env().get_data("particles", ptc);
}

template <typename Conf, template <class> class ExecPolicy>
void gather_momentum_space<Conf, ExecPolicy>::update(double dt, uint32_t step) {
  if (step % m_data_interval != 0)
    return;
  momenta.init();
  // Convert these into things that can be passed onto the gpu
  vec_t<int, 4> num_bins(m_num_bins);
  vec_t<float, 4> lower(m_lim_lower);
  vec_t<float, 4> upper(m_lim_upper);
  bool log_scale = m_use_log_scale;
  int pitch_bins = m_pitch_bins;

  // Loop over the particle array to gather momentum space information
  auto num = ptc->number();

  Logger::print_detail("gathering particle momentum space");
  ExecPolicy<Conf>::launch(
      [num, num_bins, pitch_bins, lower, upper,
       log_scale] LAMBDA(auto ptc, auto B, auto e_p, auto e_E, auto p_p,
                         auto p_E, int downsample) {
        auto &grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        auto ext_out = grid.extent_less() / downsample;
        using idx_p_t = default_idx_t<Conf::dim + 3>;
        using idx_E_t = default_idx_t<Conf::dim + 2>;
        // for (auto n : grid_stride_range(0, num)) {
        ExecPolicy<Conf>::loop(0, num, [&] LAMBDA(auto n) {
          uint32_t cell = ptc.cell[n];
          if (cell == empty_cell)
            return;

          // idx and pos of the particle in the main grid
          auto idx = Conf::idx(cell, ext);
          auto pos = get_pos(idx, ext);

          if (grid.is_in_bound(pos)) {
            // pos for momentum space
            index_t<Conf::dim + 3> pos_p(0, 0, 0,
                                         (pos - grid.guards()) / downsample);
            // pos for energy space
            index_t<Conf::dim + 2> pos_E(0, 0,
                                         (pos - grid.guards()) / downsample);

            auto weight = ptc.weight[n];
            auto flag = ptc.flag[n];
            if (check_flag(flag, PtcFlag::exclude_from_spectrum))
              return;
            auto sp = get_ptc_type(flag);

            // auto p1 = clamp(ptc.p1[n], lower[0], upper[0]);
            // auto p2 = clamp(ptc.p2[n], lower[1], upper[1]);
            // auto p3 = clamp(ptc.p3[n], lower[2], upper[2]);
            // auto E = clamp(ptc.E[n], lower[3], upper[3]);
            auto p1 = ptc.p1[n];
            auto p2 = ptc.p2[n];
            auto p3 = ptc.p3[n];

            auto interp = interp_t<1, Conf::dim>{};
            vec_t<float, 3> x(ptc.x1[n], ptc.x2[n], ptc.x3[n]);
            float B1 = interp(x, B[0], idx, ext, stagger_t(0b001));
            float B2 = interp(x, B[1], idx, ext, stagger_t(0b010));
            float B3 = interp(x, B[2], idx, ext, stagger_t(0b100));
            auto p = math::sqrt(p1 * p1 + p2 * p2 + p3 * p3);
            float mu = (B1 * p1 + B2 * p2 + B3 * p3) /
                       (p * math::sqrt(B1 * B1 + B2 * B2 + B3 * B3));

            p1 = (log_scale ? symlog(p1) : p1);
            p2 = (log_scale ? symlog(p2) : p2);
            p3 = (log_scale ? symlog(p3) : p3);
            auto E = (log_scale ? math::log(ptc.E[n] - 1.0f) : ptc.E[n] - 1.0f);

            p1 = clamp(p1, lower[0], upper[0]);
            p2 = clamp(p2, lower[1], upper[1]);
            p3 = clamp(p3, lower[2], upper[2]);
            E = clamp(E, lower[3], upper[3]);

            int bin1 = floor((p1 - lower[0]) / (upper[0] - lower[0]) *
                             (num_bins[0] - 1));
            int bin2 = floor((p2 - lower[1]) / (upper[1] - lower[1]) *
                             (num_bins[1] - 1));
            int bin3 = floor((p3 - lower[2]) / (upper[2] - lower[2]) *
                             (num_bins[2] - 1));
            int bin4 = floor((E - lower[3]) / (upper[3] - lower[3]) *
                             (num_bins[3] - 1));
            int bin_mu = floor((mu + 1.0f) / 2.0f * (pitch_bins - 1));

            if (sp == (int)PtcType::electron) {
              pos_p[0] = bin1;
              pos_p[1] = bin2;
              pos_p[2] = bin3;
              atomic_add(&e_p[idx_p_t(pos_p, extent_t<Conf::dim + 3>(
                                                 num_bins[0], num_bins[1],
                                                 num_bins[2], ext_out))],
                         weight);
              pos_E[0] = bin4;
              pos_E[1] = bin_mu;
              atomic_add(
                  &e_E[idx_E_t(pos_E, extent_t<Conf::dim + 2>(
                                          num_bins[3], pitch_bins, ext_out))],
                  weight);
            } else if (sp == (int)PtcType::positron) {
              pos_p[0] = bin1;
              pos_p[1] = bin2;
              pos_p[2] = bin3;
              atomic_add(&p_p[idx_p_t(pos_p, extent_t<Conf::dim + 3>(
                                                 num_bins[0], num_bins[1],
                                                 num_bins[2], ext_out))],
                         weight);
              pos_E[0] = bin4;
              pos_E[1] = bin_mu;
              atomic_add(
                  &p_E[idx_E_t(pos_E, extent_t<Conf::dim + 2>(
                                          num_bins[3], pitch_bins, ext_out))],
                  weight);
            }
          }
        });
        // ptc, e_p1, e_p2, e_p3, e_E, p_p1, p_p2, p_p3, p_E);
      },
      ptc, B, momenta[0], energies[0], momenta[1], energies[1], m_downsample);
  ExecPolicy<Conf>::sync();
}

} // namespace Aperture

#endif // _GATHER_MOMENTUM_SPACE_IMPL_H_
