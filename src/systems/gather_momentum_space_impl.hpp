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
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
void
gather_momentum_space<Conf, ExecPolicy>::register_data_components() {
  int downsample =
      sim_env().params().template get_as<int64_t>("momentum_downsample", 16);
  int num_bins[4] = {256, 256, 256, 256};
  sim_env().params().get_array("momentum_num_bins", num_bins);
  float lim_lower[4] = {-1.0, -1.0, -1.0, 1.0};
  sim_env().params().get_array("momentum_lower", lim_lower);
  float lim_upper[4] = {1.0, 1.0, 1.0, 100.0};
  sim_env().params().get_array("momentum_upper", lim_upper);
  bool use_log_scale = false;
  sim_env().params().get_value("momentum_use_log_scale", use_log_scale);

  if (use_log_scale) {
    for (int i = 0; i < 3; i++) {
      lim_lower[i] = symlog(lim_lower[i]);
      lim_upper[i] = symlog(lim_upper[i]);
    }
    // Energy doesn't need symlog
    lim_lower[3] = math::log(lim_lower[3]);
    lim_upper[3] = math::log(lim_upper[3]);
  }

  sim_env().params().get_value("fld_output_interval", m_data_interval);

  momentum = sim_env().register_data<momentum_space<Conf>>(
      "momentum", m_grid, downsample, num_bins, lim_lower, lim_upper,
      use_log_scale, ExecPolicy<Conf>::data_mem_type());
}

template <typename Conf, template <class> class ExecPolicy>
void
gather_momentum_space<Conf, ExecPolicy>::init() {
  sim_env().get_data("particles", ptc);
}

template <typename Conf, template <class> class ExecPolicy>
void
gather_momentum_space<Conf, ExecPolicy>::update(double dt, uint32_t step) {
  if (step % m_data_interval != 0) return;
  momentum->init();
  // Convert these into things that can be passed onto the gpu
  vec_t<int, 4> num_bins(momentum->m_num_bins);
  vec_t<float, 4> lower(momentum->m_lower);
  vec_t<float, 4> upper(momentum->m_upper);
  bool log_scale = momentum->m_log_scale;

  // Loop over the particle array to gather momentum space information
  auto num = ptc->number();

  Logger::print_detail("gathering particle momentum space");
  ExecPolicy<Conf>::launch(
      [num, num_bins, lower, upper, log_scale] LAMBDA(
          auto ptc, auto e_p1, auto e_p2, auto e_p3, auto e_E, auto p_p1,
          auto p_p2, auto p_p3, auto p_E, int downsample) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        auto ext_out = grid.extent_less() / downsample;
        using idx_t = default_idx_t<Conf::dim + 1>;
        // for (auto n : grid_stride_range(0, num)) {
        ExecPolicy<Conf>::loop(0, num, [&] LAMBDA(auto n) {
          uint32_t cell = ptc.cell[n];
          if (cell == empty_cell) return;

          auto idx = Conf::idx(cell, ext);
          // auto pos = idx.get_pos();
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            index_t<Conf::dim + 1> pos_out(0,
                                           (pos - grid.guards()) / downsample);

            auto weight = ptc.weight[n];
            auto flag = ptc.flag[n];
            if (check_flag(flag, PtcFlag::exclude_from_spectrum)) return;
            auto sp = get_ptc_type(flag);

            // auto p1 = clamp(ptc.p1[n], lower[0], upper[0]);
            // auto p2 = clamp(ptc.p2[n], lower[1], upper[1]);
            // auto p3 = clamp(ptc.p3[n], lower[2], upper[2]);
            // auto E = clamp(ptc.E[n], lower[3], upper[3]);
            auto p1 = (log_scale ? symlog(ptc.p1[n]) : ptc.p1[n]);
            auto p2 = (log_scale ? symlog(ptc.p2[n]) : ptc.p2[n]);
            auto p3 = (log_scale ? symlog(ptc.p3[n]) : ptc.p3[n]);
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

            if (sp == (int)PtcType::electron) {
              pos_out[0] = bin1;
              atomic_add(&e_p1[idx_t(pos_out, extent_t<Conf::dim + 1>(
                                                  num_bins[0], ext_out))],
                         weight);
              pos_out[0] = bin2;
              atomic_add(&e_p2[idx_t(pos_out, extent_t<Conf::dim + 1>(
                                                  num_bins[1], ext_out))],
                         weight);
              pos_out[0] = bin3;
              atomic_add(&e_p3[idx_t(pos_out, extent_t<Conf::dim + 1>(
                                                  num_bins[2], ext_out))],
                         weight);
              pos_out[0] = bin4;
              atomic_add(&e_E[idx_t(pos_out, extent_t<Conf::dim + 1>(
                                                 num_bins[3], ext_out))],
                         weight);
            } else if (sp == (int)PtcType::positron) {
              pos_out[0] = bin1;
              atomic_add(&p_p1[idx_t(pos_out, extent_t<Conf::dim + 1>(
                                                  num_bins[0], ext_out))],
                         weight);
              pos_out[0] = bin2;
              atomic_add(&p_p2[idx_t(pos_out, extent_t<Conf::dim + 1>(
                                                  num_bins[1], ext_out))],
                         weight);
              pos_out[0] = bin3;
              atomic_add(&p_p3[idx_t(pos_out, extent_t<Conf::dim + 1>(
                                                  num_bins[2], ext_out))],
                         weight);
              pos_out[0] = bin4;
              atomic_add(&p_E[idx_t(pos_out, extent_t<Conf::dim + 1>(
                                                 num_bins[3], ext_out))],
                         weight);
            }
          }
        });
        // ptc, e_p1, e_p2, e_p3, e_E, p_p1, p_p2, p_p3, p_E);
      },
      ptc, momentum->e_p1, momentum->e_p2, momentum->e_p3, momentum->e_E,
      momentum->p_p1, momentum->p_p2, momentum->p_p3, momentum->p_E,
      momentum->m_downsample);
  ExecPolicy<Conf>::sync();
}

}  // namespace Aperture

#endif  // _GATHER_MOMENTUM_SPACE_IMPL_H_
