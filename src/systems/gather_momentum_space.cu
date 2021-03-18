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

#include "framework/config.h"
#include "framework/environment.h"
#include "framework/params_store.h"
#include "gather_momentum_space.h"
#include "utils/kernel_helper.hpp"

namespace Aperture {

template <typename Conf>
void
gather_momentum_space_cu<Conf>::update(double dt, uint32_t step) {
  if (step % this->m_data_interval != 0) return;
  this->momentum->init();
  // Convert these into things that can be passed onto the gpu
  vec_t<int, 3> num_bins(this->momentum->m_num_bins);
  vec_t<float, 3> lower(this->momentum->m_lower);
  vec_t<float, 3> upper(this->momentum->m_upper);

  // Loop over the particle array to gather momentum space information
  auto num = this->ptc->number();

  Logger::print_info("gathering particle momentum space");
  kernel_launch(
      [num, num_bins, lower, upper] __device__(auto ptc, auto e_p1, auto e_p2,
                                               auto e_p3, auto p_p1, auto p_p2,
                                               auto p_p3, int downsample) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        auto ext_out = grid.extent_less() / downsample;
        using idx_t = default_idx_t<Conf::dim + 1>;
        for (auto n : grid_stride_range(0, num)) {
          uint32_t cell = ptc.cell[n];
          if (cell == empty_cell) continue;

          auto idx = Conf::idx(cell, ext);
          // auto pos = idx.get_pos();
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            index_t<Conf::dim + 1> pos_out(0, (pos - grid.guards()) / downsample);

            auto weight = ptc.weight[n];
            auto flag = ptc.flag[n];
            if (check_flag(flag, PtcFlag::exclude_from_spectrum))
              continue;
            auto sp = get_ptc_type(flag);

            auto p1 = clamp(ptc.p1[n], lower[0], upper[0]);
            auto p2 = clamp(ptc.p2[n], lower[1], upper[1]);
            auto p3 = clamp(ptc.p3[n], lower[2], upper[2]);

            int bin1 = floor((p1 - lower[0]) / (upper[0] - lower[0]) * (num_bins[0] - 1));
            int bin2 = floor((p2 - lower[1]) / (upper[1] - lower[1]) * (num_bins[1] - 1));
            int bin3 = floor((p3 - lower[2]) / (upper[2] - lower[2]) * (num_bins[2] - 1));

            if (sp == (int)PtcType::electron) {
              pos_out[0] = bin1;
              atomicAdd(&e_p1[idx_t(pos_out, extent_t<Conf::dim + 1>(num_bins[0], ext_out))],
                        weight);
              pos_out[0] = bin2;
              atomicAdd(&e_p2[idx_t(pos_out, extent_t<Conf::dim + 1>(num_bins[1], ext_out))],
                        weight);
              pos_out[0] = bin3;
              atomicAdd(&e_p3[idx_t(pos_out, extent_t<Conf::dim + 1>(num_bins[2], ext_out))],
                        weight);
            } else if (sp == (int)PtcType::positron) {
              pos_out[0] = bin1;
              atomicAdd(&p_p1[idx_t(pos_out, extent_t<Conf::dim + 1>(num_bins[0], ext_out))],
                        weight);
              pos_out[0] = bin2;
              atomicAdd(&p_p2[idx_t(pos_out, extent_t<Conf::dim + 1>(num_bins[1], ext_out))],
                        weight);
              pos_out[0] = bin3;
              atomicAdd(&p_p3[idx_t(pos_out, extent_t<Conf::dim + 1>(num_bins[2], ext_out))],
                        weight);
            }
          }
        }
      },
      this->ptc->dev_ptrs(), this->momentum->e_p1.dev_ndptr(),
      this->momentum->e_p2.dev_ndptr(), this->momentum->e_p3.dev_ndptr(),
      this->momentum->p_p1.dev_ndptr(), this->momentum->p_p2.dev_ndptr(),
      this->momentum->p_p3.dev_ndptr(), this->momentum->m_downsample);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

INSTANTIATE_WITH_CONFIG(gather_momentum_space_cu);

}  // namespace Aperture
