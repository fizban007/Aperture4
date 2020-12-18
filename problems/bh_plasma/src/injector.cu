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
#include "injector.h"
#include "systems/physics/metric_kerr_schild.hpp"
#include "utils/interpolation.hpp"
#include "utils/kernel_helper.hpp"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace Aperture {

template <typename Conf>
void
bh_injector<Conf>::register_data_components() {}

template <typename Conf>
void
bh_injector<Conf>::init() {
  m_env.get_data("E", &D);
  m_env.get_data("B", &B);
  m_env.get_data("particles", &ptc);

  auto ext = m_grid.extent();
  m_num_per_cell.resize(ext);
  m_num_per_cell.assign_dev(0);
  m_cum_num_per_cell.resize(ext);
  m_cum_num_per_cell.assign_dev(0);

  m_inj_thr = 1e-3;
  m_env.params().get_value("inj_threshold", m_inj_thr);
  m_sigma_thr = 20.0f;
  m_env.params().get_value("sigma_threshold", m_sigma_thr);

  int num_species = 2;
  this->m_env.params().get_value("num_species", num_species);
  Rho.resize(num_species);
  for (int i = 0; i < num_species; i++) {
    this->m_env.get_data(std::string("Rho_") + ptc_type_name(i), &Rho[i]);
  }
  m_rho_ptrs.set_memtype(MemType::host_device);
  m_rho_ptrs.resize(num_species);
  for (int i = 0; i < num_species; i++) {
    m_rho_ptrs[i] = Rho[i]->dev_ndptr();
  }
  m_rho_ptrs.copy_to_device();
}

template <typename Conf>
void
bh_injector<Conf>::update(double dt, uint32_t step) {
  value_t a = m_grid.a;
  value_t inj_thr = m_inj_thr;
  value_t sigma_thr = m_sigma_thr;
  int num_species = m_rho_ptrs.size();

  // Measure how many pairs to inject per cell
  kernel_launch(
      [a, inj_thr, sigma_thr, num_species] __device__(auto B, auto D, auto rho,
                                                      auto num_per_cell) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        auto interp = lerp<Conf::dim>{};
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);

          if (grid.is_in_bound(pos)) {
            value_t D1 = interp(D[0], idx, stagger_t(0b110), stagger_t(0b111));
            value_t D2 = interp(D[1], idx, stagger_t(0b101), stagger_t(0b111));
            value_t D3 = interp(D[2], idx, stagger_t(0b011), stagger_t(0b111));
            value_t B1 = interp(B[0], idx, stagger_t(0b001), stagger_t(0b111));
            value_t B2 = interp(B[1], idx, stagger_t(0b010), stagger_t(0b111));
            value_t B3 = interp(B[2], idx, stagger_t(0b100), stagger_t(0b111));

            value_t r =
                grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], true));
            value_t th =
                grid_ks_t<Conf>::theta(grid.template pos<1>(pos[1], true));
            value_t sth = math::sin(th);
            value_t cth = math::cos(th);

            value_t DdotB = Metric_KS::dot_product_u({D1, D2, D3}, {B1, B2, B3},
                                                     a, r, sth, cth);
            value_t B_sqr = Metric_KS::dot_product_u({B1, B2, B3}, {B1, B2, B3},
                                                     a, r, sth, cth);

            value_t n = 0.0f;
            for (int i = 0; i < num_species; i++) {
              n += math::abs(rho[i][idx]);
            }
            value_t sigma = B_sqr / n;

            if (sigma > sigma_thr && DdotB / B_sqr > inj_thr) {
              num_per_cell[idx] = 1;
            } else {
              num_per_cell[idx] = 0;
            }
          }
        }
      },
      B->get_ptrs(), D->get_ptrs(), m_rho_ptrs.dev_ptr(),
      m_num_per_cell.dev_ndptr());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  size_t grid_size = m_grid.extent().size();
  thrust::device_ptr<int> p_num_per_block(m_num_per_cell.dev_ptr());
  thrust::device_ptr<int> p_cum_num_per_block(m_cum_num_per_cell.dev_ptr());

  thrust::exclusive_scan(p_num_per_block, p_num_per_block + grid_size,
                         p_cum_num_per_block);
  CudaCheckError();
  m_num_per_cell.copy_to_host();
  m_cum_num_per_cell.copy_to_host();
  int new_pairs =
      2 * (m_cum_num_per_cell[grid_size - 1] + m_num_per_cell[grid_size - 1]);
  Logger::print_info("{} new pairs are injected in the box!", new_pairs);

  auto ptc_num = ptc->number();
  kernel_launch(
      [ptc_num] __device__(auto ptc, auto num_per_cell, auto cum_num) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto cell : grid_stride_range(0, ext.size())) {
          auto idx = typename Conf::idx_t(cell, ext);
          auto pos = get_pos(idx, ext);
          for (int i = 0; i < num_per_cell[cell]; i++) {
            int offset = ptc_num + cum_num[cell] * 2 + i * 2;
            ptc.x1[offset] = ptc.x1[offset + 1] = 0.5f;
            ptc.x2[offset] = ptc.x2[offset + 1] = 0.5f;
            ptc.x3[offset] = ptc.x3[offset + 1] = 0.0f;
            ptc.p1[offset] = ptc.p1[offset + 1] = 0.0f;
            ptc.p2[offset] = ptc.p2[offset + 1] = 0.0f;
            ptc.p3[offset] = ptc.p3[offset + 1] = 0.0f;
            ptc.E[offset] = ptc.E[offset + 1] = 1.0f;
            ptc.cell[offset] = ptc.cell[offset + 1] = cell;
            Scalar th = grid.template pos<1>(pos[1], ptc.x2[offset]);
            // ptc.weight[offset] = ptc.weight[offset + 1] = max(0.02,
            //     abs(2.0f * square(cos(th)) - square(sin(th))) * sin(th));
            ptc.weight[offset] = ptc.weight[offset + 1] = sin(th);
            // ptc.weight[offset] = ptc.weight[offset + 1] = 1.0f;
            ptc.flag[offset] = set_ptc_type_flag(0, PtcType::electron);
            ptc.flag[offset + 1] = set_ptc_type_flag(0, PtcType::positron);
          }
        }
      },
      ptc->get_dev_ptrs(), m_num_per_cell.dev_ndptr_const(),
      m_cum_num_per_cell.dev_ndptr_const());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  ptc->add_num(new_pairs);
}

template class bh_injector<Config<2, float>>;
template class bh_injector<Config<2, double>>;

}  // namespace Aperture
