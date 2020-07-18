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

#include "core/multi_array_exp.hpp"
#include "core/ndsubset_dev.hpp"
#include "core/math.hpp"
#include "data/curand_states.h"
#include "framework/config.h"
#include "ptc_injector_pulsar.h"

namespace Aperture {

template <typename Conf>
void compute_n_ptc(typename Conf::multi_array_t &n_ptc, particle_data_t &ptc,
                   const index_t<Conf::dim> &begin,
                   const extent_t<Conf::dim> &region_ext);

template <typename Conf, typename Func>
void
inject_pairs_along_B(const multi_array<int, Conf::dim> &num_per_cell,
                     const multi_array<int, Conf::dim> &cum_num_per_cell,
                     const typename Conf::multi_array_t &ptc_density,
                     particle_data_t &ptc, typename Conf::value_t weight,
                     typename Conf::value_t p0, curandState *states,
                     const Func *f) {
  using value_t = typename Conf::value_t;
  auto ptc_num = ptc.number();
  kernel_launch(
      [p0, ptc_num, weight, f] __device__(auto ptc, auto ptc_density,
                                      auto num_per_cell, auto cum_num,
                                      auto states) {
        auto &grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        cuda_rng_t rng(&states[id]);
        for (auto cell : grid_stride_range(0, ext.size())) {
          auto idx = typename Conf::idx_t(cell, ext);
          if (ptc_density[idx] > square(1.0f / grid.delta[0])) continue;
          auto pos = idx.get_pos();
          for (int i = 0; i < num_per_cell[cell]; i++) {
            int offset = ptc_num + cum_num[cell] * 2 + i * 2;
            ptc.x1[offset] = ptc.x1[offset + 1] = rng();
            ptc.x2[offset] = ptc.x2[offset + 1] = rng();
            ptc.x3[offset] = ptc.x3[offset + 1] = rng();

            value_t theta = grid.template pos<1>(pos[1], ptc.x2[offset]);
            value_t cth = math::cos(theta);
            value_t mag = math::sqrt(3.0f * cth * cth + 1.0f);
            ptc.p1[offset] = ptc.p1[offset + 1] = 2.0 * math::abs(cth) * p0 / mag;
            ptc.p2[offset] = ptc.p2[offset + 1] = sgn(cth) * math::sqrt(1.0f - cth * cth) * p0 / mag;
            ptc.p3[offset] = ptc.p3[offset + 1] = 0.0f;
            ptc.E[offset] = ptc.E[offset + 1] = math::sqrt(1.0f + p0 * p0);
            ptc.cell[offset] = ptc.cell[offset + 1] = cell;
            // ptc.weight[offset] = ptc.weight[offset + 1] = max(0.02,
            //     abs(2.0f * square(cos(th)) - square(sin(th))) * sin(th));
            // ptc.weight[offset] = ptc.weight[offset + 1] = f(x1, x2, x3);
            if (f == nullptr) {
              ptc.weight[offset] = ptc.weight[offset + 1] = weight;
            } else {
              Scalar x1 = grid.template pos<0>(pos[0], ptc.x1[offset]);
              Scalar x2 = grid.template pos<1>(pos[1], ptc.x2[offset]);
              Scalar x3 = grid.template pos<2>(pos[2], ptc.x3[offset]);
              ptc.weight[offset] = ptc.weight[offset + 1] =
                  weight * (*f)(x1, x2, x3);
            }
            // ptc.weight[offset] = ptc.weight[offset + 1] = 1.0f;
            ptc.flag[offset] = set_ptc_type_flag(0, PtcType::electron);
            ptc.flag[offset + 1] = set_ptc_type_flag(0, PtcType::positron);
          }
        }
      },
      ptc.get_dev_ptrs(), ptc_density.dev_ndptr_const(),
      num_per_cell.dev_ndptr_const(), cum_num_per_cell.dev_ndptr_const(),
      states);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
ptc_injector_pulsar<Conf>::update(double dt, uint32_t step) {
  // for (auto& inj : this->m_injectors) {
  for (int i = 0; i < this->m_injectors.size(); i++) {
    Logger::print_info("Working on {} of {} injectors", i,
                       this->m_injectors.size());
    auto &inj = this->m_injectors[i];
    if (step % inj.interval != 0) continue;
    this->m_num_per_cell.assign_dev(0);
    this->m_cum_num_per_cell.assign_dev(0);

    select_dev(this->m_num_per_cell, inj.begin, inj.ext) = inj.num;

    size_t grid_size = this->m_grid.extent().size();
    thrust::device_ptr<int> p_num_per_block(this->m_num_per_cell.dev_ptr());
    thrust::device_ptr<int> p_cum_num_per_block(
        this->m_cum_num_per_cell.dev_ptr());

    thrust::exclusive_scan(p_num_per_block, p_num_per_block + grid_size,
                           p_cum_num_per_block);
    CudaCheckError();
    this->m_num_per_cell.copy_to_host();
    this->m_cum_num_per_cell.copy_to_host();
    int new_pairs = 2 * (this->m_cum_num_per_cell[grid_size - 1] +
                         this->m_num_per_cell[grid_size - 1]);
    Logger::print_info("{} new pairs are injected in the box!", new_pairs);

    compute_n_ptc<Conf>(this->m_ptc_density, *(this->ptc), inj.begin, inj.ext);

    // Use the num_per_cell and cum_num info to inject actual pairs
    inject_pairs_along_B<Conf>(this->m_num_per_cell, this->m_cum_num_per_cell,
                               this->m_ptc_density, *(this->ptc), inj.weight,
                               10, this->m_rand_states->states(),
                               this->m_weight_funcs_dev[i]);
    this->ptc->add_num(new_pairs);
  }
}

template class ptc_injector_pulsar<Config<2>>;

}  // namespace Aperture
