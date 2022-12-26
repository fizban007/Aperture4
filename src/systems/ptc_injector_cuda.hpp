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

#ifndef _PTC_INJECTOR_CUDA_H_
#define _PTC_INJECTOR_CUDA_H_

#include "core/cached_allocator.hpp"
#include "core/multi_array.hpp"
#include "data/fields.h"
#include "data/particle_data.h"
#include "data/rng_states.h"
#include "framework/system.h"
#include "systems/grid.h"
#include "systems/policies/exec_policy_cuda.hpp"
#include "systems/ptc_injector_new.h"
#include "utils/range.hpp"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace Aperture {

template <typename Conf>
class ptc_injector<Conf, exec_policy_cuda> : public system_t {
 public:
  using value_t = typename Conf::value_t;
  static std::string name() { return "ptc_injector"; }

  ptc_injector(const Grid<Conf::dim, value_t>& grid) : m_grid(grid) {
    auto ext = grid.extent();
    m_num_per_cell.resize(ext);
    m_cum_num_per_cell.resize(ext);
  }

  ~ptc_injector() {}

  void init() override {
    sim_env().get_data("particles", ptc);
    sim_env().get_data("rng_states", rng_states);
  }

  template <typename FCriteria, typename FDist, typename FNumPerCell,
            typename FWeight>
  void inject(const FCriteria& f_criteria, const FNumPerCell& f_num,
              const FDist& f_dist, const FWeight& f_weight, uint32_t flag = 0) {
    using policy = exec_policy_cuda<Conf>;
    m_num_per_cell.assign_dev(0);
    m_cum_num_per_cell.assign_dev(0);
    Logger::print_detail_all("Before calculating num_per_cell");

    // First compute the number of particles per cell
    policy::launch(
        [] __device__(auto num_per_cell, auto f_criteria, auto f_num) {
          auto& grid = policy::grid();
          auto ext = grid.extent();

          for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
            auto pos = get_pos(idx, ext);
            if (grid.is_in_bound(pos)) {
              if (f_criteria(pos, grid, ext)) {
                num_per_cell[idx] = f_num(pos, grid, ext);
              }
            }
          }
        },
        m_num_per_cell, f_criteria, f_num);
    policy::sync();
    Logger::print_debug("Finished calculating num_per_cell");

    // Compute cumulative number per cell
    thrust::device_ptr<int> p_num_per_cell(m_num_per_cell.dev_ptr());
    thrust::device_ptr<int> p_cum_num_per_cell(m_cum_num_per_cell.dev_ptr());

    thrust::exclusive_scan(p_num_per_cell, p_num_per_cell + m_grid.size(),
                           p_cum_num_per_cell);
    CudaCheckError();
    m_num_per_cell.copy_to_host();
    m_cum_num_per_cell.copy_to_host();
    int new_particles = (m_cum_num_per_cell[m_grid.size() - 1] +
                         m_num_per_cell[m_grid.size() - 1]);
    auto num = ptc->number();
    auto max_num = ptc->size();
    Logger::print_debug("Current num is {}, injecting {}, max_num is {}", num,
      new_particles, max_num);
    // Logger::print_info("Injecting {}", new_particles);

    // Actually create the particles
    policy::launch(
        // kernel_exec_policy(rng_states_t::block_num, rng_states_t::thread_num),
        [flag, num, max_num] __device__(ptc_ptrs ptc, auto states, auto num_per_cell,
                               auto cum_num_per_cell, auto f_dist, auto f_weight) {
          auto& grid = policy::grid();
          auto ext = grid.extent();
          rng_t<exec_tags::device> rng(states);

          for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
            auto pos = get_pos(idx, ext);
            // uint32_t idx_linear = static_cast<uint32_t>(idx.linear);
            if (grid.is_in_bound(pos)) {
              for (int n = 0; n < num_per_cell[idx]; n += 2) {
                uint32_t offset_e = num + cum_num_per_cell[idx] + n;
                uint32_t offset_p = offset_e + 1;
                if (offset_e >= max_num || offset_p >= max_num) {
                  break;
                }

                ptc.cell[offset_e] = (uint32_t)idx.linear;
                ptc.cell[offset_p] = (uint32_t)idx.linear;
                auto x = vec_t<value_t, 3>(rng.uniform<value_t>(),
                                           rng.uniform<value_t>(),
                                           rng.uniform<value_t>());
                ptc.x1[offset_e] = x[0];
                ptc.x1[offset_p] = x[0];
                ptc.x2[offset_e] = x[1];
                ptc.x2[offset_p] = x[1];
                ptc.x3[offset_e] = x[2];
                ptc.x3[offset_p] = x[2];

                auto p = f_dist(pos, grid, ext, rng.m_local_state, PtcType::electron);
                ptc.p1[offset_e] = p[0];
                ptc.p2[offset_e] = p[1];
                ptc.p3[offset_e] = p[2];
                ptc.E[offset_e] = math::sqrt(1.0f + p.dot(p));

                p = f_dist(pos, grid, ext, rng.m_local_state, PtcType::positron);
                ptc.p1[offset_p] = p[0];
                ptc.p2[offset_p] = p[1];
                ptc.p3[offset_p] = p[2];
                ptc.E[offset_p] = math::sqrt(1.0f + p.dot(p));

                auto x_global = grid.coord_global(pos, x);
                ptc.weight[offset_e] = f_weight(x_global);
                ptc.weight[offset_p] = f_weight(x_global);
                ptc.flag[offset_e] = set_ptc_type_flag(flag, PtcType::electron);
                ptc.flag[offset_p] = set_ptc_type_flag(flag, PtcType::positron);
              }
            }
          }
        },
        ptc, rng_states, m_num_per_cell, m_cum_num_per_cell, f_dist, f_weight);
    policy::sync();
    Logger::print_detail_all("Finished injecting particles");
    ptc->add_num(new_particles);
  }

 private:
  const Grid<Conf::dim, value_t>& m_grid;
  nonown_ptr<particle_data_t> ptc;
  nonown_ptr<rng_states_t<exec_tags::device>> rng_states;

  multi_array<int, Conf::dim> m_num_per_cell, m_cum_num_per_cell;
};

}  // namespace Aperture

#endif  // _PTC_INJECTOR_CUDA_H_
