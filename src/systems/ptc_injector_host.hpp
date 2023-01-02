/*
 * Copyright (c) 2022 Alex Chen.
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

// #ifndef PTC_INJECTOR_HOST_H_
// #define PTC_INJECTOR_HOST_H_
#pragma once

// #include "core/cached_allocator.hpp"
#include "core/multi_array.hpp"
#include "data/fields.h"
#include "data/particle_data.h"
#include "data/rng_states.h"
#include "framework/system.h"
#include "systems/grid.h"
#include "systems/policies/exec_policy_host.hpp"
#include "systems/ptc_injector_new.h"
#include "utils/range.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
class ptc_injector<Conf, exec_policy_host> {
 public:
  using value_t = typename Conf::value_t;
  static std::string name() { return "ptc_injector"; }

  ptc_injector(const Grid<Conf::dim, value_t>& grid) : m_grid(grid) {
    auto ext = grid.extent();
    sim_env().get_data("particles", ptc);
    sim_env().get_data("rng_states", rng_states);
    sim_env().params().get_value("tracked_fraction", m_tracked_fraction);
  }

  ~ptc_injector() {}

  template <typename FCriteria, typename FDist, typename FNumPerCell,
            typename FWeight>
  void inject(const FCriteria& f_criteria, const FNumPerCell& f_num,
              const FDist& f_dist, const FWeight& f_weight, uint32_t flag = 0) {
    using policy = exec_policy_host<Conf>;
    auto& grid = m_grid;
    auto num = ptc->number();
    auto max_num = ptc->size();
    size_t cum_num = 0;
    auto tracked_fraction = m_tracked_fraction;

    // Inject particles per cell
    policy::launch(
        [&](ptc_ptrs ptc, auto states) {
          auto ext = grid.extent();
          rng_t<exec_tags::host> rng(states);

          for (auto idx : range(Conf::begin(ext), Conf::end(ext))) {
            auto pos = get_pos(idx, ext);
            if (grid.is_in_bound(pos)) {
              int num_inject = 0;
              // First check whether to inject in this cell
              if (f_criteria(pos, grid, ext)) {
                // Get the number of particles injected
                num_inject = f_num(pos, grid, ext);
              }
              if (num_inject > 0) {
                for (int n = 0; n < num_inject; n += 2) {
                  uint32_t offset_e = num + cum_num + n;
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

                  auto p = f_dist(pos, grid, ext, rng.m_local_state,
                                  PtcType::electron);
                  ptc.p1[offset_e] = p[0];
                  ptc.p2[offset_e] = p[1];
                  ptc.p3[offset_e] = p[2];
                  ptc.E[offset_e] = math::sqrt(1.0f + p.dot(p));

                  p = f_dist(pos, grid, ext, rng.m_local_state,
                             PtcType::positron);
                  ptc.p1[offset_p] = p[0];
                  ptc.p2[offset_p] = p[1];
                  ptc.p3[offset_p] = p[2];
                  ptc.E[offset_p] = math::sqrt(1.0f + p.dot(p));

                  auto x_global = grid.coord_global(pos, x);
                  ptc.weight[offset_e] = f_weight(x_global);
                  ptc.weight[offset_p] = f_weight(x_global);
                  auto u = rng.uniform<value_t>();
                  // printf("u is %f, tracked_fraction is %f\n", u, tracked_fraction);
                  uint32_t local_flag = flag;
                  if (u < tracked_fraction) {
                    set_flag(local_flag, PtcFlag::tracked);
                  }
                  ptc.flag[offset_e] =
                      set_ptc_type_flag(local_flag, PtcType::electron);
                  ptc.flag[offset_p] =
                      set_ptc_type_flag(local_flag, PtcType::positron);
                }
                cum_num += num_inject;
              }

            }
          }
        },
        ptc, rng_states);
    policy::sync();
    Logger::print_debug("Current num is {}, injecting {}, max_num is {}", num,
                        cum_num, max_num);
    ptc->add_num(cum_num);
  }

 private:
  const Grid<Conf::dim, value_t>& m_grid;
  nonown_ptr<particle_data_t> ptc;
  nonown_ptr<rng_states_t<exec_tags::host>> rng_states;

  float m_tracked_fraction = 0.0f;
};

}  // namespace Aperture

// #endif  // PTC_INJECTOR_HOST_H_
