/*
 * Copyright (c) 2024 Alex Chen & Yajie Yuan.
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

#pragma once

#include "data/fields.h"
#include "data/rng_states.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/domain_comm.h"
#include "systems/grid_ks.hpp"
#include "utils/nonown_ptr.hpp"
#include "utils/util_functions.h"
#include <memory>

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
class disk_boundary_condition : public system_t {
 public:
  typedef typename Conf::value_t value_t;

  static std::string name() { return "disk_boundary_condition"; }

  disk_boundary_condition(const grid_sph_t<Conf>& grid,
                     const domain_comm<Conf, ExecPolicy>* comm = nullptr)
      : m_grid(grid), m_comm(comm) {}
  ~disk_boundary_condition() = default;

  void init() override {
    sim_env().get_data("Edelta", E);
    sim_env().get_data("Bdelta", B);
    sim_env().get_data("E0", E0);
    sim_env().get_data("B0", B0);
    sim_env().get_data("particles", ptc);

    sim_env().params().get_value("disk_dth", m_disk_dth);
  }

  void update(double dt, uint32_t step) override {
    auto ext = m_grid.extent();
    typedef typename Conf::idx_t idx_t;

    value_t time = sim_env().get_time();
    value_t disk_dth = m_disk_dth;

    ExecPolicy<Conf>::launch(
        [ext, disk_dth] LAMBDA(auto e, auto b, auto e0, auto b0) {
          auto& grid = ExecPolicy<Conf>::grid();
          // Theta limits of the conducting disk
          value_t th_1 = M_PI_2 - 0.5 * disk_dth;
          value_t th_2 = M_PI_2 + 0.5 * disk_dth;

          ExecPolicy<Conf>::loop(Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
            auto pos = get_pos(idx, ext);
            if (grid.is_in_bound(pos)) {
              // compute the theta at the center of the current cell
              value_t theta =
                  grid_sph_t<Conf>::theta(grid.template coord<1>(pos[1], false));

              if (theta < th_2 && theta > th_1) {
                // TODO: finish the conductor condition
                e[0][idx] = 0.0;
                b[1][idx] = 0.0;
                b[2][idx] = 0.0;
              }
            }
          });
        },
        E, B, E0, B0);
    ExecPolicy<Conf>::sync();

    // TODO: remove particles that go into the conductor
  }

 private:
  const grid_sph_t<Conf>& m_grid;
  const domain_comm<Conf, ExecPolicy>* m_comm = nullptr;
  value_t m_disk_dth = 0.1;

  nonown_ptr<vector_field<Conf>> E, B, E0, B0;
  nonown_ptr<particles_t> ptc;
};

}  // namespace Aperture

