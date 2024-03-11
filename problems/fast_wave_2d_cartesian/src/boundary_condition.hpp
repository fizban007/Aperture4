/*
 * Copyright (c) 2024 Alex Chen.
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
#include "systems/grid.h"
#include "utils/nonown_ptr.hpp"
#include "utils/util_functions.h"
#include <memory>

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
class boundary_condition : public system_t {
 protected:
  const grid_t<Conf>& m_grid;
  const domain_comm<Conf, ExecPolicy>* m_comm = nullptr;

  double m_kB_angle = 0.0; // angle of the wave vector wrt the magnetic field
  double m_A0 = 1.0;  // wave amplitude
  double m_freq = 5.0; // frequency
  double m_Bp = 1.0; // background magnetic field (always in y direction)
  int m_num_lambda = 4; // number of wavelengths

  nonown_ptr<vector_field<Conf>> E, B, E0, B0;

 public:
  static std::string name() { return "boundary_condition"; }

  boundary_condition(const grid_t<Conf>& grid,
                     const domain_comm<Conf, ExecPolicy>* comm = nullptr)
      : m_grid(grid), m_comm(comm) {}

  void init() override {
    sim_env().get_data("Edelta", E);
    sim_env().get_data("E0", E0);
    sim_env().get_data("Bdelta", B);
    sim_env().get_data("B0", B0);

    sim_env().params().get_value("A0", m_A0);
    sim_env().params().get_value("wave_freq", m_freq);
    sim_env().params().get_value("Bp", m_Bp);
    sim_env().params().get_value("num_lambda", m_num_lambda);
  }

  void update(double dt, uint32_t step) override {
    if (m_comm == nullptr || m_comm->domain_info().is_boundary[0]) {
      typedef typename Conf::idx_t idx_t;
      typedef typename Conf::value_t value_t;

      value_t time = sim_env().get_time();
      value_t Bp = m_Bp;
      value_t theta = m_kB_angle;
      value_t amp = 0.0;
      value_t phase = 2.0 * M_PI * m_freq * time;
      if (phase < 2.0 * M_PI * m_num_lambda)
        amp = m_A0 * sin(phase);

      // Set the boundary condition. We are launching a fast wave, so the
      // electric field is out of the k-B plane, in the z direction. The
      // magnetic field is in the k-B plane, and may be non-zero in both x and
      // y directions.
      ExecPolicy<Conf>::launch(
          [=] LAMBDA(auto e, auto b, auto e0, auto b0) {
            auto& grid = ExecPolicy<Conf>::grid();
            auto ext = grid.extent();
            // Launching from the x=0 boundary
            ExecPolicy<Conf>::loop(0, grid.dims[1], [&] LAMBDA(auto n1) {
              // For quantities that are not continuous across the surface
              // int n0 = grid.guard[0];
              for (int n0 = 0; n0 < grid.guard[0]; n0++) {
                auto idx = idx_t(index_t<2>(n0, n1), ext);
                e[0][idx] = 0.0;
                b[1][idx] = amp * sin(phase) * sin(theta);
                b[2][idx] = 0.0;
              }
              // For quantities that are continuous across the surface
              for (int n0 = 0; n0 < grid.guard[0] + 1; n0++) {
                // n0 = grid.guard[0] + 1;
                auto idx2 = idx_t(index_t<2>(n0, n1), ext);
                b[0][idx2] = -amp * sin(phase) * cos(theta);
                e[1][idx2] = 0.0;
                e[2][idx2] = amp * sin(phase);
              }
            });
          },
          E, B, E0, B0);
      ExecPolicy<Conf>::sync();
    }
  }
};

}  // namespace Aperture
