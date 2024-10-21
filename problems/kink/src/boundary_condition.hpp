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

  double m_beta_z = 0.5;
  double m_B0 = 100.0;
  double m_sigma = 5.0;
  double m_ppc = 5;
  double m_nc = 10.0;
  double m_n0 = 1.0;
  double m_qe = 1.0;

  nonown_ptr<vector_field<Conf>> E, B;

 public:
  static std::string name() { return "boundary_condition"; }

  boundary_condition(const grid_t<Conf>& grid,
                     const domain_comm<Conf, ExecPolicy>* comm = nullptr)
      : m_grid(grid), m_comm(comm) {}

  void init() override {
    sim_env().get_data("Edelta", E);
    sim_env().get_data("Bdelta", B);

    sim_env().params().get_value("beta_z", m_beta_z);
    sim_env().params().get_value("sigma", m_sigma);
    sim_env().params().get_value("Bp", m_B0);
    sim_env().params().get_value("ppc", m_ppc);
    sim_env().params().get_value("q_e", m_qe);
    m_nc = m_B0 * m_B0 / m_sigma;
    m_n0 = sim_env().params().get_as<double>("n_0", m_nc / 10.0);
  }

  void update(double dt, uint32_t step) override {
    if (m_comm == nullptr || m_comm->domain_info().is_boundary[4]) {
      typedef typename Conf::idx_t idx_t;
      typedef typename Conf::value_t value_t;

      value_t time = sim_env().get_time();
      value_t B0 = m_B0;
      value_t ppc = m_ppc;
      value_t sigma = m_sigma;
      value_t qe = m_qe;
      value_t beta_z = m_beta_z;
      value_t gamma_z = 1.0 / math::sqrt(1.0 - beta_z * beta_z);
      value_t n0 = m_n0;
      value_t nc = m_nc;
      
      auto Bphi = [B0] LAMBDA (auto r) {
        return B0 * r * std::exp(1 - r);
      };

      // field boundary conditions
      ExecPolicy<Conf>::launch(
          [gamma_z, beta_z, B0, Bphi] LAMBDA(auto e, auto b) {
            auto& grid = ExecPolicy<Conf>::grid();
            auto ext = grid.extent();
            ExecPolicy<Conf>::loop(0, grid.dims[0] * grid.dims[1], [&] LAMBDA(auto n) {
              int n0 = n % grid.dims[0];
              int n1 = n / grid.dims[0];
              value_t x = grid.coord(0, n0, false);
              value_t xs = grid.coord(0, n0, true);
              value_t y = grid.coord(1, n1, false);
              value_t ys = grid.coord(1, n1, true);
              // For quantities that are not continuous across the surface
              // int n0 = grid.guard[0];
              for (int n2 = 0; n2 < grid.guard[2]; n2++) {
                auto idx = idx_t(index_t<3>(n0, n1, n2), ext);
                value_t r = math::sqrt(x*x + ys*ys);
                e[0][idx] = gamma_z * beta_z * Bphi(r) * x / r;
                b[1][idx] = gamma_z * Bphi(r) * x / r;
                b[2][idx] = 0.0;
              }
              // For quantities that are continuous across the surface
              for (int n2 = 0; n2 < grid.guard[2] + 1; n2++) {
                auto idx = idx_t(index_t<3>(n0, n1, n2), ext);
                value_t r = math::sqrt(xs*xs + y*y);
                b[0][idx] = -gamma_z * y * Bphi(r) / r;
                e[1][idx] = gamma_z * beta_z * Bphi(r) * y / r;
                e[2][idx] = 0.0;
              }
            });
          },
          E, B);
      ExecPolicy<Conf>::sync();

      // particle injection at the boundary
    }
  }
};

}  // namespace Aperture

