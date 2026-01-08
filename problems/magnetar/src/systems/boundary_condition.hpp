/*
 * Copyright (c) 2023 Alex Chen.
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
#include "systems/grid_sph.hpp"
#include "utils/nonown_ptr.hpp"
#include "utils/util_functions.h"
#include <memory>

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
class boundary_condition : public system_t {
 public:
  typedef typename Conf::value_t value_t;

  static std::string name() { return "boundary_condition"; }

  boundary_condition(const grid_sph_t<Conf>& grid,
                     const domain_comm<Conf, ExecPolicy>* comm = nullptr)
      : m_grid(grid), m_comm(comm) {
    if (comm != nullptr) {
      m_track_rank = comm->rank();
      m_track_rank <<= 32;
    }
  }
  ~boundary_condition() = default;

  void init() override {
    sim_env().get_data("Edelta", E);
    sim_env().get_data("Bdelta", B);
    sim_env().get_data("E0", E0);
    sim_env().get_data("B0", B0);
    sim_env().get_data("particles", ptc);
    sim_env().get_data("rng_states", rng_states);

    sim_env().params().get_value("atm_time", m_atm_time);
    
    sim_env().params().get_value("ramp_time", ramp_time);
    sim_env().params().get_value("twist_omega", m_twist_omega);
    sim_env().params().get_value("twist_time", m_twist_time);
    sim_env().params().get_value("twist_rmax_1", m_twist_rmax_1);
    sim_env().params().get_value("twist_rmax_2", m_twist_rmax_2);
    sim_env().params().get_value("q_e", m_qe);
    sim_env().params().get_value("tracked_fraction", m_tracked_fraction);
    sim_env().params().get_value("min_surface_E", m_min_surface_E);
  }

  void update(double dt, uint32_t step) override {
    // Check for multi_rank and when we are at r=r_* boundary
    if (m_comm != nullptr && !m_comm->domain_info().is_boundary[0]) {
      return;
    }

    auto ext = m_grid.extent();
    typedef typename Conf::idx_t idx_t;

    value_t time = sim_env().get_time();
    value_t omega;
    if (time <= m_atm_time) {
      omega = 0.0;
    } else if (time <= m_atm_time + m_twist_time) {
      value_t  t_after_atm = time - m_atm_time;
      omega = m_twist_omega;
      if (t_after_atm < ramp_time) {
        omega *= square(std::sin(0.5 * M_PI * t_after_atm / ramp_time));
      }
      if (m_twist_time - t_after_atm < ramp_time) {
        omega *= square(std::sin(
            0.5 * M_PI * (m_twist_time - t_after_atm) / ramp_time));
      }

      // omega = m_twist_omega *
      //         square(std::sin(M_PI * (time - m_atm_time) / m_twist_time));
      // omega = m_twist_omega * ((time - m_atm_time) / m_twist_time);
    } else {
      omega = 0.0;
    }

    // Impart twist on the stellar surface
    value_t twist_th1 = math::asin(math::sqrt(1.0 / m_twist_rmax_1));
    value_t twist_th2 = math::asin(math::sqrt(1.0 / m_twist_rmax_2));
    Logger::print_debug("time is {}, Omega is {}, th1 is {}, th2 is {}", time,
                        omega, twist_th1, twist_th2);

    ExecPolicy<Conf>::launch(
        [ext, time, omega, twist_th1, twist_th2] LAMBDA(auto e, auto b, auto e0,
                                                        auto b0) {
          auto& grid = ExecPolicy<Conf>::grid();
          value_t th_m = (twist_th1 + twist_th2) * 0.5f;
          // for (auto n1 : grid_stride_range(0, grid.dims[1])) {
          ExecPolicy<Conf>::loop(0, grid.dims[1], [&] LAMBDA(auto n1) {
            value_t theta =
                grid_sph_t<Conf>::theta(grid.template coord<1>(n1, false));
            value_t theta_s =
                grid_sph_t<Conf>::theta(grid.template coord<1>(n1, true));
            if ((theta_s >= twist_th1 && theta < twist_th2) ||
                (theta_s < M_PI - twist_th1 && theta >= M_PI - twist_th2)) {
              value_t s = (theta > 0.5f * M_PI ? -1.0f : 1.0f); // enforcing sign of the twist to be hemisphere dependent
              if (theta > 0.5f * M_PI) {
                th_m = M_PI - th_m;
              }
              // For quantities that are not continuous across the surface
              for (int n0 = 0; n0 < grid.guard[0]; n0++) {
                auto idx = idx_t(index_t<2>(n0, n1), ext);
                value_t r =
                    grid_sph_t<Conf>::radius(grid.template coord<0>(n0, false));
                e[0][idx] = s * omega * sin(theta_s) * r * b0[1][idx] *
                            square(math::cos(M_PI * (theta_s - th_m) /
                                             (twist_th2 - twist_th1)));
                b[1][idx] = 0.0;
                b[2][idx] = 0.0;
              }
              // For quantities that are continuous across the surface
              for (int n0 = 0; n0 < grid.guard[0] + 1; n0++) {
                auto idx = idx_t(index_t<2>(n0, n1), ext);
                value_t r_s =
                    grid_sph_t<Conf>::radius(grid.template coord<0>(n0, true));
                b[0][idx] = 0.0;
                e[1][idx] = -s * omega * sin(theta) * r_s * b0[0][idx] *
                            square(math::cos(M_PI * (theta - th_m) /
                                             (twist_th2 - twist_th1)));
                e[2][idx] = 0.0;
              }
            }
          });
        },
        E, B, E0, B0);
    // ExecPolicy<Conf>::sync();

    // Inject particles at the surface according to surface electric field
    // Define a variable to hold the moving position in the ptc array where we
    // insert new particles
    buffer<unsigned long long int> pos(1, ExecPolicy<Conf>::data_mem_type());
    pos[0] = ptc->number();
    pos.copy_to_device();
    auto ptc_num_orig = pos[0];
    value_t qe = m_qe;
    value_t tracked_fraction = m_tracked_fraction;
    auto track_rank = m_track_rank;
    auto min_E = m_min_surface_E;

    ExecPolicy<Conf>::launch(
        [ext, qe, tracked_fraction, track_rank, min_E] LAMBDA(
            auto e, auto ptc, auto ptc_pos, auto ptc_id, auto states) {
          auto& grid = ExecPolicy<Conf>::grid();
          rng_t<typename ExecPolicy<Conf>::exec_tag> rng(states);
          ExecPolicy<Conf>::loop(0, grid.dims[1], [&] LAMBDA(auto n1) {
            value_t theta =
                grid_sph_t<Conf>::theta(grid.template coord<1>(n1, false));
            int n0 = grid.guard[0] + 3;
            auto idx = idx_t(index_t<2>(n0, n1), ext);
            value_t E_surface = e[0][idx];

            // TODO: Limit the injection range to the same region that we are
            // twisting (applying field boundary conditions)

            if (math::abs(E_surface) > min_E) {
              value_t w = math::abs(E_surface) / qe * 0.1;
              size_t ptc_offset = atomic_add(&ptc_pos[0], 2);

              if (ptc.cell[ptc_offset] != empty_cell) {
                return;
              }

              float u = rng.template uniform<float>();
              // ptc_offset is electron and ptc_offset + 1 is positron
              ptc.x1[ptc_offset] = ptc.x1[ptc_offset + 1] = 0.0f;
              ptc.x2[ptc_offset] = ptc.x2[ptc_offset + 1] = u;
              ptc.x3[ptc_offset] = ptc.x3[ptc_offset + 1] = 0.0f;
              // Initializing the particle at rest
              ptc.p1[ptc_offset] = ptc.p1[ptc_offset + 1] = 0.0f;
              ptc.p2[ptc_offset] = ptc.p2[ptc_offset + 1] = 0.0f;
              ptc.p3[ptc_offset] = ptc.p3[ptc_offset + 1] = 0.0f;
              ptc.E[ptc_offset] = ptc.E[ptc_offset + 1] = 1.0f;

              ptc.weight[ptc_offset] = ptc.weight[ptc_offset + 1] =
                  w * math::sin(theta);
              ptc.cell[ptc_offset] = ptc.cell[ptc_offset + 1] = idx.linear;
              u = rng.template uniform<float>();
              uint32_t flag = 0;
              if (u < tracked_fraction) {
                flag = flag_or(PtcFlag::tracked);
                ptc.id[ptc_offset] = track_rank + atomic_add(ptc_id, 1);
                ptc.id[ptc_offset + 1] = track_rank + atomic_add(ptc_id, 1);
              }
              ptc.flag[ptc_offset] = set_ptc_type_flag(flag, PtcType::electron);
              ptc.flag[ptc_offset + 1] =
                  set_ptc_type_flag(flag, PtcType::positron);
            }
          });
        },
        E, ptc, pos, ptc->ptc_id(), rng_states);
    ExecPolicy<Conf>::sync();

    pos.copy_to_host();
    ptc->set_num(pos[0]);
    Logger::print_info("{} particles are injected!", pos[0] - ptc_num_orig);
  }

 private:
  const grid_sph_t<Conf>& m_grid;
  const domain_comm<Conf, ExecPolicy>* m_comm = nullptr;
  value_t m_twist_omega = 0.0;
  value_t m_atm_time = 5.0;
  value_t m_twist_time = 20.0;
  value_t m_twist_rmax_1 = 3.0;
  value_t m_twist_rmax_2 = 5.0;
  value_t m_qe = 1.0;
  value_t m_tracked_fraction = 0.1;
  value_t m_min_surface_E = 0.01;
  uint64_t m_track_rank = 0;

  value_t ramp_time = 1.0;

  nonown_ptr<vector_field<Conf>> E, B, E0, B0;
  nonown_ptr<particles_t> ptc;
  nonown_ptr<rng_states_t<typename ExecPolicy<Conf>::exec_tag>> rng_states;
};

}  // namespace Aperture
