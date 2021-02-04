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

#ifndef __PTC_UPDATER_BASE_IMPL_H_
#define __PTC_UPDATER_BASE_IMPL_H_

#include "framework/environment.h"
#include "systems/helpers/ptc_update_helper.hpp"
#include "systems/ptc_updater_base.h"
#include "utils/interpolation.hpp"

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::ptc_updater(
    const grid_t<Conf>& grid)
    : m_grid(grid) {
  sim_env().params().get_value("fld_output_interval", m_data_interval);
  // By default, rho_interval is the same as field output interval
  m_rho_interval = m_data_interval;
  // Override if there is a specific rho_interval option specified
  sim_env().params().get_value("rho_interval", m_rho_interval);

  sim_env().params().get_value("sort_interval", m_sort_interval);
  sim_env().params().get_value("current_smoothing", m_filter_times);
  init_charge_mass();

  ExecPolicy<Conf>::set_grid(m_grid);
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::init_charge_mass() {
  // Default values are 1.0
  float q_e = 1.0;
  float ion_mass = 1.0;
  sim_env().params().get_value("q_e", q_e);
  sim_env().params().get_value("ion_mass", ion_mass);

  for (int i = 0; i < (max_ptc_types); i++) {
    m_charges[i] = q_e;
    m_masses[i] = q_e;
  }
  m_charges[(int)PtcType::electron] *= -1.0;
  m_masses[(int)PtcType::ion] *= ion_mass;
  for (int i = 0; i < (max_ptc_types); i++) {
    m_q_over_m[i] = m_charges[i] / m_masses[i];
  }

#ifdef CUDA_ENABLED
  init_dev_charge_mass(this->m_charges.data(), this->m_masses.data());
#endif
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::init() {
  // Allocate the tmp array for current filtering
  // jtmp = std::make_unique<typename Conf::multi_array_t>(m_grid.extent(),
  //                                                       MemType::host_only);

  sim_env().get_data_optional("photons", ph);
  sim_env().get_data_optional("Rho_ph", rho_ph);
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy,
            PhysicsPolicy>::register_data_components() {
  size_t max_ptc_num = 10000;
  sim_env().params().get_value("max_ptc_num", max_ptc_num);

  ptc = sim_env().register_data<particle_data_t>(
      "particles", max_ptc_num, ExecPolicy<Conf>::tmp_mem_type());
  ptc->include_in_snapshot(true);

  E = sim_env().register_data<vector_field<Conf>>(
      "E", m_grid, field_type::edge_centered,
      ExecPolicy<Conf>::data_mem_type());
  B = sim_env().register_data<vector_field<Conf>>(
      "B", m_grid, field_type::face_centered,
      ExecPolicy<Conf>::data_mem_type());
  J = sim_env().register_data<vector_field<Conf>>(
      "J", m_grid, field_type::edge_centered,
      ExecPolicy<Conf>::data_mem_type());

  sim_env().params().get_value("num_species", m_num_species);
  if (m_num_species > max_ptc_types) {
    Logger::print_err("Too many species of particles requested! Exiting");
    throw std::runtime_error("too many species");
  }

  Rho.resize(m_num_species);
  for (int i = 0; i < m_num_species; i++) {
    Rho.set(i,
            sim_env().register_data<scalar_field<Conf>>(
                std::string("Rho_") + ptc_type_name(i), m_grid,
                field_type::vert_centered, ExecPolicy<Conf>::data_mem_type()));
  }
  Rho.copy_to_device();

  size_t seed = default_random_seed;
  sim_env().params().get_value("random_seed", seed);
  rng_states = sim_env().register_data<rng_states_t>("rng_states", seed);
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::update(
    double dt, uint32_t step) {
  update_particles(dt, step);

  // Communicate deposited current and charge densities
  if (m_comm != nullptr) {
    m_comm->send_add_guard_cells(*J);
    m_comm->send_guard_cells(*J);
    // if ((step + 1) % m_data_interval == 0) {
    if (step % m_rho_interval == 0) {
      for (uint32_t i = 0; i < Rho.size(); i++) {
        m_comm->send_add_guard_cells(*(Rho[i]));
        m_comm->send_guard_cells(*(Rho[i]));
      }
    }
  }

  // Send particles
  if (m_comm != nullptr) {
    m_comm->send_particles(*ptc, m_grid);
  }

  // Also move photons if the data component exists
  if (ph != nullptr) {
    Logger::print_info("Moving {} photons", ph->number());
    update_photons(dt, step);

    if (m_comm != nullptr) {
      m_comm->send_particles(*ph, m_grid);
    }
  }

  // Clear guard cells
  clear_guard_cells();

  // sort at the given interval. Turn off sorting if m_sort_interval is 0
  if (m_sort_interval > 0 && (step % m_sort_interval) == 0) {
    sort_particles();
  }
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::update_particles(
    value_t dt, uint32_t step) {
  auto num = ptc->number();
  if (num == 0) return;
  int rho_interval = m_rho_interval;
  auto charges = m_charges;
  auto masses = m_masses;

  // Main particle update loop
  ExecPolicy<Conf>::launch(
      [num, dt, rho_interval, step, charges, masses] LAMBDA(
          auto ptc, auto E, auto B, auto J, auto Rho) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        auto interp = interpolator<typename Conf::spline_t, Conf::dim>{};
        bool deposit_rho = (step % rho_interval == 0);
        ExecPolicy<Conf>::loop(
            [&ext, &charges, &masses, dt, deposit_rho, interp] LAMBDA(
                auto n, auto& ptc, auto& E, auto& B, auto& J, auto& Rho,
                auto& grid) {
              ptc_context<Conf::dim, value_t> context;
              context.cell = ptc.cell[n];
              if (context.cell == empty_cell) return;

              auto idx = Conf::idx(context.cell, ext);
              auto pos = get_pos(idx, ext);

              context.x = vec_t<value_t, 3>(ptc.x1[n], ptc.x2[n], ptc.x3[n]);
              context.p = vec_t<value_t, 3>(ptc.p1[n], ptc.p2[n], ptc.p3[n]);
              context.gamma = ptc.E[n];

              context.flag = ptc.flag[n];
              context.sp = get_ptc_type(context.flag);
              context.weight = charges[context.sp] * ptc.weight[n];

              context.E[0] = interp(E[0], context.x, idx, stagger_t(0b110));
              context.E[1] = interp(E[1], context.x, idx, stagger_t(0b101));
              context.E[2] = interp(E[2], context.x, idx, stagger_t(0b011));
              context.B[0] = interp(B[0], context.x, idx, stagger_t(0b001));
              context.B[1] = interp(B[1], context.x, idx, stagger_t(0b010));
              context.B[2] = interp(B[2], context.x, idx, stagger_t(0b100));

              CoordPolicy<Conf>::update_ptc(
                  grid, context, pos, charges[context.sp] / masses[context.sp],
                  dt);

              ptc.p1[n] = context.p[0];
              ptc.p2[n] = context.p[1];
              ptc.p3[n] = context.p[2];
              ptc.E[n] = context.gamma;

              deposit_t<Conf::dim, typename Conf::spline_t> deposit{};
              deposit(context, J, Rho, idx, dt, deposit_rho);

              ptc.x1[n] = context.new_x[0];
              ptc.x2[n] = context.new_x[1];
              ptc.x3[n] = context.new_x[2];
              ptc.cell[n] = Conf::idx(pos, ext).linear;
            },
            0ul, num, ptc, E, B, J, Rho, grid);
      },
      *ptc, *E, *B, *J, Rho);
  ExecPolicy<Conf>::sync();

  // ExecPolicy<Conf>::launch(CoordPolicy<Conf>::process_J_Rho, *J, *Rho);
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::update_photons(
    value_t dt, uint32_t step) {
  auto num = ph->number();
  if (num == 0) return;
  int rho_interval = m_rho_interval;

  // Photon movement loop
  ExecPolicy<Conf>::launch(
      [num, dt, rho_interval, step] LAMBDA(auto ph, auto Rho_ph) {

      },
      *ph, *rho_ph);
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::clear_guard_cells() {
  auto num = ptc->number();

  ExecPolicy<Conf>::launch(
      [num] LAMBDA(auto ptc) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        ExecPolicy<Conf>::loop(
            [ext] LAMBDA(auto n, auto& grid, auto& ptc) {
              auto cell = ptc.cell[n];
              if (cell == empty_cell) return;

              auto idx = Conf::idx(cell, ext);
              auto pos = get_pos(idx, ext);
              if (!grid.is_in_bound(pos)) ptc.cell[n] = empty_cell;
            },
            0ul, num, grid, ptc);
      },
      *ptc);

  ExecPolicy<Conf>::launch(
      [num] LAMBDA(auto ph) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        ExecPolicy<Conf>::loop(
            [ext] LAMBDA(auto n, auto& grid, auto& ph) {
              auto cell = ph.cell[n];
              if (cell == empty_cell) return;

              auto idx = Conf::idx(cell, ext);
              auto pos = get_pos(idx, ext);
              if (!grid.is_in_bound(pos)) ph.cell[n] = empty_cell;
            },
            0ul, num, grid, ph);
      },
      *ph);
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::sort_particles() {
  ptc->sort_by_cell(m_grid.extent().size());
  if (ph != nullptr) ph->sort_by_cell(m_grid.extent().size());
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::fill_multiplicity(
    int mult, value_t weight) {
  CoordPolicy<Conf>::template fill_multiplicity<ExecPolicy<Conf>>(
      *ptc, *rng_states, mult, weight);
  ptc->set_num(ptc->number() + 2 * mult * m_grid.extent().size());
}

}  // namespace Aperture

#endif  // __PTC_UPDATER_BASE_IMPL_H_
