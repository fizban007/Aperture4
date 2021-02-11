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

#include "data/rng_states.h"
#include "framework/environment.h"
#include "systems/helpers/filter_field.hpp"
#include "systems/helpers/ptc_update_helper.hpp"
#include "systems/ptc_updater_base.h"
#include "utils/interpolation.hpp"

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
ptc_updater_new<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::ptc_updater_new(
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

  // Allocate the tmp array for current filtering
  m_tmpj.set_memtype(ExecPolicy<Conf>::tmp_mem_type());
  m_tmpj.resize(m_grid.extent());

  ExecPolicy<Conf>::set_grid(m_grid);

  m_coord_policy = std::make_unique<CoordPolicy<Conf>>(grid);
  m_phys_policy = std::make_unique<PhysicsPolicy<Conf>>();
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
ptc_updater_new<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::ptc_updater_new(
    const grid_t<Conf>& grid, const domain_comm<Conf>& comm)
    : ptc_updater_new(grid) {
  m_comm = &comm;
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
ptc_updater_new<Conf, ExecPolicy, CoordPolicy,
                PhysicsPolicy>::~ptc_updater_new() = default;

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater_new<Conf, ExecPolicy, CoordPolicy,
                PhysicsPolicy>::init_charge_mass() {
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

  // #ifdef CUDA_ENABLED
  //   init_dev_charge_mass(this->m_charges.data(), this->m_masses.data());
  // #endif
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater_new<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::init() {
  sim_env().get_data_optional("photons", ph);
  sim_env().get_data_optional("Rho_ph", rho_ph);

  m_phys_policy->init();
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater_new<Conf, ExecPolicy, CoordPolicy,
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
    Logger::print_err("Too many species of particles requested! Aborting");
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
ptc_updater_new<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::update(
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

  Logger::print_detail("Finished sending particles");

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

  Logger::print_detail("Finished clearing guard cells");

  // sort at the given interval. Turn off sorting if m_sort_interval is 0
  if (m_sort_interval > 0 && (step % m_sort_interval) == 0) {
    sort_particles();
  }
  Logger::print_detail("Finished sorting");
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater_new<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::update_particles(
    value_t dt, uint32_t step, size_t begin, size_t end) {
  if (end - begin <= 0) return;
  int rho_interval = m_rho_interval;
  auto charges = m_charges;
  auto masses = m_masses;
  auto coord_policy = *m_coord_policy;
  auto phys_policy = *m_phys_policy;
  bool deposit_rho = (step % rho_interval == 0);

  // Main particle update loop
  ExecPolicy<Conf>::launch(
      [begin, end, dt, rho_interval, deposit_rho, charges, masses, coord_policy,
       phys_policy] LAMBDA(auto ptc, auto E, auto B, auto J, auto Rho) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        // auto interp = interpolator<typename Conf::spline_t, Conf::dim>{};
        auto interp = interp_t<1, Conf::dim>{};
        ExecPolicy<Conf>::loop(
            begin, end,
            [&ext, &charges, &masses, &coord_policy, dt,
             deposit_rho, interp] LAMBDA(auto n, auto& ptc, auto& E, auto& B,
                                         auto& J, auto& Rho, auto& grid, auto& phys_policy) {
              ptc_context<Conf::dim, int32_t, uint32_t, value_t> context;
              context.cell = ptc.cell[n];
              if (context.cell == empty_cell) return;

              typename Conf::idx_t idx = Conf::idx(context.cell, ext);

              context.x = vec_t<value_t, 3>(ptc.x1[n], ptc.x2[n], ptc.x3[n]);
              context.p = vec_t<value_t, 3>(ptc.p1[n], ptc.p2[n], ptc.p3[n]);
              context.gamma = ptc.E[n];

              context.flag = ptc.flag[n];
              context.sp = get_ptc_type(context.flag);
              context.weight = charges[context.sp] * ptc.weight[n];
              context.q = charges[context.sp];
              context.m = masses[context.sp];

              // context.E[0] = interp(E[0], context.x, idx, stagger_t(0b110));
              // context.E[1] = interp(E[1], context.x, idx, stagger_t(0b101));
              // context.E[2] = interp(E[2], context.x, idx, stagger_t(0b011));
              // context.B[0] = interp(B[0], context.x, idx, stagger_t(0b001));
              // context.B[1] = interp(B[1], context.x, idx, stagger_t(0b010));
              // context.B[2] = interp(B[2], context.x, idx, stagger_t(0b100));
              context.E[0] =
                  interp(context.x, E[0], idx, ext, stagger_t(0b110));
              context.E[1] =
                  interp(context.x, E[1], idx, ext, stagger_t(0b101));
              context.E[2] =
                  interp(context.x, E[2], idx, ext, stagger_t(0b011));
              context.B[0] =
                  interp(context.x, B[0], idx, ext, stagger_t(0b001));
              context.B[1] =
                  interp(context.x, B[1], idx, ext, stagger_t(0b010));
              context.B[2] =
                  interp(context.x, B[2], idx, ext, stagger_t(0b100));

              // printf("x1: %f, x2: %f, p1: %f, p2: %f, q_over_m: %f, dt:
              // %f\n",
              //        context.x[0], context.x[1], context.p[0], context.p[1],
              //        charges[context.sp] / masses[context.sp], dt);

              auto pos = get_pos(idx, ext);
              coord_policy.update_ptc(grid, context, pos,
                                      // charges[context.sp] / masses[context.sp],
                                      dt);

              phys_policy(grid, context, pos, dt);

              ptc.p1[n] = context.p[0];
              ptc.p2[n] = context.p[1];
              ptc.p3[n] = context.p[2];
              ptc.E[n] = context.gamma;

              deposit_t<Conf::dim, typename Conf::spline_t> deposit{};
              deposit(context, J, Rho, idx, ext, dt, deposit_rho);

              ptc.x1[n] = context.new_x[0];
              ptc.x2[n] = context.new_x[1];
              ptc.x3[n] = context.new_x[2];
              // ptc.cell[n] = Conf::idx(pos, ext).linear;
              ptc.cell[n] = context.cell + context.dc.dot(ext.strides());
            },
            ptc, E, B, J, Rho, grid, phys_policy);
      },
      *ptc, *E, *B, *J, Rho);
  ExecPolicy<Conf>::sync();

  coord_policy.template process_J_Rho<ExecPolicy<Conf>>(*J, Rho, dt,
                                                        deposit_rho);

  filter_current(m_filter_times, step);
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater_new<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::update_particles(
    value_t dt, uint32_t step) {
  update_particles(dt, step, 0, ptc->number());
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater_new<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::update_photons(
    value_t dt, uint32_t step) {
  auto num = ph->number();
  if (num == 0) return;
  int rho_interval = m_rho_interval;
  auto coord_policy = *m_coord_policy;

  // Photon movement loop
  ExecPolicy<Conf>::launch(
      [num, dt, rho_interval, step, coord_policy] LAMBDA(auto ph, auto Rho_ph) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        bool deposit_rho = (step % rho_interval == 0);
        ExecPolicy<Conf>::loop(
            0ul, num,
            [&ext, &coord_policy, dt, deposit_rho] LAMBDA(
                auto n, auto& ph, auto& Rho_ph, auto& grid) {
              ph_context<Conf::dim, value_t> context;
              context.cell = ph.cell[n];
              if (context.cell == empty_cell) return;

              auto idx = Conf::idx(context.cell, ext);

              context.x = vec_t<value_t, 3>(ph.x1[n], ph.x2[n], ph.x3[n]);
              context.p = vec_t<value_t, 3>(ph.p1[n], ph.p2[n], ph.p3[n]);
              context.gamma = ph.E[n];

              context.flag = ph.flag[n];

              auto pos = get_pos(idx, ext);
              coord_policy.update_ph(grid, context, pos, dt);

              ph.p1[n] = context.p[0];
              ph.p2[n] = context.p[1];
              ph.p3[n] = context.p[2];
              // Photon enery should not change
              // ph.E[n] = context.gamma;

              auto idx_new = Conf::idx(pos + context.dc, ext);
              if (deposit_rho) {
                // Simple deposit, do not care about weight function
                deposit_add(&Rho_ph[idx_new], ph.weight[n]);
              }

              ph.x1[n] = context.new_x[0];
              ph.x2[n] = context.new_x[1];
              ph.x3[n] = context.new_x[2];
              ph.cell[n] = idx_new.linear;
              ph.path_left[n] -= dt;
            },
            ph, Rho_ph, grid);
      },
      *ph, *rho_ph);
  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater_new<Conf, ExecPolicy, CoordPolicy,
                PhysicsPolicy>::clear_guard_cells() {
  auto num = ptc->number();

  ExecPolicy<Conf>::launch(
      [num] LAMBDA(auto ptc) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        ExecPolicy<Conf>::loop(
            0ul, num,
            [ext] LAMBDA(auto n, auto& grid, auto& ptc) {
              auto cell = ptc.cell[n];
              if (cell == empty_cell) return;

              auto idx = Conf::idx(cell, ext);
              auto pos = get_pos(idx, ext);
              if (!grid.is_in_bound(pos)) ptc.cell[n] = empty_cell;
            },
            grid, ptc);
      },
      *ptc);

  if (ph != nullptr) {
    ExecPolicy<Conf>::launch(
        [num] LAMBDA(auto ph) {
          auto& grid = ExecPolicy<Conf>::grid();
          auto ext = grid.extent();
          ExecPolicy<Conf>::loop(
              0ul, num,
              [ext] LAMBDA(auto n, auto& grid, auto& ph) {
                auto cell = ph.cell[n];
                if (cell == empty_cell) return;

                auto idx = Conf::idx(cell, ext);
                auto pos = get_pos(idx, ext);
                if (!grid.is_in_bound(pos)) ph.cell[n] = empty_cell;
              },
              grid, ph);
        },
        *ph);
  }
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater_new<Conf, ExecPolicy, CoordPolicy,
                PhysicsPolicy>::sort_particles() {
  ptc->sort_by_cell(m_grid.extent().size());
  if (ph != nullptr) ph->sort_by_cell(m_grid.extent().size());
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater_new<Conf, ExecPolicy, CoordPolicy,
                PhysicsPolicy>::fill_multiplicity(int mult, value_t weight,
                                                  value_t dp) {
  // CoordPolicy<Conf>::template fill_multiplicity<ExecPolicy<Conf>>(
  //     *ptc, *rng_states, mult, weight);
  auto num = ptc->number();

  ExecPolicy<Conf>::launch(
      [num, mult, weight, dp] LAMBDA(auto ptc, auto states) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        rng_t rng(states);
        ExecPolicy<Conf>::loop(
            // 0, ext.size(),
            Conf::begin(ext), Conf::end(ext),
            [&grid, num, ext, mult, weight, dp] LAMBDA(
                auto idx, auto& ptc,
                // auto n, auto& ptc,
                // [&grid, num, ext, mult, weight] LAMBDA(auto n, auto& ptc,
                auto& rng) {
              // auto idx = Conf::idx(n, ext);
              auto pos = get_pos(idx, ext);
              if (grid.is_in_bound(pos)) {
                for (int i = 0; i < mult; i++) {
                  uint32_t offset = num + idx.linear * mult * 2 + i * 2;

                  ptc.x1[offset] = ptc.x1[offset + 1] =
                      rng.template uniform<value_t>();
                  ptc.x2[offset] = ptc.x2[offset + 1] =
                      rng.template uniform<value_t>();
                  ptc.x3[offset] = ptc.x3[offset + 1] =
                      rng.template uniform<value_t>();
                  value_t x1 = CoordPolicy<Conf>::x1(
                      grid.template pos<0>(pos[0], ptc.x1[offset]));
                  value_t x2 = CoordPolicy<Conf>::x2(
                      grid.template pos<1>(pos[1], ptc.x2[offset]));
                  value_t x3 = CoordPolicy<Conf>::x3(
                      grid.template pos<2>(pos[2], ptc.x3[offset]));

                  ptc.p1[offset] = ptc.p1[offset + 1] =
                      rng.template gaussian<value_t>(dp);
                  ptc.p2[offset] = ptc.p2[offset + 1] =
                      rng.template gaussian<value_t>(dp);
                  ptc.p3[offset] = ptc.p3[offset + 1] =
                      rng.template gaussian<value_t>(dp);
                  ptc.E[offset] = ptc.E[offset + 1] = 1.0;
                  ptc.cell[offset] = ptc.cell[offset + 1] = idx.linear;
                  ptc.weight[offset] = ptc.weight[offset + 1] =
                      weight * CoordPolicy<Conf>::weight_func(x1, x2, x3);
                  ptc.flag[offset] = set_ptc_type_flag(
                      flag_or(PtcFlag::primary), PtcType::electron);
                  ptc.flag[offset + 1] = set_ptc_type_flag(
                      flag_or(PtcFlag::primary), PtcType::positron);
                }
              }
            },
            ptc, rng);
      },
      *ptc, *rng_states);
  ExecPolicy<Conf>::sync();
  ptc->set_num(ptc->number() + 2 * mult * m_grid.extent().size());

  ptc->sort_by_cell(m_grid.extent().size());
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater_new<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::filter_current(
    int num_times, uint32_t step) {
  if (num_times <= 0) return;

  // filter_field<ExecPolicy<Conf>>(J->at(0), m_tmpj, )
  vec_t<bool, Conf::dim * 2> is_boundary;
  is_boundary.set(true);
  if (m_comm != nullptr) is_boundary = m_comm->domain_info().is_boundary;

  for (int i = 0; i < num_times; i++) {
    // filter_field<ExecPolicy<Conf>>(J->at(0), m_tmpj, is_boundary);
    // filter_field<ExecPolicy<Conf>>(J->at(1), m_tmpj, is_boundary);
    // filter_field<ExecPolicy<Conf>>(J->at(2), m_tmpj, is_boundary);
    m_coord_policy->template filter_field<ExecPolicy<Conf>>(*J, m_tmpj,
                                                            is_boundary);

    if (m_comm != nullptr) m_comm->send_guard_cells(*J);

    if (step % m_rho_interval == 0) {
      for (int sp = 0; sp < m_num_species; sp++) {
        // filter_field<ExecPolicy<Conf>>(Rho[sp]->at(0), m_tmpj, is_boundary);
        m_coord_policy->template filter_field<ExecPolicy<Conf>>(
            *Rho[sp], m_tmpj, is_boundary);
        if (m_comm != nullptr) m_comm->send_guard_cells(*Rho[sp]);
      }
    }
  }
}

}  // namespace Aperture

#endif  // __PTC_UPDATER_BASE_IMPL_H_
