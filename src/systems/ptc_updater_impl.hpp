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

#pragma once

#include "data/rng_states.h"
#include "framework/environment.h"
#include "systems/helpers/filter_field.hpp"
#include "systems/helpers/ptc_update_helper.hpp"
#include "systems/ptc_updater.h"
#include "utils/interpolation.hpp"
#include "utils/range.hpp"
#include "utils/timer.h"
#include "utils/type_traits.hpp"

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::ptc_updater(
    const grid_t<Conf> &grid)
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
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::ptc_updater(
    const grid_t<Conf> &grid, const domain_comm<Conf, ExecPolicy> *comm)
    : ptc_updater(grid) {
  m_comm = comm;
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::~ptc_updater() =
    default;

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

  // #ifdef CUDA_ENABLED
  //   init_dev_charge_mass(this->m_charges.data(), this->m_masses.data());
  // #endif
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::init() {
  sim_env().get_data_optional("photons", ph);
  sim_env().get_data_optional("Rho_ph", rho_ph);

  m_coord_policy->init();
  m_phys_policy->init();
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy,
            PhysicsPolicy>::register_data_components() {
  size_t max_ptc_num = 10000;
  sim_env().params().get_value("max_ptc_num", max_ptc_num);
  int segment_size = max_ptc_num;
  sim_env().params().get_value("ptc_segment_size", segment_size);

  ptc = sim_env().register_data<particle_data_t>(
      "particles", max_ptc_num, ExecPolicy<Conf>::tmp_mem_type());
  ptc->include_in_snapshot(true);
  if (segment_size > max_ptc_num) segment_size = max_ptc_num;
  ptc->set_segment_size(segment_size);

  E = sim_env().register_data<vector_field<Conf>>(
      "E", m_grid, field_type::edge_centered,
      ExecPolicy<Conf>::data_mem_type());
  // E->include_in_snapshot(true);
  B = sim_env().register_data<vector_field<Conf>>(
      "B", m_grid, field_type::face_centered,
      ExecPolicy<Conf>::data_mem_type());
  // B->include_in_snapshot(true);
  J = sim_env().register_data<vector_field<Conf>>(
      "J", m_grid, field_type::edge_centered,
      ExecPolicy<Conf>::data_mem_type());
  J->include_in_snapshot(true);

  rho_total = sim_env().register_data<scalar_field<Conf>>(
    "Rho_total", m_grid, field_type::vert_centered,
    ExecPolicy<Conf>::data_mem_type());
  // rho_total->include_in_snapshot(true);

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
    Rho[i]->include_in_snapshot(true);
  }
  Rho.copy_to_device();

  size_t seed = default_random_seed;
  sim_env().params().get_value("random_seed", seed);
  rng_states =
      sim_env()
          .register_data<rng_states_t<typename ExecPolicy<Conf>::exec_tag>>(
              "rng_states", seed);
  rng_states->skip_output(true);
  rng_states->include_in_snapshot(true);

  extent_t<Conf::dim> domain_ext;
  domain_ext.set(1);
  if (m_comm != nullptr) {
    for (int i = 0; i < Conf::dim; i++) {
      domain_ext[i] = m_comm->domain_info().mpi_dims[i];
    }
  }
  ptc_number = sim_env().register_data<multi_array_data<uint32_t, Conf::dim>>(
      "ptc_number", domain_ext, MemType::host_only);
  ptc_number->gather_to_root = false;
}

CREATE_MEMBER_FUNC_CALL(update);

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::update(
    double dt, uint32_t step) {
  Logger::print_detail("Updating {} particles", ptc->number());
  // timer::stamp();
  update_particles(dt, step);
  ExecPolicy<Conf>::sync();
  // timer::show_duration_since_stamp("update", "ms");

  // if (traits::has_update<CoordPolicy<Conf>,
  //                        void(const ExecPolicy<Conf> &)>::value) {
  //   m_coord_policy->update(ExecPolicy<Conf>{});
  // }
  // if (traits::has_update<PhysicsPolicy<Conf>,
  //                        void(const ExecPolicy<Conf> &)>::value) {
  //   m_phys_policy->update(ExecPolicy<Conf>{});
  // }
  traits::call_update_if_exists(*m_coord_policy, ExecPolicy<Conf>{});
  traits::call_update_if_exists(*m_phys_policy, ExecPolicy<Conf>{});

  // timer::stamp();
  // Communicate deposited current and charge densities
  if (m_comm != nullptr) {
    m_comm->send_add_guard_cells(*J);
    m_comm->send_guard_cells(*J);
    m_comm->send_add_guard_cells(*rho_total);
    m_comm->send_guard_cells(*rho_total);
    // if ((step + 1) % m_data_interval == 0) {
    if (step % m_rho_interval == 0) {
      for (uint32_t i = 0; i < Rho.size(); i++) {
        m_comm->send_add_guard_cells(*(Rho[i]));
        m_comm->send_guard_cells(*(Rho[i]));
      }
    }
  }
  ExecPolicy<Conf>::sync();
  // timer::show_duration_since_stamp("commJRho", "ms");

  // timer::stamp();
  filter_current(m_filter_times, step);
  // timer::show_duration_since_stamp("filter_current", "ms");
  Logger::print_detail("Finished filtering current");

  // Send particles
  // timer::stamp();
  if (m_comm != nullptr) {
    m_comm->send_particles(*ptc, m_grid);
  }
  ExecPolicy<Conf>::sync();
  // timer::show_duration_since_stamp("send_particles", "ms");

  Logger::print_detail("Finished sending particles");

  // Also move photons if the data component exists
  if (ph != nullptr) {
    Logger::print_detail("Moving {} photons", ph->number());
    update_photons(dt, step);

    if (m_comm != nullptr) {
      m_comm->send_particles(*ph, m_grid);
    }
    if (step % m_rho_interval == 0) {
      m_comm->send_add_guard_cells(*rho_ph);
      m_comm->send_guard_cells(*rho_ph);
    }
  }

  // Clear guard cells
  clear_guard_cells();

  Logger::print_detail("Finished clearing guard cells");

  // sort at the given interval. Turn off sorting if m_sort_interval is 0
  if (m_sort_interval > 0 && (step % m_sort_interval) == 0) {
    sort_particles();
    tally_ptc_number(*ptc);
    Logger::print_detail("Finished sorting");
    if (ph != nullptr) {
      tally_ptc_number(*ph);
    }
  }
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::update_particles(
    value_t dt, uint32_t step, size_t begin, size_t end) {
  J->init();
  rho_total->init();
  for (int i = 0; i < Rho.size(); i++) {
    Rho[i]->init();
  }

  if (end - begin <= 0) return;
  int rho_interval = m_rho_interval;
  auto charges = m_charges;
  auto masses = m_masses;
  auto coord_policy = *m_coord_policy;
  auto phys_policy = *m_phys_policy;
  bool deposit_rho = (step % rho_interval == 0);

  // timer::stamp("ptc");
  // Main particle update loop
  ExecPolicy<Conf>::launch(
      [begin, end, dt, rho_interval, deposit_rho, charges, masses, coord_policy,
       phys_policy] LAMBDA(auto ptc, auto E, auto B, auto J, auto Rho,
                           auto rho_total, auto states) {
        auto &grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        rng_t<typename ExecPolicy<Conf>::exec_tag> rng(states);
        auto interp = interpolator<typename Conf::spline_t, Conf::dim>{};
        // auto interp = interp_t<1, Conf::dim>{};
        ExecPolicy<Conf>::loop(begin, end, [&] LAMBDA(auto n) {
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
          context.local_state = &rng.m_local_state;
          context.aux1 = ptc.aux1[n];  // Auxiliary variable, can be used for
                                       // different quantities

          // context.E[0] = interp(E[0], context.x, idx, stagger_t(0b110));
          // context.E[1] = interp(E[1], context.x, idx, stagger_t(0b101));
          // context.E[2] = interp(E[2], context.x, idx, stagger_t(0b011));
          // context.B[0] = interp(B[0], context.x, idx, stagger_t(0b001));
          // context.B[1] = interp(B[1], context.x, idx, stagger_t(0b010));
          // context.B[2] = interp(B[2], context.x, idx, stagger_t(0b100));
          context.E[0] = interp(context.x, E[0], idx, ext, stagger_t(0b110));
          context.E[1] = interp(context.x, E[1], idx, ext, stagger_t(0b101));
          context.E[2] = interp(context.x, E[2], idx, ext, stagger_t(0b011));
          context.B[0] = interp(context.x, B[0], idx, ext, stagger_t(0b001));
          context.B[1] = interp(context.x, B[1], idx, ext, stagger_t(0b010));
          context.B[2] = interp(context.x, B[2], idx, ext, stagger_t(0b100));

          // printf("x1: %f, x2: %f, p1: %f, p2: %f, q_over_m: %f, dt:
          // %f\n",
          //        context.x[0], context.x[1], context.p[0], context.p[1],
          //        charges[context.sp] / masses[context.sp], dt);

          auto pos = get_pos(idx, ext);
          coord_policy.update_ptc(grid, ext, context, pos, dt);

          phys_policy(grid, context, pos, dt);

          ptc.p1[n] = context.p[0];
          ptc.p2[n] = context.p[1];
          ptc.p3[n] = context.p[2];
          ptc.E[n] = context.gamma;

          if (!check_flag(context.flag, PtcFlag::ignore_current) &&
              !check_flag(context.flag, PtcFlag::test_particle)) {
            deposit_t<Conf::dim, typename Conf::spline_t> deposit{};
            deposit(context, J, Rho, rho_total, idx, ext, dt, deposit_rho);
          }

          ptc.x1[n] = context.new_x[0];
          ptc.x2[n] = context.new_x[1];
          ptc.x3[n] = context.new_x[2];
          ptc.cell[n] = Conf::idx(pos, ext).linear;
          // ptc.cell[n] = context.cell + context.dc.dot(ext.strides());
        });
        // ptc, E, B, J, Rho, grid, phys_policy);
      },
      *ptc, *E, *B, *J, Rho, *rho_total, *rng_states);
  ExecPolicy<Conf>::sync();
  // timer::show_duration_since_stamp("ptc update loop", "ms", "ptc");

  // timer::stamp("process");
  coord_policy.template process_J_Rho<ExecPolicy<Conf>>(*J, Rho, *rho_total, dt,
                                                        deposit_rho);
  // timer::show_duration_since_stamp("ptc update process J rho", "ms",
  // "process");
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::update_particles(
    value_t dt, uint32_t step) {
  update_particles(dt, step, 0, ptc->number());
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
  auto coord_policy = *m_coord_policy;
  bool deposit_rho = (step % rho_interval == 0);

  rho_ph->init();

  // Photon movement loop
  ExecPolicy<Conf>::launch(
      [num, dt, rho_interval, deposit_rho, coord_policy] LAMBDA(auto ph,
                                                                auto Rho_ph) {
        auto &grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        ExecPolicy<Conf>::loop(0ul, num, [&] LAMBDA(auto n) {
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
          ph.E[n] = context.gamma;

          // auto idx_new = Conf::idx(pos + context.dc, ext);
          auto idx_new = Conf::idx(pos, ext);
          if (deposit_rho) {
            // Simple deposit, do not care about weight function
            atomic_add(&Rho_ph[idx_new], ph.weight[n]);
          }

          ph.x1[n] = context.new_x[0];
          ph.x2[n] = context.new_x[1];
          ph.x3[n] = context.new_x[2];
          ph.cell[n] = idx_new.linear;
          ph.path_left[n] -= dt;
        });
        // ph, Rho_ph, grid);
      },
      *ph, *rho_ph);
  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::clear_guard_cells() {
  auto num = ptc->number();
  ExecPolicy<Conf>::launch(
      [num] LAMBDA(auto ptc) {
        auto &grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        ExecPolicy<Conf>::loop(0, num, [&] LAMBDA(auto n) {
          auto cell = ptc.cell[n];
          if (cell == empty_cell) return;

          auto idx = Conf::idx(cell, ext);
          auto pos = get_pos(idx, ext);
          if (!grid.is_in_bound(pos)) ptc.cell[n] = empty_cell;
        });
        // grid, ptc);
      },
      *ptc);

  if (ph != nullptr) {
    auto ph_num = ph->number();
    ExecPolicy<Conf>::launch(
        [ph_num] LAMBDA(auto ph) {
          auto &grid = ExecPolicy<Conf>::grid();
          auto ext = grid.extent();
          ExecPolicy<Conf>::loop(0, ph_num, [&] LAMBDA(auto n) {
            auto cell = ph.cell[n];
            if (cell == empty_cell) return;

            auto idx = Conf::idx(cell, ext);
            auto pos = get_pos(idx, ext);
            if (!grid.is_in_bound(pos)) ph.cell[n] = empty_cell;
          });
          // grid, ph);
        },
        *ph);
  }
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::sort_particles() {
  // ptc->sort_by_cell(typename ExecPolicy<Conf>::exec_tag{},
  // m_grid.extent().size());
  ptc_sort_by_cell(typename ExecPolicy<Conf>::exec_tag{}, *ptc,
                   m_grid.extent().size());
  Logger::print_debug("Sorting complete, there are {} particles in the pool",
                     ptc->number());
  Logger::print_detail_all("There are {} particles in the pool", ptc->number());
  if (ph != nullptr) {
    // ph->sort_by_cell(typename ExecPolicy<Conf>::exec_tag{},
    // m_grid.extent().size());
    ptc_sort_by_cell(typename ExecPolicy<Conf>::exec_tag{}, *ph,
                     m_grid.extent().size());
    Logger::print_debug("Sorting complete, there are {} photons in the pool",
                       ph->number());
    Logger::print_detail_all("There are {} photons in the pool", ph->number());
  }
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
template <typename PtcType>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::tally_ptc_number(
    particles_base<PtcType> &ptc) {
  // Tally particle number of this rank and store it in ptc_number
  size_t total_num = 0, max_num = 0;
  if (m_comm != nullptr && m_comm->size() > 1) {
    extent_t<Conf::dim> domain_ext = ptc_number->extent();
    for (auto idx : range(Conf::begin(domain_ext), Conf::end(domain_ext))) {
      index_t<Conf::dim> pos = get_pos(idx, domain_ext);
      index_t<Conf::dim> mpi_coord(m_comm->domain_info().mpi_coord);
      if (mpi_coord == pos) {
        (*ptc_number)[idx] = ptc.number();
      } else {
        (*ptc_number)[idx] = 0;
      }
    }
    m_comm->gather_to_root(*ptc_number);
    for (auto idx : range(Conf::begin(domain_ext), Conf::end(domain_ext))) {
      if ((*ptc_number)[idx] > max_num) {
        max_num = (*ptc_number)[idx];
      }
      total_num += (*ptc_number)[idx];
    }
  } else {
    total_num = max_num = (*ptc_number)[0] = ptc.number();
  }
  if (std::is_same<PtcType, ptc_buffer>::value) {
    Logger::print_info("Total ptc number: {}, max ptc number on a rank: {}",
                        total_num, max_num);
  } else if (std::is_same<PtcType, ph_buffer>::value) {
    Logger::print_info("Total ph number: {}, max ph number on a rank: {}",
                        total_num, max_num);
  }
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::fill_multiplicity(
    int mult, value_t weight, value_t kT) {
  auto num = ptc->number();

  ExecPolicy<Conf>::launch(
      [num, mult, weight, kT] LAMBDA(auto ptc, auto states) {
        auto &grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        rng_t<typename ExecPolicy<Conf>::exec_tag> rng(states);
        ExecPolicy<Conf>::loop(
            // 0, ext.size(),
            Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
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
                      grid.template coord<0>(pos[0], ptc.x1[offset]));
                  value_t x2 = CoordPolicy<Conf>::x2(
                      grid.template coord<1>(pos[1], ptc.x2[offset]));
                  value_t x3 = CoordPolicy<Conf>::x3(
                      grid.template coord<2>(pos[2], ptc.x3[offset]));

                  auto p = rng.maxwell_juttner_3d(kT);
                  ptc.p1[offset] = p[0];
                  ptc.p2[offset] = p[1];
                  ptc.p3[offset] = p[2];
                  ptc.E[offset] = math::sqrt(1.0f + p.dot(p));
                  p = rng.maxwell_juttner_3d(kT);
                  ptc.p1[offset + 1] = p[0];
                  ptc.p2[offset + 1] = p[1];
                  ptc.p3[offset + 1] = p[2];
                  ptc.E[offset + 1] = math::sqrt(1.0f + p.dot(p));
                  ptc.cell[offset] = ptc.cell[offset + 1] = idx.linear;
                  ptc.weight[offset] = ptc.weight[offset + 1] =
                      weight * CoordPolicy<Conf>::weight_func(x1, x2, x3);
                  ptc.flag[offset] = set_ptc_type_flag(
                      flag_or(PtcFlag::primary), PtcType::electron);
                  ptc.flag[offset + 1] = set_ptc_type_flag(
                      flag_or(PtcFlag::primary), PtcType::positron);
                }
              }
            });
        // ptc, rng);
      },
      *ptc, *rng_states);
  ExecPolicy<Conf>::sync();
  ptc->set_num(ptc->number() + 2 * mult * m_grid.extent().size());

  // ptc->sort_by_cell(typename ExecPolicy<Conf>::exec_tag{},
  // m_grid.extent().size());
  ptc_sort_by_cell(typename ExecPolicy<Conf>::exec_tag{}, *ptc,
                   m_grid.extent().size());
}

template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater<Conf, ExecPolicy, CoordPolicy, PhysicsPolicy>::filter_current(
    int num_times, uint32_t step) {
  if (num_times <= 0) return;

  vec_t<bool, Conf::dim * 2> is_boundary;
  is_boundary.set(true);
  if (m_comm != nullptr) is_boundary = m_comm->domain_info().is_boundary;

  for (int i = 0; i < num_times; i++) {
    m_coord_policy->template filter_field<ExecPolicy<Conf>>(*J, m_tmpj,
                                                            is_boundary);

    if (m_comm != nullptr) {
      m_comm->send_guard_cells(*J);
    }

    m_coord_policy->template filter_field<ExecPolicy<Conf>>(
        *rho_total, m_tmpj, is_boundary);
    if (m_comm != nullptr) {
      m_comm->send_guard_cells(*rho_total);
    }
        
    if (step % m_rho_interval == 0) {
      for (int sp = 0; sp < m_num_species; sp++) {
        m_coord_policy->template filter_field<ExecPolicy<Conf>>(
            *Rho[sp], m_tmpj, is_boundary);
        if (m_comm != nullptr) {
          m_comm->send_guard_cells(*Rho[sp]);
        }
      }
    }
  }
}

}  // namespace Aperture
