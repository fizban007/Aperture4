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

#ifndef _PTC_UPDATER_H_
#define _PTC_UPDATER_H_

#include "core/enum_types.h"
#include "data/particle_data.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/domain_comm.h"
#include "systems/physics/pushers.hpp"
#include "systems/grid.h"
#include "utils/interpolation.hpp"
#include <array>

namespace Aperture {

class curand_states_t;

template <typename Conf>
class ptc_updater : public system_t {
 protected:
  const grid_t<Conf>& m_grid;
  const domain_comm<Conf>* m_comm = nullptr;

  Pusher m_pusher = Pusher::higuera;
  typedef typename Conf::spline_t spline_t;

  particle_data_t* ptc;
  photon_data_t* ph;
  vector_field<Conf> *E, *B, *J;
  std::vector<scalar_field<Conf>*> Rho;
  scalar_field<Conf>* rho_ph;

  std::unique_ptr<typename Conf::multi_array_t> jtmp;

  // Parameters for this module
  uint32_t m_num_species = 2;
  uint32_t m_data_interval = 1;
  uint32_t m_rho_interval = 1;
  uint32_t m_sort_interval = 20;
  uint32_t m_filter_times = 1;

  // By default the maximum number of species is 8
  // float m_charges[max_ptc_types];
  // float m_masses[max_ptc_types];
  // float m_q_over_m[max_ptc_types];
  std::array<float, max_ptc_types> m_charges;
  std::array<float, max_ptc_types> m_masses;
  std::array<float, max_ptc_types> m_q_over_m;

  void init_charge_mass();

 public:
  typedef typename Conf::value_t value_t;
  static std::string name() { return "ptc_updater"; }

  ptc_updater(sim_environment& env, const grid_t<Conf>& grid,
              const domain_comm<Conf>* comm = nullptr);

  void init() override;
  void update(double dt, uint32_t step) override;
  void register_data_components() override;

  void move_and_deposit(double dt, uint32_t step);
  void move_photons(double dt, uint32_t step);
  void filter_current(int n_times, uint32_t step);

  template <typename P>
  void push(double dt, P& pusher);
  // void move(double dt);

  virtual void push_default(double dt);
  virtual void move_deposit_1d(value_t dt, uint32_t step);
  virtual void move_deposit_2d(value_t dt, uint32_t step);
  virtual void move_deposit_3d(value_t dt, uint32_t step);
  virtual void move_photons_1d(value_t dt, uint32_t step);
  virtual void move_photons_2d(value_t dt, uint32_t step);
  virtual void move_photons_3d(value_t dt, uint32_t step);
  virtual void clear_guard_cells();
  virtual void sort_particles();
  virtual void filter_field(vector_field<Conf>& f, int comp);
  virtual void filter_field(scalar_field<Conf>& f);
  virtual void fill_multiplicity(int n, value_t weight = 1.0);

  void use_pusher(Pusher p) {
    m_pusher = p;
  }
};

template <typename Conf>
class ptc_updater_cu : public ptc_updater<Conf> {
 public:
  using value_t = typename Conf::value_t;
  using rho_ptrs_t = buffer<ndptr<value_t, Conf::dim>>;
  using base_class = ptc_updater<Conf>;

  static std::string name() { return "ptc_updater"; }


  using base_class::base_class;

  void init() override;
  // void update(double dt, uint32_t step);
  void register_data_components() override;

  template <typename P>
  void push(double dt, P& pusher);
  // void move_and_deposit(double dt, uint32_t step);

  // Need to override this because we can't make push virtual
  void push_default(double dt) override;
  void move_deposit_1d(value_t dt, uint32_t step) override;
  void move_deposit_2d(value_t dt, uint32_t step) override;
  void move_deposit_3d(value_t dt, uint32_t step) override;
  void move_photons_1d(value_t dt, uint32_t step) override;
  void move_photons_2d(value_t dt, uint32_t step) override;
  void move_photons_3d(value_t dt, uint32_t step) override;
  void clear_guard_cells() override;
  void sort_particles() override;
  void filter_field(vector_field<Conf>& f, int comp) override;
  void filter_field(scalar_field<Conf>& f) override;
  void fill_multiplicity(int n, value_t weight = 1.0) override;

  rho_ptrs_t& get_rho_ptrs() { return m_rho_ptrs; }

 protected:
  rho_ptrs_t m_rho_ptrs;
  curand_states_t* m_rand_states;
};

}  // namespace Aperture

#endif  // _PTC_UPDATER_H_
