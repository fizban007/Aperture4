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

#ifndef __RADIATIVE_TRANSFER_NEW_IMPL_H_
#define __RADIATIVE_TRANSFER_NEW_IMPL_H_

#include "framework/environment.h"
#include "framework/params_store.h"
#include "radiative_transfer_new.h"

namespace Aperture {

template <typename Conf>
radiative_transfer_base<Conf>::radiative_transfer_base(
    const grid_t<Conf>& grid, const domain_comm<Conf>* comm)
    : m_grid(grid), m_comm(comm) {
  if (comm != nullptr) {
    m_track_rank = comm->rank();
    m_track_rank <<= 32;
  }
}

template <typename Conf>
radiative_transfer_base<Conf>::~radiative_transfer_base() = default;

template <typename Conf>
void
radiative_transfer_base<Conf>::init() {
  sim_env().params().get_value("fld_output_interval", m_data_interval);
  sim_env().params().get_value("sort_interval", m_sort_interval);
  sim_env().params().get_value("ph_per_scatter", m_ph_per_scatter);
  sim_env().params().get_value("tracked_fraction", m_tracked_fraction);

  sim_env().get_data("particles", ptc);
  sim_env().get_data("rng_states", rng_states);
}

template <typename Conf>
void
radiative_transfer_base<Conf>::update(double dt, uint32_t step) {
  produce_pairs(dt);
  emit_photons(dt);
}

template <class Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class RadiationPolicy>
radiative_transfer<Conf, ExecPolicy, CoordPolicy, RadiationPolicy>::
    radiative_transfer(const grid_t<Conf>& grid, const domain_comm<Conf>* comm)
    : radiative_transfer_base<Conf>(grid, comm) {
  m_coord_policy = std::make_unique<CoordPolicy<Conf>>(grid);
  m_rad_policy = std::make_unique<RadiationPolicy<Conf>>(grid);
}

template <class Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class RadiationPolicy>
radiative_transfer<Conf, ExecPolicy, CoordPolicy,
                   RadiationPolicy>::~radiative_transfer() {}

template <class Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class RadiationPolicy>
void
radiative_transfer<Conf, ExecPolicy, CoordPolicy,
                   RadiationPolicy>::init() {
  m_rad_policy->init();
}

template <class Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class RadiationPolicy>
void
radiative_transfer<Conf, ExecPolicy, CoordPolicy,
                   RadiationPolicy>::register_data_components() {
  size_t max_ph_num = 10000;
  sim_env().params().get_value("max_ph_num", max_ph_num);

  this->ph = sim_env().template register_data<photon_data_t>(
      "photons", max_ph_num, MemType::host_only);
  this->rho_ph = sim_env().template register_data<scalar_field<Conf>>(
      "Rho_ph", this->m_grid, field_type::vert_centered, MemType::host_only);
  this->photon_produced = sim_env().template register_data<scalar_field<Conf>>(
      "photon_produced", this->m_grid, field_type::vert_centered,
      MemType::host_only);
  this->pair_produced = sim_env().template register_data<scalar_field<Conf>>(
      "pair_produced", this->m_grid, field_type::vert_centered,
      MemType::host_only);
  this->photon_produced->reset_after_output(true);
  this->pair_produced->reset_after_output(true);
}

template <class Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class RadiationPolicy>
void
radiative_transfer<Conf, ExecPolicy, CoordPolicy,
                   RadiationPolicy>::emit_photons(double dt) {}

template <class Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class RadiationPolicy>
void
radiative_transfer<Conf, ExecPolicy, CoordPolicy,
                   RadiationPolicy>::produce_pairs(double dt) {}

}  // namespace Aperture

#endif  // __RADIATIVE_TRANSFER_NEW_IMPL_H_
