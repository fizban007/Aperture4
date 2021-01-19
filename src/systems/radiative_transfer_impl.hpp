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

#ifndef _RADIATIVE_TRANSFER_IMPL_H_
#define _RADIATIVE_TRANSFER_IMPL_H_

#include "radiative_transfer.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "framework/params_store.h"
#include <exception>

namespace Aperture {

template <typename Conf>
radiative_transfer_common<Conf>::radiative_transfer_common(
    sim_environment& env, const grid_t<Conf>& grid,
    const domain_comm<Conf>* comm)
    : system_t(env), m_grid(grid), m_comm(comm) {
  m_env.params().get_value("fld_output_interval", m_data_interval);
  m_env.params().get_value("sort_interval", m_sort_interval);
  m_env.params().get_value("ph_per_scatter", m_ph_per_scatter);
  m_env.params().get_value("tracked_fraction", m_tracked_fraction);
}

template <typename Conf>
void
radiative_transfer_common<Conf>::init() {
  m_env.get_data("particles", ptc);
}

template <typename Conf>
void
radiative_transfer_common<Conf>::update(double dt, uint32_t step) {
  produce_pairs(dt);
  emit_photons(dt);
}

template <typename Conf, typename RadImpl>
radiative_transfer<Conf, RadImpl>::radiative_transfer(
    sim_environment& env, const grid_t<Conf>& grid,
    const domain_comm<Conf>* comm)
    : radiative_transfer_common<Conf>(env, grid, comm) {
  m_rad = std::make_unique<RadImpl>(env);
}

template <typename Conf, typename RadImpl>
radiative_transfer<Conf, RadImpl>::~radiative_transfer() {}

// template <typename Conf, typename RadImpl>
// void
// radiative_transfer<Conf, RadImpl>::init() {}

template <typename Conf, typename RadImpl>
void
radiative_transfer<Conf, RadImpl>::register_data_components() {
  size_t max_ph_num = 10000;
  this->m_env.params().get_value("max_ph_num", max_ph_num);

  this->ph = this->m_env.template register_data<photon_data_t>(
      "photons", max_ph_num, MemType::host_only);
  this->rho_ph = this->m_env.template register_data<scalar_field<Conf>>(
      "Rho_ph", this->m_grid, field_type::vert_centered, MemType::host_only);
  this->photon_produced =
      this->m_env.template register_data<scalar_field<Conf>>(
          "photon_produced", this->m_grid, field_type::vert_centered,
          MemType::host_only);
  this->pair_produced = this->m_env.template register_data<scalar_field<Conf>>(
      "pair_produced", this->m_grid, field_type::vert_centered,
      MemType::host_only);
}

template <typename Conf, typename RadImpl>
void
radiative_transfer<Conf, RadImpl>::emit_photons(double dt) {}

template <typename Conf, typename RadImpl>
void
radiative_transfer<Conf, RadImpl>::produce_pairs(double dt) {}

// template class radiative_transfer<Config<1>>;
// template class radiative_transfer<Config<2>>;
// template class radiative_transfer<Config<3>>;

}  // namespace Aperture



#endif  // _RADIATIVE_TRANSFER_IMPL_H_
