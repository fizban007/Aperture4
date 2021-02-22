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

#ifndef _RADIATIVE_TRANSFER_H_
#define _RADIATIVE_TRANSFER_H_

#include "data/fields.h"
#include "data/particle_data.h"
#include "data/rng_states.h"
#include "framework/system.h"
#include "systems/domain_comm.h"
#include "systems/grid.h"
#include "systems/policies.h"
#include "utils/nonown_ptr.hpp"
#include <memory>

namespace Aperture {

template <class Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy,
          template <class> class RadiationPolicy>
class radiative_transfer : public system_t {
 public:
  using value_t = typename Conf::value_t;
  static std::string name() { return "radiative_transfer"; }

  radiative_transfer(const grid_t<Conf>& grid,
                     const domain_comm<Conf>* comm = nullptr);
  ~radiative_transfer();

  virtual void init() override;
  virtual void register_data_components() override;
  virtual void update(double dt, uint32_t step) override;

  void emit_photons(value_t dt);
  void create_pairs(value_t dt);

 protected:
  std::unique_ptr<CoordPolicy<Conf>> m_coord_policy;
  std::unique_ptr<RadiationPolicy<Conf>> m_rad_policy;

  const grid_t<Conf>& m_grid;
  const domain_comm<Conf>* m_comm = nullptr;

  // associated data components
  nonown_ptr<photon_data_t> ph;
  nonown_ptr<scalar_field<Conf>> rho_ph;
  nonown_ptr<scalar_field<Conf>> photon_produced;
  nonown_ptr<scalar_field<Conf>> pair_produced;

  // data components managed by other systems
  nonown_ptr<particle_data_t> ptc;
  nonown_ptr<rng_states_t> rng_states;

  // parameters for this module
  uint32_t m_data_interval = 1;
  uint32_t m_sort_interval = 20;
  int m_ph_per_scatter = 1;
  float m_tracked_fraction = 0.01;
  uint64_t m_track_rank = 0;

};

}


#endif  // _RADIATIVE_TRANSFER_H_
