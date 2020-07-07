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

#include "core/math.hpp"
#include "core/particle_structs.h"
#include "framework/config.h"
#include "systems/radiation/photon_pair_creation.hpp"
#include "systems/radiation/threshold_emission.hpp"
#include "systems/radiation/fixed_photon_path.hpp"
#include "systems/radiative_transfer_cu_impl.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
struct ph_freepath_dev_t
    : public threshold_emission_t,
      public fixed_photon_path<Conf>,
      public photon_pair_creation_t<typename Conf::value_t> {
  typedef typename Conf::value_t value_t;

  // float gamma_thr = 30.0f;
  // float E_s = 2.0f;
  // float photon_path = 0.0f;

  HOST_DEVICE ph_freepath_dev_t() {}
  ph_freepath_dev_t(sim_environment& env) {
    env.params().get_value("gamma_thr", this->gamma_thr);
    env.params().get_value("E_s", this->E_s);
    env.params().get_value("photon_path", this->photon_path);
  }
  HOST_DEVICE ph_freepath_dev_t(const ph_freepath_dev_t& other) = default;

};

template class radiative_transfer_cu<Config<2>, ph_freepath_dev_t<Config<2>>>;

}  // namespace Aperture
