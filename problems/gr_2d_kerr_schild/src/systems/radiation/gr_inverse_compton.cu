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

#include "core/math.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "gr_inverse_compton.h"
#include "systems/grid_ks.h"
#include "systems/inverse_compton.h"
#include "systems/physics/metric_kerr_schild.hpp"
#include "systems/physics/spectra.hpp"

namespace Aperture {

template <typename Conf>
void
gr_ic_radiation_scheme<Conf>::init() {
  value_t emin = 1.0e-5;
  sim_env().params().get_value("emin", emin);
  value_t ic_path = 1.0;
  sim_env().params().get_value("ic_path", ic_path);
  sim_env().params().get_value("bh_spin", m_a);

  // Configure the spectrum here and initialize the ic module
  Spectra::broken_power_law spec(1.25, 1.1, emin, 1.0e-10, 0.1);

  auto ic = sim_env().register_system<inverse_compton_t>();
  ic->compute_coefficients(spec, spec.emin(), spec.emax(), 1.5e24 / ic_path);

  m_ic_module = ic->get_ic_module();
}

template <typename Conf>
HOST_DEVICE bool
gr_ic_radiation_scheme<Conf>::check_emit_photon(ptc_ptrs &ptc, size_t tid,
                                                rng_t &rng) {
  // First obtain the particle Lorentz factor in the correct frame
  // value_t r = grid_ks_t<Conf>::radius(value_t x1)
  // value_t u_0 = Metric_KS::u_0
  return true;
}

template <typename Conf>
HOST_DEVICE void
gr_ic_radiation_scheme<Conf>::emit_photon(ptc_ptrs &ptc, size_t tid,
                                          ph_ptrs &ph, size_t offset,
                                          rng_t &rng) {}

template <typename Conf>
HOST_DEVICE bool
gr_ic_radiation_scheme<Conf>::check_produce_pair(ph_ptrs &ph, size_t tid,
                                                 rng_t &rng) {
  return true;
}

template <typename Conf>
HOST_DEVICE void
gr_ic_radiation_scheme<Conf>::produce_pair(ph_ptrs &ph, size_t tid,
                                           ptc_ptrs &ptc, size_t offset,
                                           rng_t &rng) {}

template class gr_ic_radiation_scheme<Config<2>>;

}  // namespace Aperture
