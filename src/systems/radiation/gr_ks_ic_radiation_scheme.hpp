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

#ifndef _GR_KS_IC_RADIATION_SCHEME_H_
#define _GR_KS_IC_RADIATION_SCHEME_H_

#include "core/cuda_control.h"
#include "core/particle_structs.h"
#include "core/random.h"
#include "framework/environment.h"
#include "systems/grid.h"
#include "systems/inverse_compton.h"
#include "systems/physics/ic_scattering.hpp"
#include "systems/physics/metric_kerr_schild.hpp"
#include "systems/physics/spectra.hpp"

namespace Aperture {

template <typename Conf>
struct gr_ks_ic_radiation_scheme {
  using value_t = typename Conf::value_t;

  const grid_t<Conf>& m_grid;
  ic_scatter_t m_ic_module;
  value_t m_a = 0.99;

  gr_ks_ic_radiation_scheme(const grid_t<Conf>& grid) : m_grid(grid) {}

  void init() {
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

  HOST_DEVICE size_t emit_photon(ptc_ptrs& ptc, size_t tid, ph_ptrs& ph,
                                 size_t ph_num, unsigned long long int* ph_pos,
                                 rng_t& rng) {
    return 0;
  }

  HOST_DEVICE size_t produce_pair(ph_ptrs& ph, size_t tid, ptc_ptrs& ptc,
                                  size_t ptc_num,
                                  unsigned long long int* ptc_pos, rng_t& rng) {
    return 0;
  }
};

}  // namespace Aperture

#endif  // _GR_KS_IC_RADIATION_SCHEME_H_
