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

#ifndef IC_RADIATION_SCHEME_H_
#define IC_RADIATION_SCHEME_H_

#include "core/cuda_control.h"
#include "core/particle_structs.h"
#include "core/random.h"
#include "framework/environment.h"
#include "systems/grid.h"
#include "systems/inverse_compton.h"
#include "systems/physics/ic_scattering.hpp"
#include "systems/physics/spectra.hpp"

namespace Aperture {

template <typename Conf>
struct IC_radiation_scheme {
  using value_t = typename Conf::value_t;

  const grid_t<Conf> &m_grid;
  ic_scatter_t m_ic_module;

  IC_radiation_scheme(const grid_t<Conf> &grid) : m_grid(grid) {}

  void init() {
    value_t emin = 1.0e-5;
    sim_env().params().get_value("emin", emin);
    value_t ic_path = 1.0;
    sim_env().params().get_value("IC_path", ic_path);
    value_t ic_alpha = 1.25;
    sim_env().params().get_value("IC_alpha", ic_alpha);
    value_t bb_kT = 1e-5;
    sim_env().params().get_value("bb_kT", bb_kT);

    // Configure the spectrum here and initialize the ic module
    Spectra::power_law_soft spec(ic_alpha, emin, 1.0);
    // Spectra::black_body spec(bb_kT);

    auto ic = sim_env().register_system<inverse_compton_t>();
    // ic->compute_coefficients(spec, spec.emin(), spec.emax(), 1.5e24 / ic_path);
    ic->compute_coefficients(spec, spec.emin(), spec.emax());

    m_ic_module = ic->get_ic_module();
  }

  HOST_DEVICE size_t emit_photon(const Grid<Conf::dim, value_t> &grid,
                                 const extent_t<Conf::dim> &ext, ptc_ptrs &ptc,
                                 size_t tid, ph_ptrs &ph, size_t ph_num,
                                 unsigned long long int *ph_pos, rng_t &rng,
                                 value_t dt) {
    value_t gamma = ptc.E[tid];
    value_t p = math::sqrt(square(ptc.p1[tid]) + square(ptc.p2[tid]) + square(ptc.p3[tid]));
    value_t lambda = m_ic_module.ic_scatter_rate(gamma) * dt;
    // printf("ic_prob is %f\n", ic_prob);
    int num_scattering = rng.poisson(lambda);

    value_t d_gamma = 0.0f;
    // printf("num_scattering is %f, e_ph / gamma is %f\n", weight, e_ph / gamma);
    for (int i = 0; i < num_scattering; i++) {
      value_t e_ph = m_ic_module.gen_photon_e(gamma, rng) * gamma;
      // value_t e_ph = m_ic_module.e_mean * gamma * gamma;
      gamma -= e_ph;
      d_gamma += e_ph;
    }
    printf("lambda is %f, num_scattering is %d, d_gamma is %f, cooling is %f\n",
           lambda, num_scattering, d_gamma, m_ic_module.compactness * (4.0f / 3.0f) * gamma * gamma * dt);
    // if (e_ph > gamma - 1.0f)
    //   e_ph = gamma - 1.0f;
    value_t new_p = math::sqrt(max(square(gamma) - 1.0f, 0.0f));

    ptc.p1[tid] *= new_p / p;
    ptc.p2[tid] *= new_p / p;
    ptc.p3[tid] *= new_p / p;
    ptc.E[tid] = gamma;

    return 0;
  }

  HOST_DEVICE size_t produce_pair(const Grid<Conf::dim, value_t> &grid,
                                  const extent_t<Conf::dim> &ext, ph_ptrs &ph,
                                  size_t tid, ptc_ptrs &ptc, size_t ptc_num,
                                  unsigned long long int *ptc_pos, rng_t &rng,
                                  value_t dt) {
    return 0;
  }
};


}

#endif // IC_RADIATION_SCHEME_H_
