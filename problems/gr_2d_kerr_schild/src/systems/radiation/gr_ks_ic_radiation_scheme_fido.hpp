/*
 * Copyright (c) 2024 Alex Chen.
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

#include "core/cuda_control.h"
#include "core/particle_structs.h"
#include "core/random.h"
#include "framework/environment.h"
#include "systems/grid_ks.h"
#include "systems/inverse_compton.h"
#include "systems/physics/ic_scattering.hpp"
#include "systems/physics/metric_kerr_schild.hpp"
#include "systems/physics/spectra.hpp"

namespace Aperture {

template <typename Conf>
struct gr_ks_ic_radiation_scheme_fido {
  using value_t = typename Conf::value_t;

  const grid_t<Conf> &m_grid;
  ic_scatter_t m_ic_module;
  value_t m_a = 0.99;
  value_t m_ic_opacity = 1.0;

  gr_ks_ic_radiation_scheme_fido(const grid_t<Conf> &grid) : m_grid(grid) {}

  void init() {
    value_t ph_kT = 1.0e-3;
    sim_env().params().get_value("ph_kT", ph_kT);
    value_t e_min = 1.0e-3, e_max = 0.5, spec_alpha = 1.5;
    sim_env().params().get_value("bh_spin", m_a);
    sim_env().params().get_value("IC_opacity", m_ic_opacity);
    sim_env().params().get_value("spectrum_emin", e_min);
    sim_env().params().get_value("spectrum_emax", e_max);
    sim_env().params().get_value("spectrum_alpha", spec_alpha);

    // Configure the spectrum here and initialize the ic module
    // Spectra::broken_power_law spec(1.25, 1.1, emin, 1.0e-10, 0.1);
    Spectra::black_body spec(ph_kT);
    // Spectra::power_law_soft spec(spec_alpha, e_min, 0.5);

    auto ic = sim_env().register_system<inverse_compton_t>();
    // ic->compute_coefficients(spec, spec.emin(), spec.emax(), 1.5e24 /
    // ic_path);
    ic->compute_coefficients(spec, spec.emin(), spec.emax());

    m_ic_module = ic->get_ic_module();
  }

  HOST_DEVICE size_t emit_photon(const Grid<Conf::dim, value_t> &grid,
                                 const extent_t<Conf::dim> &ext, ptc_ptrs &ptc,
                                 size_t tid, ph_ptrs &ph, size_t ph_num,
                                 unsigned long long int *ph_pos,
                                 rand_state &state, value_t dt) {
    // First obtain the global position of the particle
    vec_t<value_t, 3> x(ptc.x1[tid], ptc.x2[tid], ptc.x3[tid]);
    auto idx = Conf::idx(ptc.cell[tid], ext);
    auto pos = get_pos(idx, ext);

    vec_t<value_t, 3> x_global = grid.coord_global(pos, x);
    x_global[0] = grid_ks_t<Conf>::radius(x_global[0]);
    x_global[1] = grid_ks_t<Conf>::theta(x_global[1]);
    value_t sth = math::sin(x_global[1]);
    value_t cth = math::cos(x_global[1]);
    const value_t& r = x_global[0];
    const value_t& th = x_global[1];

    // if (r < Metric_KS::rH(m_a) + 0.1f || r > 6.0f ||
    //     th < 0.1 * grid.guard[1] || math::abs(th - M_PI) < 0.1 * grid.guard[1]) {
    if (r < Metric_KS::rH(m_a)) {
      return 0;
    }

    // Obtain the lower 4-momentum components
    vec_t<value_t, 3> u3(ptc.p1[tid], ptc.p2[tid], ptc.p3[tid]);
    value_t u_0 = Metric_KS::u_0(m_a, r, sth, cth, u3, false);
    // if (u_0 != u_0) {
    //   printf("u0 is %f, p is (%f, %f, %f), r is %f, th is %f\n", u_0, ptc.p1[tid], ptc.p2[tid], ptc.p3[tid], r, th);
    // }
    vec_t<value_t, 4> u(u_0, ptc.p1[tid], ptc.p2[tid], ptc.p3[tid]);

    // transform the lower momentum components to FIDO frame
    auto u_fido = Metric_KS::convert_to_FIDO_lower(u, m_a, r, x_global[1]);

    // dt = dt * sqrt_delta * sqrt_rho2 / sqrt_sigma;
    auto gamma = -u_fido[0];
    if (gamma < 1.1f) {
      // Lorentz factor is too low, won't be able to emit a pair-producing photon anyway
      return 0;
    }

    // if (gamma != gamma) {
    //   // guard against nan
    //   printf("u is (%f, %f, %f, %f), u_fido[0] is %f\n", u[0], u[1], u[2], u[3], u_fido[0]);
    //   return 0;
    // }
    value_t gamma0 = gamma;
    value_t p0 = math::sqrt(gamma0 * gamma0 - 1.0);
    value_t alf = Metric_KS::alpha(m_a, r, th);
    value_t ic_prob = m_ic_module.ic_scatter_rate(gamma) * alf * dt;

    // printf("gamma is %f, ic_prob is %f, e_ph is %f\n", gamma, ic_prob, e_ph);
    // printf("gamma is %f, ic_prob is %f\n", gamma, ic_prob);

    value_t rand = rng_uniform<value_t>(state);
    if (rand >= ic_prob) {
      // no photon emitted
      return 0;
    }

    // draw emitted photon energy.
    // e_ph is a number between 0 and 1, the fraction of the photon energy with
    // respect to the electron energy
    value_t e_ph = m_ic_module.gen_photon_e(gamma, state) * gamma;
    if (e_ph > gamma - 1.1f) {
      e_ph = gamma - 1.1f;
    }
    // printf("u_fido is (%f, %f, %f, %f)\n", u_fido[0], u_fido[1], u_fido[2], u_fido[3]);
    // printf("gamma is %f, e_ph %f\n", gamma, e_ph);

    // Regardless of whether the photon is tracked, subtract its energy from the particle
    gamma -= e_ph;
    u_fido[0] = -gamma;
    value_t p_new = math::sqrt(gamma * gamma - 1.0); 
    u_fido[1] *= p_new / p0;
    u_fido[2] *= p_new / p0;
    u_fido[3] *= p_new / p0;
    u = Metric_KS::convert_from_FIDO_lower(u_fido, m_a, r, th);

    ptc.p1[tid] = u[1];
    ptc.p2[tid] = u[2];
    ptc.p3[tid] = u[3];
    ptc.E[tid] = Metric_KS::u0(m_a, r, th, u.template subset<1, 4>());
    // if (ptc.E[tid] != ptc.E[tid]) {
    //   printf("nan detected! u_fido is (%f, %f, %f, %f)\n", u_fido[0], u_fido[1], u_fido[2], u_fido[3]);
    // }

    if (e_ph < 5.1f || r > 6.0f) {
      // Do not track low energy photons or photons too far away
      return 0;
    }

    u_fido[0] = -e_ph;
    u_fido[1] *= e_ph / p_new;
    u_fido[2] *= e_ph / p_new;
    u_fido[3] *= e_ph / p_new;
    u = Metric_KS::convert_from_FIDO_lower(u_fido, m_a, r, th);

    size_t offset = ph_num + atomic_add(ph_pos, 1);
    ph.x1[offset] = ptc.x1[tid];
    ph.x2[offset] = ptc.x2[tid];
    ph.x3[offset] = ptc.x3[tid];
    ph.cell[offset] = ptc.cell[tid];
    ph.weight[offset] = ptc.weight[tid];

    ph.p1[offset] = u[1];
    ph.p2[offset] = u[2];
    ph.p3[offset] = u[3];
    ph.E[offset] = Metric_KS::u0(m_a, r, th, u.template subset<1, 4>());
    ph.flag[offset] = 0;

    return offset;
    return 0;
  }

  HOST_DEVICE size_t produce_pair(const Grid<Conf::dim, value_t> &grid,
                                  const extent_t<Conf::dim> &ext, ph_ptrs &ph,
                                  size_t tid, ptc_ptrs &ptc, size_t ptc_num,
                                  unsigned long long int *ptc_pos,
                                  rand_state &state, value_t dt) {
    // First obtain the global position of the photon
    vec_t<value_t, 3> x(ph.x1[tid], ph.x2[tid], ph.x3[tid]);
    auto idx = Conf::idx(ph.cell[tid], ext);
    auto pos = get_pos(idx, ext);

    vec_t<value_t, 3> x_global = grid.coord_global(pos, x);
    x_global[0] = grid_ks_t<Conf>::radius(x_global[0]);
    x_global[1] = grid_ks_t<Conf>::theta(x_global[1]);
    value_t sth = math::sin(x_global[1]);
    value_t cth = math::cos(x_global[1]);
    const value_t& r = x_global[0];
    const value_t& th = x_global[1];

    // Censor photons at large distances
    // TODO: potentially deposit the photon at large radii to form raytracing
    if (r < Metric_KS::rH(m_a) || r > 6.0f) {
      ph.cell[tid] = empty_cell;
      return 0;
    }

    // Obtain the lower 4-momentum components
    // vec_t<value_t, 3> u(ph.p1[tid], ph.p2[tid], ph.p3[tid]);
    value_t u_0 = Metric_KS::u_0(m_a, r, sth, cth, {ph.p1[tid], ph.p2[tid], ph.p3[tid]}, true);
    vec_t<value_t, 4> u(u_0, ph.p1[tid], ph.p2[tid], ph.p3[tid]);

    // transform the lower momentum components to FIDO frame
    auto u_fido = Metric_KS::convert_to_FIDO_lower(u, m_a, r, th);

    value_t alf = Metric_KS::alpha(m_a, r, th);
    value_t gg_prob = m_ic_module.gg_scatter_rate(-u_fido[0]) * alf * dt;
    // printf("u_0 is %f, gg_prob is %f\n", -u_fido[0], gg_prob);
    if (gg_prob < dt * m_ic_opacity * 1e-4) {
      // censor photons that have too low chance of producing a pair
      ph.cell[tid] = empty_cell;
      return 0;
    }

    if (rng_uniform<value_t>(state) >= gg_prob) {
      return 0;  // Does not produce a pair
    }

    size_t offset = ptc_num + atomic_add(ptc_pos, 2);
    size_t offset_e = offset;
    size_t offset_p = offset + 1;

    ptc.x1[offset_e] = ptc.x1[offset_p] = ph.x1[tid];
    ptc.x2[offset_e] = ptc.x2[offset_p] = ph.x2[tid];
    ptc.x3[offset_e] = ptc.x3[offset_p] = ph.x3[tid];

    u_fido *= 0.5f;
    u = Metric_KS::convert_from_FIDO_lower(u_fido, m_a, r, th);

    ptc.p1[offset_e] = ptc.p1[offset_p] = u[1];
    ptc.p2[offset_e] = ptc.p2[offset_p] = u[2];
    ptc.p3[offset_e] = ptc.p3[offset_p] = u[3];

    ptc.E[offset_e] = ptc.E[offset_p] = Metric_KS::u0(m_a, r, th, u.template subset<1, 4>());

// #ifndef NDEBUG
//     assert(ptc.cell[offset_e] == empty_cell);
//     assert(ptc.cell[offset_p] == empty_cell);
// #endif
    ptc.weight[offset_e] = ptc.weight[offset_p] = ph.weight[tid];
    ptc.cell[offset_e] = ptc.cell[offset_p] = ph.cell[tid];
    uint32_t base_flag = flag_or(PtcFlag::secondary);
    ptc.flag[offset_e] =
        set_ptc_type_flag(base_flag, PtcType::electron);
    ptc.flag[offset_p] =
        set_ptc_type_flag(base_flag, PtcType::positron);

    return offset;
    // return 0;
  }
};

}  // namespace Aperture
