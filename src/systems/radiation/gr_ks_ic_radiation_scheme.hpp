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
#include "systems/grid_ks.h"
#include "systems/inverse_compton.h"
#include "systems/physics/ic_scattering.hpp"
#include "systems/physics/metric_kerr_schild.hpp"
#include "systems/physics/spectra.hpp"

namespace Aperture {

template <typename Conf>
struct gr_ks_ic_radiation_scheme {
  using value_t = typename Conf::value_t;

  const grid_t<Conf> &m_grid;
  ic_scatter_t m_ic_module;
  value_t m_a = 0.99;

  gr_ks_ic_radiation_scheme(const grid_t<Conf> &grid) : m_grid(grid) {}

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

  HOST_DEVICE size_t emit_photon(const Grid<Conf::dim, value_t> &grid,
                                 const extent_t<Conf::dim> &ext, ptc_ptrs &ptc,
                                 size_t tid, ph_ptrs &ph, size_t ph_num,
                                 unsigned long long int *ph_pos, rng_t &rng,
                                 value_t dt) {
    // First obtain the global position of the particle
    vec_t<value_t, 3> x(ptc.x1[tid], ptc.x2[tid], ptc.x3[tid]);
    auto idx = Conf::idx(ptc.cell[tid], ext);
    auto pos = get_pos(idx, ext);

    vec_t<value_t, 3> x_global = grid.pos_global(pos, x);
    x_global[0] = grid_ks_t<Conf>::radius(x_global[0]);
    x_global[1] = grid_ks_t<Conf>::theta(x_global[1]);
    value_t sth = math::sin(x_global[1]);
    value_t cth = math::cos(x_global[1]);
    value_t r = x_global[0];

    if (r < Metric_KS::rH(m_a) + 0.1f) {
      return 0;
    }

    // Obtain the lower 4-momentum components
    vec_t<value_t, 3> u(ptc.p1[tid], ptc.p2[tid], ptc.p3[tid]);
    value_t u_0 = Metric_KS::u_0(m_a, r, sth, cth, u, false);
    value_t u0 = Metric_KS::u0(m_a, r, sth, cth, u, false);

    // transform the lower momentum components to ZAMO frame
    value_t sqrt_rho2 = math::sqrt(Metric_KS::rho2(m_a, r, sth, cth));
    value_t sqrt_delta = math::sqrt(Metric_KS::Delta(m_a, r));
    value_t sqrt_sigma = math::sqrt(Metric_KS::Sigma(m_a, r, sth, cth));
    // U_0 in zamo is the particle energy (with a minus sign?)
    value_t zamo_u_0 = (sqrt_sigma * u_0 + 2.0f * m_a * r * u[2] / sqrt_sigma) /
                       sqrt_delta * sqrt_rho2;
    printf("particle r is %f, theta is %f\n", r, x_global[1]);
    printf("particle u_0 is %f, u_i is (%f, %f, %f), zamo_u_0 is %f\n", u_0,
           u[0], u[1], u[2], zamo_u_0);

    value_t gamma = math::abs(zamo_u_0);
    // Transform dt into ZAMO frame
    value_t delta_x1 = ((Metric_KS::gu11(m_a, r, sth, cth) * u[0] +
                         Metric_KS::gu13(m_a, r, sth, cth) * u[2]) /
                            Metric_KS::u0(m_a, r, sth, cth, u, false) -
                        Metric_KS::beta1(m_a, r, sth, cth)) *
                       dt;
    dt = dt * sqrt_delta * sqrt_rho2 / sqrt_sigma -
         2.0f * r * sqrt_rho2 / (sqrt_delta * sqrt_sigma) * delta_x1;
    // dt = dt * sqrt_delta * sqrt_rho2 / sqrt_sigma;
    value_t ic_prob = m_ic_module.ic_scatter_rate(gamma) * math::abs(dt);

    // printf("gamma is %f, ic_prob is %f, e_ph is %f\n", gamma, ic_prob, e_ph);
    printf("gamma is %f, ic_prob is %f\n", gamma, ic_prob);

    value_t rand = rng.uniform<value_t>();
    if (rand >= ic_prob) {
      // no photon emitted
      return 0;
    }

    // draw emitted photon energy
    rand = rng.uniform<value_t>();
    // e_ph is a number between 0 and 1, the fraction of the photon energy with
    // respect to the electron energy
    value_t e_ph = m_ic_module.gen_photon_e(gamma, rand);
    printf("emitting photon with energy e_ph %f\n", e_ph);
    if (e_ph * u0 < 2.01f) {
      ptc.p1[tid] *= (1.0f - e_ph);
      ptc.p2[tid] *= (1.0f - e_ph);
      ptc.p3[tid] *= (1.0f - e_ph);
      // photon energy not enough. TODO: is this the best criteria?
      return 0;
    }

    value_t zamo_u_1_ph =
        e_ph * ((2.0f * r * u_0 + m_a * u[2]) / (sqrt_delta * sqrt_rho2) +
                sqrt_delta * u[0] / sqrt_rho2);
    value_t zamo_u_2_ph = e_ph * u[1] / sqrt_rho2;
    value_t zamo_u_3_ph = e_ph * (sqrt_rho2 / sqrt_sigma) * u[2] / sth;

    size_t offset = ph_num + atomic_add(ph_pos, 1);
    ph.x1[offset] = ptc.x1[tid];
    ph.x2[offset] = ptc.x2[tid];
    ph.x3[offset] = ptc.x3[tid];
    ph.cell[offset] = ptc.cell[tid];
    ph.weight[offset] = ptc.weight[tid];
    // In ZAMO frame, photon lower u_i are aligned with particle lower u_i. Not
    // necessarily true for KS coords. We need to transform from ZAMO back to KS
    // ph.p1[offset] =
    //     -2.0f * r * sqrt_rho2 * e_ph * zamo_u_0 / (sqrt_delta * sqrt_sigma) +
    //     sqrt_rho2 * zamo_u_1_ph / sqrt_delta +
    //     (4.0f * r * r - square(sqrt_sigma)) * m_a * sth * zamo_u_3_ph /
    //         (square(sqrt_delta) * sqrt_sigma * sqrt_rho2);
    // ph.p2[offset] = sqrt_rho2 * zamo_u_2_ph;
    // ph.p3[offset] = (sqrt_sigma / sqrt_rho2) * zamo_u_3_ph;
    ph.p1[offset] = e_ph * ptc.p1[tid];
    ph.p2[offset] = e_ph * ptc.p2[tid];
    ph.p3[offset] = e_ph * ptc.p3[tid];

    ptc.p1[tid] -= ph.p1[offset];
    ptc.p2[tid] -= ph.p2[offset];
    ptc.p3[tid] -= ph.p3[offset];
    // Note: not necessary to write ptc.E[tid] since we really don't use it at
    // all
    printf("particle u_i is now (%f, %f, %f)\n", ptc.p1[tid], ptc.p2[tid],
           ptc.p3[tid]);

    return offset;
  }

  HOST_DEVICE size_t produce_pair(const Grid<Conf::dim, value_t> &grid,
                                  const extent_t<Conf::dim> &ext, ph_ptrs &ph,
                                  size_t tid, ptc_ptrs &ptc, size_t ptc_num,
                                  unsigned long long int *ptc_pos, rng_t &rng,
                                  value_t dt) {
    // First obtain the global position of the photon
    vec_t<value_t, 3> x(ph.x1[tid], ph.x2[tid], ph.x3[tid]);
    auto idx = Conf::idx(ph.cell[tid], ext);
    auto pos = get_pos(idx, ext);

    vec_t<value_t, 3> x_global = grid.pos_global(pos, x);
    x_global[0] = grid_ks_t<Conf>::radius(x_global[0]);
    x_global[1] = grid_ks_t<Conf>::theta(x_global[1]);
    value_t sth = math::sin(x_global[1]);
    value_t cth = math::cos(x_global[1]);
    value_t r = x_global[0];

    // Obtain the lower 4-momentum components
    vec_t<value_t, 3> u(ph.p1[tid], ph.p2[tid], ph.p3[tid]);
    value_t u_0 = Metric_KS::u_0(m_a, r, sth, cth, u, true);

    // transform the lower momentum components to ZAMO frame
    value_t sqrt_rho2 = math::sqrt(Metric_KS::rho2(m_a, r, sth, cth));
    value_t sqrt_delta = math::sqrt(Metric_KS::Delta(m_a, r));
    value_t sqrt_sigma = math::sqrt(Metric_KS::Sigma(m_a, r, sth, cth));
    // U_0 in zamo is the photon energy (with a minus sign?)
    value_t zamo_u_0 = (sqrt_sigma * u_0 + 2.0f * m_a * r * u[2] / sqrt_sigma) /
                       (sqrt_delta * sqrt_rho2);
    // Transform the photon interval into dt in ZAMO
    value_t delta_x1 = (Metric_KS::gu11(m_a, r, sth, cth) * u[0] +
                        Metric_KS::gu13(m_a, r, sth, cth) * u[2]) /
                           Metric_KS::u0(m_a, r, sth, cth, u, true) -
                       Metric_KS::beta1(m_a, r, sth, cth);
    dt = dt * sqrt_delta * sqrt_rho2 / sqrt_sigma -
         2.0f * r * sqrt_rho2 / (sqrt_delta * sqrt_sigma) * delta_x1;
    value_t gg_prob = m_ic_module.gg_scatter_rate(math::abs(zamo_u_0)) * dt;

    if (rng.uniform<value_t>() >= gg_prob) {
      return 0;  // Does not produce a pair
    }

    size_t offset = ptc_num + atomic_add(ptc_pos, 2);
    size_t offset_e = offset;
    size_t offset_p = offset + 1;

    ptc.x1[offset_e] = ptc.x1[offset_p] = ph.x1[tid];
    ptc.x2[offset_e] = ptc.x2[offset_p] = ph.x2[tid];
    ptc.x3[offset_e] = ptc.x3[offset_p] = ph.x3[tid];

    // At very high energies in ZAMO, this approximation should be valid
    ptc.p1[offset_e] = ptc.p1[offset_p] = 0.5f * ph.p1[tid];
    ptc.p2[offset_e] = ptc.p2[offset_p] = 0.5f * ph.p2[tid];
    ptc.p3[offset_e] = ptc.p3[offset_p] = 0.5f * ph.p3[tid];

#ifndef NDEBUG
    assert(ptc.cell[offset_e] == empty_cell);
    assert(ptc.cell[offset_p] == empty_cell);
#endif
    ptc.weight[offset_e] = ptc.weight[offset_p] = ph.weight[tid];
    ptc.cell[offset_e] = ptc.cell[offset_p] = ph.cell[tid];
    ptc.flag[offset_e] =
        set_ptc_type_flag(flag_or(PtcFlag::secondary), PtcType::electron);
    ptc.flag[offset_p] =
        set_ptc_type_flag(flag_or(PtcFlag::secondary), PtcType::positron);

    return offset;
  }
};

}  // namespace Aperture

#endif  // _GR_KS_IC_RADIATION_SCHEME_H_
