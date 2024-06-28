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

#include "core/particle_structs.h"
#include "core/random.h"
#include "data/fields.h"
#include "data/phase_space.hpp"
#include "framework/environment.h"
#include "systems/grid.h"
#include "utils/interpolation.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
struct resonant_scattering_scheme{
  using value_t = typename Conf::value_t;

  const grid_t<Conf> &m_grid;
  value_t BQ = 1.0e7;
  value_t star_kT = 1.0e-3;
  value_t res_drag_coef = 4.72e13; // This is the default value from Beloborodov
                                   // 2013, normalized to time unit Rstar/c
  value_t ph_path = 0.0f;
  int downsample = 8;
  // int num_bins[Conf::dim];
  // value_t lower[Conf::dim];
  // value_t upper[Conf::dim];
  vec_t<int, 2> num_bins; // The two dimensions are energy and theta, in that order
  vec_t<value_t, 2> lower;
  vec_t<value_t, 2> upper;
  vec_t<ndptr_const<value_t, Conf::dim>, 3> m_B;
  ndptr<value_t, Conf::dim + 2> m_ph_flux;

  resonant_scattering_scheme(const grid_t<Conf> &grid) : m_grid(grid) {}

  void init() {
    sim_env().params().get_value("B_Q", BQ);
    sim_env().params().get_value("star_kT", star_kT);
    sim_env().params().get_value("res_drag_coef", res_drag_coef);

    nonown_ptr<vector_field<Conf>> B;
    // This is total B field, i.e. B0 + Bdelta
    sim_env().get_data("B", B);
#ifdef GPU_ENABLED
    m_B[0] = B->at(0).dev_ndptr_const();
    m_B[1] = B->at(1).dev_ndptr_const();
    m_B[2] = B->at(2).dev_ndptr_const();
#else
    m_B[0] = B->at(0).host_ndptr_const();
    m_B[1] = B->at(1).host_ndptr_const();
    m_B[2] = B->at(2).host_ndptr_const();
#endif

    sim_env().params().get_value("ph_flux_downsample", downsample);
    sim_env().params().get_vec_t("ph_flux_bins", num_bins);
    sim_env().params().get_vec_t("ph_flux_lower", lower);
    sim_env().params().get_vec_t("ph_flux_upper", upper);

    nonown_ptr<phase_space<Conf, 2>> ph_flux; // 2D for theta and energy
    ph_flux = sim_env().register_data<phase_space<Conf, 2>>(
        std::string("resonant_ph_flux"), m_grid, downsample, num_bins.data(),
        lower.data(), upper.data(), false, default_mem_type);
    ph_flux->reset_after_output(true);

#ifdef GPU_ENABLED
    m_ph_flux = ph_flux->data.dev_ndptr();
#else
    m_ph_flux = ph_flux->data.host_ndptr();
#endif
  }

  value_t absorption_rate(value_t b, value_t eph, value_t sinth) {
    return 0.0f;
  }

  HOST_DEVICE size_t emit_photon(const Grid<Conf::dim, value_t> &grid,
                                 const extent_t<Conf::dim> &ext, ptc_ptrs &ptc,
                                 size_t tid, ph_ptrs &ph, size_t ph_num,
                                 unsigned long long int *ph_pos,
                                 rand_state &state, value_t dt) {
    auto flag = ptc.flag[tid];
    if (check_flag(flag, PtcFlag::ignore_radiation)) {
      return 0;
    }

    // get particle information
    auto cell = ptc.cell[tid];
    auto idx = Conf::idx(cell, ext);
    auto pos = get_pos(idx, ext);

    value_t gamma = ptc.E[tid];
    value_t p1 = ptc.p1[tid];
    value_t p2 = ptc.p2[tid];
    value_t p3 = ptc.p3[tid];
    vec_t<value_t, 3> rel_x(ptc.x1[tid], ptc.x2[tid], ptc.x3[tid]);
    // x_global gives the global coordinate of the particle
    auto x_global = grid.coord_global(pos, rel_x);
    value_t r = grid_sph_t<Conf>::radius(x_global[0]);

    // Get local B field
    vec_t<value_t, 3> B;
    auto interp = interp_t<1, Conf::dim>{};
    B[0] = interp(rel_x, m_B[0], idx, ext, stagger_t(0b001));
    B[1] = interp(rel_x, m_B[1], idx, ext, stagger_t(0b010));
    B[2] = interp(rel_x, m_B[2], idx, ext, stagger_t(0b100));
    value_t B_mag = math::sqrt(B.dot(B));
    value_t b = B_mag / BQ;
    value_t p = math::sqrt(p1 * p1 + p2 * p2 + p3 * p3);
    value_t p_para = (p1 * B[0] + p2 * B[1] + p3 * B[2]) / B_mag;

    // Compute resonant cooling and emit photon if necessary
    value_t mu = math::abs(B[0] / B_mag); // mu is already absolute value
    value_t p_para_signed = sgn(p1) * math::abs(p_para);
    value_t gamma_para = math::sqrt(1.0f + p_para_signed * p_para_signed);
    // This beta is the absolute value of v_parallel/c
    value_t beta = math::sqrt(1.0f - 1.0f / square(gamma_para));
    // TODO: check whether this definition of y is correct
    value_t y = math::abs(b / (star_kT * (gamma_para - p_para_signed * mu)));

    if (y > 20.0f || y <= 0.0f)
      return 0; // Way out of resonance, do not do anything

    // This is based on Beloborodov 2013, Eq. B4. The resonant drag coefficient
    // is the main rescaling parameter, res_drag_coef = alpha * c / 4 / lambda_bar
    value_t coef = res_drag_coef * square(star_kT) * y * y /
        (r * r * (math::exp(y) - 1.0f));
    value_t Nph = math::abs(coef) * dt / gamma;
    // This is the general energy of the outgoing photon, in the electron rest frame
    // value_t Eph =
    //     std::min(g - 1.0f, g * (1.0f - 1.0f / math::sqrt(1.0f + 2.0f * B_mag / BQ)));

    // Now we need to compute the outgoing photon energy. It is a fixed energy
    // in the electron rest frame, but needs to be Lorenz transformed to the lab
    // frame. We start with generating a random cos theta from -1 to 1
    float u = 2.0f * rng_uniform<float>(state) - 1.0f;
    // TODO: check Eph expression!
    value_t Eph = math::abs(gamma * (1.0f + beta * u) *
                            (1.0f - 1.0f / math::sqrt(1.0f + 2.0f * b)));

    // Photon direction
    float phi_p = 2.0f * M_PI * rng_uniform<float>(state);
    float cphi = math::cos(phi_p);
    float sphi = math::sin(phi_p);
    value_t sth = sqrt(1.0f - u * u);

    // Lorentz transform u to the lab frame
    // TODO: Check whether this is correct
    u = (u + beta) / (1 + beta * u);
    value_t n1 = p1 / p;
    value_t n2 = p2 / p;
    value_t n3 = p3 / p;
    value_t np = math::sqrt(n1 * n1 + n2 * n2);

    value_t n_ph1 = n1 * u + sth * (n2 * cphi + n1 * n3 * sphi) / np;
    value_t n_ph2 = n2 * u + sth * (-n1 * cphi + n2 * n3 * sphi) / np;
    value_t n_ph3 = n3 * u - sth * (-np * sphi);

    bool produce_photon = false;
    // Need to take Nph < 1 and > 1 differently, since the photon production
    // may take a significant amount of energy from the emitting particle
    if (Eph > 2.0f) { // Photon energy larger than 1MeV, treat as discrete photon
      if (Nph > 1.0f || rng_uniform<float>(state) < Nph) {
        // Produce a photon when Nph > 1 or when a dice roll is lower than the rate
        produce_photon = true;
      }
    } else {
      // Just do drag and deposit the photon into an angle bin

      // Compute analytically the drag force on the particle and apply it. This is taken
      // from Beloborodov 2013, Eq. B6. Need to check the sign. TODO
      value_t drag_coef = coef * star_kT * y * (gamma_para * mu - p_para_signed);
      if (B[0] < 0.0f) drag_coef = -drag_coef; // To account for the direction of B field
      p1 += B[0] * dt * drag_coef / B_mag;
      p2 += B[1] * dt * drag_coef / B_mag;
      p3 += B[2] * dt * drag_coef / B_mag;

      ptc.p1[tid] = p1;
      ptc.p2[tid] = p2;
      ptc.p3[tid] = p3;
      ptc.E[tid] = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

      // TODO: deposit the outgoing photon into some array
      value_t th_ph = math::acos(n_ph3);
      // value_t
      // value_t phi_ph = math::atan2()
      return 0;
    }

    if (!produce_photon) return 0;

    // value_t ph1 = Eph * (n1 * u + sth * (n2 * cphi + n1 * n3 * sphi) / np);
    // value_t ph2 = Eph * (n2 * u + sth * (-n1 * cphi + n2 * n3 * sphi) / np);
    // value_t ph3 = Eph * (n3 * u - sth * (-np * sphi));

    ptc.p1[tid] = (p1 -= Eph * n_ph1);
    ptc.p2[tid] = (p2 -= Eph * n_ph2);
    ptc.p3[tid] = (p3 -= Eph * n_ph3);
    ptc.E[tid] = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

    // Actually produce the photons
    size_t offset = ph_num + atomic_add(ph_pos, 1);
    ph.x1[offset] = ptc.x1[tid];
    ph.x2[offset] = ptc.x2[tid];
    ph.x3[offset] = ptc.x3[tid];
    ph.p1[offset] = Eph * n_ph1;
    ph.p2[offset] = Eph * n_ph2;
    ph.p3[offset] = Eph * n_ph3;
    ph.E[offset] = Eph;
    ph.weight[offset] = ptc.weight[tid];
    ph.path_left[offset] = ph_path;
    ph.cell[offset] = ptc.cell[tid];
    // TODO: Set polarization

    return offset;
  }

  HOST_DEVICE size_t produce_pair(const Grid<Conf::dim, value_t> &grid,
                                  const extent_t<Conf::dim> &ext, ph_ptrs &ph,
                                  size_t tid, ptc_ptrs &ptc, size_t ptc_num,
                                  unsigned long long int *ptc_pos,
                                  rand_state &state, value_t dt) {
    // Get the magnetic field vector at the particle location
    auto cell = ph.cell[tid];
    auto idx = Conf::idx(cell, ext);
    auto x = vec_t<value_t, 3>(ph.x1[tid], ph.x2[tid], ph.x3[tid]);
    auto p = vec_t<value_t, 3>(ph.p1[tid], ph.p2[tid], ph.p3[tid]);
    auto pos = get_pos(idx, ext);

    // x_global gives the cartesian coordinate of the photon.
    auto x_global = grid.coord_global(pos, x);

    vec_t<value_t, 3> B;
    auto interp = interp_t<1, Conf::dim>{};
    B[0] = interp(x, m_B[0], idx, ext, stagger_t(0b001));
    B[1] = interp(x, m_B[1], idx, ext, stagger_t(0b010));
    B[2] = interp(x, m_B[2], idx, ext, stagger_t(0b100));

    // Compute the angle between photon and B field and compute the quantum
    // parameter chi value_t chi = quantum_chi(p, B, m_BQ);
    value_t B_mag = math::sqrt(B.dot(B));
    value_t eph = ph.E[tid];
    auto pxB = cross(p, B);
    auto pdotB = p.dot(B);
    value_t sinth = math::abs(math::sqrt(pxB.dot(pxB)) / B_mag / eph);
    value_t b = B_mag / BQ;

    // There are two requirements for pair production: 1. The photon energy
    // needs to exceed the threshold E_thr = 2me c^2 / sin th; 2. The absorption
    // rate is proportional to 4.3e7 b exp(-8/3bE\sin\theta).
    // TODO: This threshold energy depends on polarization!
    value_t Ethr = 2.0f / sinth;
    if (eph < Ethr)
      return 0;

    value_t chi = 0.5f * eph * b * sinth;
    // TODO: Is there any rescaling that we need to do? Also check units
    value_t prob = 4.3e7 * 1e6 * b * math::exp(-4.0 / 3.0 / chi) * dt;

    value_t u = rng_uniform<value_t>(state);
    if (u < prob) {
      // Actually produce the electron-positron pair
      size_t offset = ptc_num + atomic_add(ptc_pos, 2);
      size_t offset_e = offset;
      size_t offset_p = offset + 1;

      value_t p_ptc =
          math::sqrt(0.25f - 1.0f / square(eph)) * math::abs(pdotB) / B_mag;
      // printf("sinth is %f, path is %f, eph is %f, prob is %f, chi is %f,
      // p_ptc is %f\n", sinth, ph.path_left[tid], eph, prob,
      //        0.5f * eph * B_mag/m_BQ * sinth * m_zeta, p_ptc);

      // Immediately cool to zero magnetic moment and reduce Lorentz factor as
      // needed
      // value_t gamma = 0.5f * eph;
      value_t gamma = math::sqrt(1.0f + p_ptc * p_ptc);
      // if (sinth > TINY && gamma > 1.0f / sinth) {
      //   gamma = 1.0f / sinth;
      //   if (gamma < 1.01f) gamma = 1.01;
      //   p_ptc = math::sqrt(gamma * gamma - 1.0f);
      // }

      ptc.x1[offset_e] = ptc.x1[offset_p] = x[0];
      ptc.x2[offset_e] = ptc.x2[offset_p] = x[1];
      ptc.x3[offset_e] = ptc.x3[offset_p] = x[2];

      // Particle momentum is along B, direction is inherited from initial
      // photon direction
      value_t sign = sgn(pdotB);
      ptc.p1[offset_e] = ptc.p1[offset_p] = p_ptc * sign * B[0] / B_mag;
      ptc.p2[offset_e] = ptc.p2[offset_p] = p_ptc * sign * B[1] / B_mag;
      ptc.p3[offset_e] = ptc.p3[offset_p] = p_ptc * sign * B[2] / B_mag;
      ptc.E[offset_e] = ptc.E[offset_p] = gamma;
      ptc.aux1[offset_e] = ptc.aux1[offset_p] = 0.0f;

      ptc.weight[offset_e] = ptc.weight[offset_p] = ph.weight[tid];
      ptc.cell[offset_e] = ptc.cell[offset_p] = cell;
      ptc.flag[offset_e] =
          set_ptc_type_flag(flag_or(PtcFlag::secondary), PtcType::electron);
      ptc.flag[offset_p] =
          set_ptc_type_flag(flag_or(PtcFlag::secondary), PtcType::positron);

      return offset;
    }

    return 0;
  }
};

}  // namespace Aperture
