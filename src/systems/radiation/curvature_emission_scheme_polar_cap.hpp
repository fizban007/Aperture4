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

#ifndef _CURVATURE_EMISSION_SCHEME_POLAR_CAP_H_
#define _CURVATURE_EMISSION_SCHEME_POLAR_CAP_H_

#include "core/cuda_control.h"
#include "core/particle_structs.h"
#include "core/random.h"
#include "data/fields.h"
#include "data/phase_space.hpp"
#include "framework/environment.h"
#include "systems/grid.h"
#include "systems/policies/coord_policy_cartesian_gca_lite.hpp"
#include "systems/physics/sync_emission_helper.hpp"
#include "systems/sync_curv_emission.h"
#include "utils/interpolation.hpp"
#include "utils/util_functions.h"

namespace Aperture {

HOST_DEVICE Scalar
dipole_curv_radius(Scalar r, Scalar th) {
  Scalar sinth =
      std::max(math::sin(th), (Scalar)1.0e-5);  // Avoid the fringe case of sinth = 0
  Scalar costh2 = 1.0f - sinth * sinth;
  Scalar tmp = 1.0f + 3.0f * costh2;
  Scalar Rc = r * tmp * math::sqrt(tmp) / (3.0f * sinth * (1.0f + costh2));
  return Rc;
}

HOST_DEVICE Scalar
dipole_curv_radius_above_polar_cap(Scalar x, Scalar y, Scalar z) {
  Scalar r_cyl = math::sqrt(x * x + y * y);
  Scalar z_r = z + 1.0f;  // R* is our unit
  Scalar th = atan2(r_cyl, z_r);
  Scalar r = math::sqrt(z_r * z_r + r_cyl * r_cyl);
  return dipole_curv_radius(r, th);
}

HOST_DEVICE Scalar
magnetic_pair_production_rate(Scalar b, Scalar eph, Scalar sinth, Scalar Rpc_over_Rstar) {
  // The coefficient is 0.23 * \alpha_f * R_pc / \labmdabar_c, seems no reason to rescale
  // return 4.35e13 * b * sinth * math::exp(-4.0f / 3.0f / (0.5f * eph * b * sinth));
  return 4.35e13 * Rpc_over_Rstar * b * math::exp(-4.0f / 3.0f / (0.5f * eph * b * sinth));
}

template <typename Conf>
struct curvature_emission_scheme_polar_cap {
  using value_t = typename Conf::value_t;

  const grid_t<Conf> &m_grid;
  sync_emission_helper_t m_sync_module;
  value_t m_BQ = 1.0e7;    // B_Q determines the spectrum
  value_t m_e0 = 1.0e-6;
  value_t m_nc = 1.0;
  value_t m_re = 3.0 * m_e0 * m_nc / 2.0;   // r_e determines the overall curvature loss rate
  value_t m_zeta = 7.0;
  value_t m_rpc = 1.0;  // r_pc is the polar cap radius
  value_t m_Rstar = 10.0;
  value_t m_omega;
  vec_t<ndptr_const<value_t, Conf::dim>, 3> m_B;
  vec_t<ndptr_const<value_t, Conf::dim>, 3> m_E;

  curvature_emission_scheme_polar_cap(const grid_t<Conf> &grid)
      : m_grid(grid) {}

  void init() {
    // initialize the spectrum related parameters
    sim_env().params().get_value("B_Q", m_BQ);
    sim_env().params().get_value("e0", m_e0);
    sim_env().params().get_value("nc", m_nc);
    sim_env().params().get_value("zeta", m_zeta);
    sim_env().params().get_value("Rpc", m_rpc);
    sim_env().params().get_value("R_star", m_Rstar);

    m_re = 3.0 * m_e0 * m_nc / 2.0;
    m_omega = square(m_rpc / m_Rstar);

    // initialize synchro-curvature radiation module
    auto sync_module =
        sim_env().register_system<sync_curv_emission_t>(default_mem_type);
    m_sync_module = sync_module->get_helper();

    // Get data pointers of B field
    nonown_ptr<vector_field<Conf>> B, E;
    sim_env().get_data("B", B);
    sim_env().get_data("E", E);

#ifdef CUDA_ENABLED
    m_B[0] = B->at(0).dev_ndptr_const();
    m_B[1] = B->at(1).dev_ndptr_const();
    m_B[2] = B->at(2).dev_ndptr_const();
    m_E[0] = E->at(0).dev_ndptr_const();
    m_E[1] = E->at(1).dev_ndptr_const();
    m_E[2] = E->at(2).dev_ndptr_const();
#else
    m_B[0] = B->at(0).host_ndptr_const();
    m_B[1] = B->at(1).host_ndptr_const();
    m_B[2] = B->at(2).host_ndptr_const();
    m_E[0] = E->at(0).host_ndptr_const();
    m_E[1] = E->at(1).host_ndptr_const();
    m_E[2] = E->at(2).host_ndptr_const();
#endif
  }

  HOST_DEVICE size_t emit_photon(const Grid<Conf::dim, value_t> &grid,
                                 const extent_t<Conf::dim> &ext, ptc_ptrs &ptc,
                                 size_t tid, ph_ptrs &ph, size_t ph_num,
                                 unsigned long long int *ph_pos, rng_t &rng,
                                 value_t dt) {
    auto flag = ptc.flag[tid];
    if (check_flag(flag, PtcFlag::ignore_radiation)) {
      return 0;
    }

    value_t gamma = ptc.E[tid];
    value_t p1 = ptc.p1[tid];
    value_t p2 = ptc.p2[tid];
    value_t p3 = ptc.p3[tid];
    vec_t<value_t, 3> rel_x(ptc.x1[tid], ptc.x2[tid], ptc.x3[tid]);

    auto cell = ptc.cell[tid];
    auto idx = Conf::idx(cell, ext);
    // Compute the radius of curvature using particle location
    auto pos = get_pos(idx, ext);
    // x_global gives the cartesian coordinate of the particle. We renormalize
    // it to units of R_star
    auto x_global = grid.pos_global(pos, rel_x) * (m_rpc / m_Rstar);

    vec_t<value_t, 3> B, E;
    auto interp = interp_t<1, Conf::dim>{};
    B[0] = interp(rel_x, m_B[0], idx, ext, stagger_t(0b001));
    B[1] = interp(rel_x, m_B[1], idx, ext, stagger_t(0b010));
    B[2] = interp(rel_x, m_B[2], idx, ext, stagger_t(0b100));
    E[0] = interp(rel_x, m_E[0], idx, ext, stagger_t(0b110));
    E[1] = interp(rel_x, m_E[1], idx, ext, stagger_t(0b101));
    E[2] = interp(rel_x, m_E[2], idx, ext, stagger_t(0b011));
    value_t B_mag = math::sqrt(B.dot(B));

    value_t p = math::sqrt(p1*p1 + p2*p2 + p3*p3);

    // Rc is computed in units of Rstar, we renormalize it to rpc units
    value_t Rc = dipole_curv_radius_above_polar_cap(x_global[0], x_global[1],
                                                    x_global[2]) * (m_Rstar / m_rpc);

    // printf("x_global[2] is %f, Rc is %f\n", x_global[2], Rc);

    // Expected number of emitted photon over the time interval dt
    value_t dn = m_nc * gamma / Rc;
    value_t u = rng.uniform<value_t>();
    if (u < dn) {
      // Draw photon energy. e0 is our rescaling parameter in action
      value_t e_c = m_e0 * cube(gamma) / Rc;
      value_t eph = m_sync_module.gen_curv_photon(e_c, gamma, rng);
      if (eph > gamma - 1.01f) {
        eph = gamma - 1.01f;
      }

      // Energy loss over the time interval dt.
      // value_t dE = 2.0f / 3.0f * m_re / square(Rc) * square(square(gamma)) *
      // dt;
      // value_t dE = m_e0 * m_nc / square(Rc) * square(square(gamma)) * dt;

      // printf("e_c is %f, eph is %f\n", e_c, eph);

      // Do not allow gamma to go below 1
      // dE = std::min(dE, gamma - 1.01f);
      value_t Ef = gamma - eph;
      value_t p_new =
          math::sqrt(square(Ef) - 1.0f);

      ptc.p1[tid] = p_new * p1 / p;
      ptc.p2[tid] = p_new * p2 / p;
      ptc.p3[tid] = p_new * p3 / p;
      ptc.E[tid] = Ef;

      // printf("Current particle energy is %f, emitted eph of %f\n", Ef, eph);

      // TODO: Refine criterion for photon to potentially convert to pair
      // if (eph > 2.1f) {
      value_t sinth_max = grid.sizes[2] / Rc;
      value_t chi_max = 0.5 * eph * m_zeta * B_mag/m_BQ * sinth_max;
      // printf("sinth_max is %f, chi_max is %f\n", sinth_max, chi_max);
      if (chi_max > 0.05f && eph > 2.01f) {
        // value_t pi = std::sqrt(p1 * p1 + p2 * p2 + p3 * p3);

        size_t offset = ph_num + atomic_add(ph_pos, 1);
        ph.x1[offset] = ptc.x1[tid];
        ph.x2[offset] = ptc.x2[tid];
        ph.x3[offset] = ptc.x3[tid];
        ph.p1[offset] = eph * p1 / p;
        ph.p2[offset] = eph * p2 / p;
        ph.p3[offset] = eph * p3 / p;
        ph.E[offset] = eph;
        ph.weight[offset] = ptc.weight[tid];
        ph.cell[offset] = ptc.cell[tid];
        ph.path_left[offset] = 0.0f;

        return offset;
      }
      // TODO: tally the untracked photons
    }

    return 0;
  }

  HOST_DEVICE size_t produce_pair(const Grid<Conf::dim, value_t> &grid,
                                  const extent_t<Conf::dim> &ext, ph_ptrs &ph,
                                  size_t tid, ptc_ptrs &ptc, size_t ptc_num,
                                  unsigned long long int *ptc_pos, rng_t &rng,
                                  value_t dt) {
    // Get the magnetic field vector at the particle location
    auto cell = ph.cell[tid];
    auto idx = Conf::idx(cell, ext);
    auto x = vec_t<value_t, 3>(ph.x1[tid], ph.x2[tid], ph.x3[tid]);
    auto p = vec_t<value_t, 3>(ph.p1[tid], ph.p2[tid], ph.p3[tid]);
    auto pos = get_pos(idx, ext);

    // x_global gives the cartesian coordinate of the photon.
    auto x_global = grid.pos_global(pos, x);

    value_t r = math::sqrt(square(x_global[0]) + square(x_global[1]) +
                           square(x_global[2] + m_Rstar / m_rpc));
    value_t r_max = r / (1.0f - square((x_global[2] + m_Rstar / m_rpc) / r));
    // if (x_global[2] <= (grid.guard[2] + 5) * grid.delta[2] || p[2] < 0.0f) {
    if (x_global[2] <= (grid.guard[2] + 1) * grid.delta[2]
        || r_max / m_Rstar < 1.2f / m_omega) {
        // || p[2] < 0.0f) {
      return 0;
    }

    vec_t<value_t, 3> B;
    auto interp = interp_t<1, Conf::dim>{};
    B[0] = interp(x, m_B[0], idx, ext, stagger_t(0b001));
    B[1] = interp(x, m_B[1], idx, ext, stagger_t(0b010));
    B[2] = interp(x, m_B[2], idx, ext, stagger_t(0b100));

    // Compute the angle between photon and B field and compute the quantum parameter chi
    // value_t chi = quantum_chi(p, B, m_BQ);
    value_t B_mag = math::sqrt(B.dot(B));
    value_t eph = ph.E[tid];
    auto pxB = cross(p, B);
    value_t sinth = math::abs(math::sqrt(pxB.dot(pxB)) / B_mag / eph);
    // Note here that eph is multiplied by zeta. This is rescaling parameter in action
    value_t prob = magnetic_pair_production_rate(B_mag/m_BQ, m_zeta * eph, sinth, m_rpc / m_Rstar) * dt;
    value_t chi = 0.5f * m_zeta * eph * B_mag/m_BQ * sinth;
    // printf("sinth is %f, path is %f, eph is %f, prob is %f, chi is %f\n", sinth, ph.path_left[tid], eph, prob,
    //        0.5f * eph * B_mag/m_BQ * sinth * m_zeta);

    value_t u = rng.uniform<value_t>();
    if (u < prob && eph * sinth * m_zeta > 2.01f) {
    // if (u < prob && eph * sinth > 2.01f) {
    // if (u < prob) {
      // Actually produce the electron-positron pair
      size_t offset = ptc_num + atomic_add(ptc_pos, 2);
      size_t offset_e = offset;
      size_t offset_p = offset + 1;

      value_t p_ptc = math::sqrt(0.25f - 1.0f / square(eph)) * eph;
      // Immediately cool to zero magnetic moment and reduce Lorentz factor as
      // needed
      value_t gamma = 0.5f * eph;
      if (sinth > TINY && gamma > 1.0f / sinth) {
        gamma = 1.0f / sinth;
        if (gamma < 1.01f) gamma = 1.01;
        p_ptc = math::sqrt(gamma * gamma - 1.0f);
      }

      ptc.x1[offset_e] = ptc.x1[offset_p] = x[0];
      ptc.x2[offset_e] = ptc.x2[offset_p] = x[1];
      ptc.x3[offset_e] = ptc.x3[offset_p] = x[2];

      // Particle momentum is along B, direction is inherited from initial
      // photon direction
      value_t sign = sgn(p.dot(B));
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

#endif  // _CURVATURE_EMISSION_SCHEME_POLAR_CAP_H_
