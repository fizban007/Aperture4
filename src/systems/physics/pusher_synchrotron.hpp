/*
 * Copyright (c) 2023 Alex Chen.
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

#ifndef PUSHER_SYNCHROTRON_H_
#define PUSHER_SYNCHROTRON_H_

#include "data/fields.h"
#include "data/multi_array_data.hpp"
#include "data/phase_space.hpp"
#include "data/scalar_data.hpp"
#include "framework/environment.h"
#include "systems/physics/sync_emission_helper.hpp"
#include "systems/sync_curv_emission.h"


namespace Aperture {

template <typename Conf>
class pusher_synchrotron {
 public:
  using value_t = typename Conf::value_t;
  using vec3 = vec_t<value_t, 3>;
  using grid_type = grid_t<Conf>;

  pusher_synchrotron(const grid_t<Conf>& grid) : m_grid(grid) {}
  ~pusher_synchrotron() = default;

  void init() {
    value_t t_cool = 100.0f, sigma = 10.0f, Bg = 0.0f;
    sim_env().params().get_value("cooling", m_use_cooling);
    sim_env().params().get_value("sync_compactness", m_sync_compactness);

    sim_env().params().get_value("cooling_time", t_cool);
    if (m_sync_compactness < 0.0f) {
      m_sync_compactness = 1.0f / t_cool;
    }
    sim_env().params().get_value("sigma", sigma);
    sim_env().params().get_value("guide_field", Bg);
    if (Bg > 0.0f) {
      sigma = sigma + Bg * Bg * sigma;
    }
    // The cooling coefficient is effectively 2r_e\omega_p/3c in the
    // dimensionless units. In the reconnection setup, sigma = B_tot^2, so this
    // makes sense.
    if (!m_use_cooling) {
      m_cooling_coef = 0.0f;
    } else {
      m_cooling_coef = 2.0f * m_sync_compactness / sigma;
    }

    // If the config file specifies a synchrotron cooling coefficient, then we
    // use that instead. Sync cooling coefficient is roughly 2l_B/B^2
    if (sim_env().params().has("sync_cooling_coef")) {
      sim_env().params().get_value("sync_cooling_coef", m_cooling_coef);
    }
    // If the config file specifies a synchrotron gamma_rad, then we
    // use that to determine sync_compactness.
    if (sim_env().params().has("sync_gamma_rad") &&
        sim_env().params().has("sigma")) {
      value_t sync_gamma_rad;
      sim_env().params().get_value("sync_gamma_rad", sync_gamma_rad);
      m_sync_compactness =
          0.3f * math::sqrt(sigma) / (square(sync_gamma_rad) * 4.0f);
      m_cooling_coef = 2.0f * m_sync_compactness / sigma;
    }

    auto sync_loss = sim_env().register_data<scalar_field<Conf>>(
        "sync_loss", m_grid, field_type::cell_centered, MemType::host_device);
    // m_sync_loss = sync_loss->dev_ndptr();
#ifdef GPU_ENABLED
    m_sync_loss = sync_loss->dev_ndptr();
#else
    m_sync_loss = sync_loss->host_ndptr();
#endif
    sync_loss->reset_after_output(true);

    // Initialize the spectrum related parameters
    sim_env().params().get_value("B_Q", m_BQ);
    sim_env().params().get_value("ph_num_bins", m_num_bins);
    sim_env().params().get_value("sync_spec_lower", m_lim_lower);
    sim_env().params().get_value("sync_spec_upper", m_lim_upper);
    sim_env().params().get_value("momentum_downsample", m_downsample);
    // Always use logarithmic bins
    m_lim_lower = math::log(m_lim_lower);
    m_lim_upper = math::log(m_lim_upper);

    auto photon_dist = sim_env().register_data<phase_space<Conf, 1>>(
        "sync_spectrum", m_grid, m_downsample, &m_num_bins, &m_lim_lower,
        &m_lim_upper, true, MemType::host_device);
    m_spec_ptr = photon_dist->data.dev_ndptr();
    photon_dist->reset_after_output(true);

    // initialize synchrotron module
    auto sync_module =
        sim_env().register_system<sync_curv_emission_t>(MemType::host_device);
    m_sync = sync_module->get_helper();

    // synchrotron angular distribution
    sim_env().params().get_value("ph_dist_n_th", m_ph_nth);
    sim_env().params().get_value("ph_dist_n_phi", m_ph_nphi);
    int ph_dist_interval = 10;
    sim_env().params().get_value("fld_output_interval", ph_dist_interval);
    // If no "ph_dist_interval" specified, we use fld_output_interval
    sim_env().params().get_value("ph_dist_interval", ph_dist_interval);

    auto photon_angular_dist =
        sim_env().register_data<multi_array_data<value_t, 3>>(
            "sync_dist", m_ph_nth, m_ph_nphi, m_num_bins);
    m_angle_dist_ptr = photon_angular_dist->dev_ndptr();
    photon_angular_dist->reset_after_output(true);
    photon_angular_dist->m_special_output_interval = ph_dist_interval;

    auto stokes_parameter_I =
        sim_env().register_data<multi_array_data<value_t, 3>>(
            "I", m_ph_nth, m_ph_nphi, m_eph_bins);
    I_ptr = stokes_parameter_I->dev_ndptr();
    stokes_parameter_I->reset_after_output(true);

    auto stokes_parameter_Q =
        sim_env().register_data<multi_array_data<value_t, 3>>(
            "Q", m_ph_nth, m_ph_nphi, m_eph_bins);
    Q_ptr = stokes_parameter_Q->dev_ndptr();
    stokes_parameter_Q->reset_after_output(true);

    auto stokes_parameter_U =
        sim_env().register_data<multi_array_data<value_t, 3>>(
            "U", m_ph_nth, m_ph_nphi, m_eph_bins);
    U_ptr = stokes_parameter_U->dev_ndptr();
    stokes_parameter_U->reset_after_output(true);
  }

  // Inline functions to be called in the particle update loop
  template <typename PtcContext, typename UIntT>
  HOST_DEVICE void push(const Grid<Conf::dim, value_t>& grid,
                        const extent_t<Conf::dim>& ext, PtcContext& context,
                        vec_t<UIntT, Conf::dim>& pos, value_t dt) const {
    value_t p1 = context.p[0];
    value_t p2 = context.p[1];
    value_t p3 = context.p[2];
    value_t gamma = context.gamma;
    value_t p = math::sqrt(p1 * p1 + p2 * p2 + p3 * p3);
    auto flag = context.flag;

    value_t loss = 0.0f;
    // Turn off synchrotron cooling for gamma < 1.0001
    if (gamma <= 1.0001f || check_flag(flag, PtcFlag::ignore_radiation) ||
        m_cooling_coef == 0.0f) {
      m_pusher(context.p[0], context.p[1], context.p[2], context.gamma,
               context.E[0], context.E[1], context.E[2], context.B[0],
               context.B[1], context.B[2], dt * context.q / context.m * 0.5f,
               decltype(context.q)(dt));
    } else {
      printf("p1: %f, p2: %f, p3: %f\n", p1, p2, p3);
      m_pusher(p1, p2, p3, gamma, context.E[0], context.E[1], context.E[2],
               context.B[0], context.B[1], context.B[2],
               dt * context.q / context.m * 0.5f, decltype(context.q)(dt));

      iterate(context.x, context.p, context.E, context.B, context.q / context.m,
              m_cooling_coef, dt);
      // printf("p1: %f, p2: %f, p3: %f\n", context.p[0], context.p[1],
      // context.p[2]);
      p = math::sqrt(context.p.dot(context.p));
      context.gamma = math::sqrt(1.0f + p * p);
      // Need to divide by q here because context.weight has q in it
      loss = context.weight * max(gamma - context.gamma, 0.0f) / context.q;
    }
    auto idx = Conf::idx(pos, ext);
    atomic_add(&m_sync_loss[idx], loss);

    // Compute synchrotron spectrum
    if (!check_flag(context.flag, PtcFlag::exclude_from_spectrum)) {
      auto aL = context.E + cross(context.p, context.B) / context.gamma;
      auto p = math::sqrt(context.p.dot(context.p));
      auto aL_perp = cross(aL, context.p) / p;
      value_t a_perp = math::sqrt(aL_perp.dot(aL_perp));
      value_t eph = m_sync.gen_sync_photon(context.gamma, a_perp, m_BQ,
                                           *context.local_state);
      if (eph > math::exp(m_lim_lower)) {
        value_t log_eph = clamp(math::log(max(eph, math::exp(m_lim_lower))),
                                m_lim_lower, m_lim_upper);
        auto ext_out = grid.extent_less() / m_downsample;
        auto ext_spec = extent_t<Conf::dim + 1>(m_num_bins, ext_out);
        index_t<Conf::dim + 1> pos_out(0, (pos - grid.guards()) / m_downsample);
        int bin = round((log_eph - m_lim_lower) / (m_lim_upper - m_lim_lower) *
                        (m_num_bins - 1));
        pos_out[0] = bin;
        atomic_add(&m_spec_ptr[default_idx_t<Conf::dim + 1>(pos_out, ext_spec)],
                   loss / eph);
        // atomic_add(&m_spec_ptr[default_idx_t<Conf::dim + 1>(pos_out,
        // ext_spec)], loss);

        // Simply deposit the photon direction along the particle direction,
        // without computing the 1/gamma cone
        value_t th = math::acos(context.p[2] / p);
        value_t phi = math::atan2(context.p[1], context.p[0]) + M_PI;
        int th_bin = round(th / M_PI * (m_ph_nth - 1));
        int phi_bin = round(phi * 0.5 / M_PI * (m_ph_nphi - 1));
        // int omega_bin = round(omega - omega_min / domega + 1);
        index_t<3> pos_ph_dist(th_bin, phi_bin, bin);
        atomic_add(
            &m_angle_dist_ptr[default_idx_t<3>(pos_ph_dist, m_ext_ph_dist)],
            loss);

        for (value_t log_eph = math::log(eph_min);
             log_eph <= math::log(eph_max); log_eph += deph) {
          value_t eph = math::exp(log_eph);

          // Stokes parameter
          vec3 x_prime = (-math::sin(phi), math::cos(phi), 0.0);
          vec3 y_prime = (math::cos(th) * cos(phi), math::cos(th) * sin(phi),
                          -math::sin(th));
          vec3 B_perp_prime =
              context.E + cross(context.p, context.B) -
              (context.p[0] * context.E[0] + context.p[1] * context.E[1] +
               context.p[2] * context.E[2]) *
                  context.p; /* perpendicular part of the Lorentz force */
          
          value_t B_perp_prime_mag =
              math::sqrt(B_perp_prime[0] * B_perp_prime[0] +
                         B_perp_prime[1] * B_perp_prime[1] +
                         B_perp_prime[2] * B_perp_prime[2]);
          value_t x_prime_mag =
              math::sqrt(x_prime[0] * x_prime[0] + x_prime[1] * x_prime[1] +
                         x_prime[2] * x_prime[2]);
          value_t y_prime_mag =
              math::sqrt(y_prime[0] * y_prime[0] + y_prime[1] * y_prime[1] +
                         y_prime[2] * y_prime[2]);
          value_t denominator =
              std::sqrt(std::pow(B_perp_prime_mag * B_perp_prime_mag +
                                     x_prime_mag * x_prime_mag,
                                 2) +
                        std::pow(B_perp_prime_mag * B_perp_prime_mag +
                                     y_prime_mag * y_prime_mag,
                                 2));
          value_t cos_chi = B_perp_prime.dot(y_prime) / denominator;
          value_t chi = std::acos(cos_chi);
          value_t eph_c = m_sync.e_c(context.gamma, a_perp, m_BQ);
          
          value_t zeta = eph / eph_c;
          index_t<3> Stokes_parameters_index(th_bin, phi_bin, eph);
          atomic_add(
              &I_ptr[default_idx_t<3>(Stokes_parameters_index, m_ext_ph_dist)],
              m_sync.Fx(zeta));
          atomic_add(
              &Q_ptr[default_idx_t<3>(Stokes_parameters_index, m_ext_ph_dist)],
              cos(2 * chi) * m_sync.Gx(zeta));
          atomic_add(
              &U_ptr[default_idx_t<3>(Stokes_parameters_index, m_ext_ph_dist)],
              sin(2 * chi) * m_sync.Gx(zeta));
        }
      }
    }
  }

  HD_INLINE vec3 rhs_x(const vec3& u, value_t dt) const {
    value_t gamma = math::sqrt(1.0f + u.dot(u));
    return u * (dt / gamma);
  }

  HD_INLINE vec3 rhs_u(const vec3& E, const vec3& B, const vec3& u,
                       value_t e_over_m, value_t cooling_coef,
                       value_t dt) const {
    vec3 result;
    value_t gamma = math::sqrt(1.0f + u.dot(u));
    vec3 Epbetaxb = E + cross(u, B) / gamma;

    result =
        e_over_m * Epbetaxb +
        cooling_coef *
            (cross(Epbetaxb, B) + E * u.dot(E) / gamma -
             u * (gamma * (Epbetaxb.dot(Epbetaxb) - square(u.dot(E) / gamma))));
    // u * (-gamma * (Epbetaxb.dot(Epbetaxb) - square(u.dot(E) / gamma)));

    return result * dt;
  }

  HD_INLINE vec3 sync_force(const vec3& E, const vec3& B, const vec3& u,
                            value_t e_over_m, value_t cooling_coef,
                            value_t dt) const {
    vec3 result;
    value_t gamma = math::sqrt(1.0f + u.dot(u));
    vec3 Epbetaxb = E + cross(u, B) / gamma;

    result =
        cooling_coef *
        (cross(Epbetaxb, B) + E * u.dot(E) / gamma -
         u * (gamma * (Epbetaxb.dot(Epbetaxb) - square(u.dot(E) / gamma))));
    // u * (-gamma * (Epbetaxb.dot(Epbetaxb) - square(u.dot(E) / gamma)));

    return result * dt;
  }

  HD_INLINE void iterate(vec3& x, vec3& u, const vec3& E, const vec3& B,
                         double e_over_m, double cooling_coef,
                         double dt) const {
    // vec3 x0 = x, x1 = x;
    vec3 u0 = u, u1 = u;

    for (int i = 0; i < 4; i++) {
      // x1 = x0 + rhs_x((u0 + u) * 0.5, dt);
      u1 = u0 + rhs_u(E, B, (u0 + u) * 0.5, e_over_m, cooling_coef, dt);
      // x = x1;
      u = u1;
    }
  }

 private:
  const grid_t<Conf>& m_grid;
  mutable typename Conf::pusher_t m_pusher;
  bool m_use_cooling = true;
  value_t m_cooling_coef = 0.0f;
  value_t m_sync_compactness = -1.0f;
  mutable ndptr<value_t, Conf::dim> m_sync_loss;
  int m_num_bins = 512;
  value_t m_BQ = 1e5;
  float m_lim_lower = 1.0e-6;
  float m_lim_upper = 1.0e2;
  int m_downsample = 16;
  int m_ph_nth = 32;
  int m_ph_nphi = 64;
  value_t coef = 3.79e-12;
  extent_t<3> m_ext_ph_dist;
  value_t eph_min = 1.0e-8;
  value_t eph_max = 1.0e3;
  int m_eph_bins = 100;
  // value_t domega = omega_max - omega_min / m_omega_bins;
  value_t deph = (math::log(eph_max) - math::log(eph_min)) / m_eph_bins;

  mutable ndptr<float, Conf::dim + 1> m_spec_ptr;
  mutable ndptr<value_t, 3> m_angle_dist_ptr;
  sync_emission_helper_t m_sync;
  mutable ndptr<value_t, 3> I_ptr;
  mutable ndptr<value_t, 3> Q_ptr;
  mutable ndptr<value_t, 3> U_ptr;
};

}  // namespace Aperture

#endif  // PUSHER_SYNCHROTRON_H_
