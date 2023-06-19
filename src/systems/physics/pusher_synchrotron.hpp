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
    if (sim_env().params().has("sigma")) {
      sim_env().params().get_value("sigma", sigma);
      sim_env().params().get_value("guide_field", Bg);
      if (Bg > 0.0f) {
        sigma = sigma + Bg * Bg * sigma;
      }
    }
    // The cooling coefficient is effectively 2r_e t_0/3c in the dimensionless
    // units, where t_0 is the time unit. In the reconnection setup, t_0 is
    // 1/omega_p, sigma = B_tot^2, so this makes sense.
    if (!m_use_cooling) {
      m_cooling_coef = 0.0f;
    } else {
      m_cooling_coef = 2.0f * m_sync_compactness / sigma;
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
    // If the config file specifies a synchrotron cooling coefficient, then we
    // use that instead. Sync cooling coefficient is roughly 2l_B/B^2, where l_B
    // is the magnetic compactness for one length unit
    if (sim_env().params().has("sync_cooling_coef")) {
      sim_env().params().get_value("sync_cooling_coef", m_cooling_coef);
    }

    auto sync_loss = sim_env().register_data<scalar_field<Conf>>(
        "sync_loss", m_grid, field_type::cell_centered,
        MemType::host_device);
    // m_sync_loss = sync_loss->dev_ndptr();
#ifdef GPU_ENABLED
    m_sync_loss = sync_loss->dev_ndptr();
#else
    m_sync_loss = sync_loss->host_ndptr();
#endif
    sync_loss->reset_after_output(true);

    // sync_loss_total accounts for the radiation that are excluded in the spectrum
    auto sync_loss_total = sim_env().register_data<scalar_field<Conf>>(
        "sync_loss_total", m_grid, field_type::cell_centered,
        MemType::host_device);
    // m_sync_loss = sync_loss->dev_ndptr();
#ifdef GPU_ENABLED
    m_sync_loss_total = sync_loss_total->dev_ndptr();
#else
    m_sync_loss_total = sync_loss_total->host_ndptr();
#endif
    sync_loss_total->reset_after_output(true);

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
    // value_t th_loss =
    //     context.weight * (4.0f / 3.0f) * m_sync_compactness * gamma * p * dt / context.q;
    // Turn off synchrotron cooling for gamma < 1.001
    if (gamma <= 1.0001f || check_flag(flag, PtcFlag::ignore_radiation) ||
        m_cooling_coef == 0.0f) {
      m_pusher(context.p[0], context.p[1], context.p[2], context.gamma,
               context.E[0], context.E[1], context.E[2], context.B[0],
               context.B[1], context.B[2], dt * context.q / context.m * 0.5f,
               decltype(context.q)(dt));
    } else {
      // printf("p1: %f, p2: %f, p3: %f\n", p1, p2, p3);
      m_pusher(p1, p2, p3, gamma, context.E[0], context.E[1], context.E[2],
               context.B[0], context.B[1], context.B[2],
               dt * context.q / context.m * 0.5f, decltype(context.q)(dt));

      iterate(context.x, context.p, context.E, context.B, context.q / context.m,
              m_cooling_coef, dt);
      // printf("p1: %f, p2: %f, p3: %f\n", context.p[0], context.p[1], context.p[2]);
      p = math::sqrt(context.p.dot(context.p));
      context.gamma = math::sqrt(1.0f + p * p);
    }
    auto idx = Conf::idx(pos, ext);
    // Need to divide by q here because context.weight has q in it
    loss = context.weight * max(gamma - context.gamma, 0.0f) / context.q;
    // context.weight * max(gamma - context.gamma, 0.0f);
    atomic_add(&m_sync_loss_total[idx], loss);
    // printf("gamma is %f, sync loss is %f, new gamma is %f, weight is %f\n",
    //        gamma, th_loss, context.gamma, context.weight);
    if (!check_flag(context.flag, PtcFlag::exclude_from_spectrum)) {
      atomic_add(&m_sync_loss[idx], loss);
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
        index_t<3> pos_ph_dist(th_bin, phi_bin, bin);
        atomic_add(
            &m_angle_dist_ptr[default_idx_t<3>(pos_ph_dist, m_ext_ph_dist)],
            loss);
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
  mutable ndptr<value_t, Conf::dim> m_sync_loss_total;
  int m_num_bins = 512;
  value_t m_BQ = 1e5;
  float m_lim_lower = 1.0e-6;
  float m_lim_upper = 1.0e2;
  int m_downsample = 16;
  int m_ph_nth = 32;
  int m_ph_nphi = 64;
  extent_t<3> m_ext_ph_dist;

  mutable ndptr<float, Conf::dim + 1> m_spec_ptr;
  mutable ndptr<value_t, 3> m_angle_dist_ptr;
  sync_emission_helper_t m_sync;
};

}  // namespace Aperture

#endif  // PUSHER_SYNCHROTRON_H_
