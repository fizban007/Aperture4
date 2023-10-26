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

#pragma once

#include "framework/environment.h"
#include "grid_ks.h"
#include "systems/physics/metric_kerr_schild.hpp"
#include "utils/gauss_quadrature.h"
#include "utils/logger.h"
#include "utils/timer.h"

namespace Aperture {

template <typename Conf>
grid_ks_t<Conf>::grid_ks_t() {
  initialize();
}

// template <typename Conf>
// grid_ks_t<Conf>::grid_ks_t(const domain_comm<Conf> &comm) :
// grid_t<Conf>(comm) {
//   initialize();
// }

template <typename Conf>
void
grid_ks_t<Conf>::initialize() {
  sim_env().params().get_value("bh_spin", a);

  Logger::print_info("In grid, a is {}", a);

  for (int i = 0; i < 3; i++) {
    m_Ab[i].resize(this->extent());
    m_Ad[i].resize(this->extent());
  }
  m_ag11dr_e.resize(this->extent());
  m_ag11dr_h.resize(this->extent());
  m_ag13dr_e.resize(this->extent());
  m_ag13dr_h.resize(this->extent());
  m_ag13dr_d.resize(this->extent());
  m_ag13dr_b.resize(this->extent());
  m_ag22dth_e.resize(this->extent());
  m_ag22dth_h.resize(this->extent());
  m_gbetadth_e.resize(this->extent());
  m_gbetadth_h.resize(this->extent());
  m_gbetadth_d.resize(this->extent());
  m_gbetadth_b.resize(this->extent());

#ifdef GPU_ENABLED
  using exec_tag = exec_tags::device;
#else
  using exec_tag = exec_tags::host;
#endif
  for (int i = 0; i < 3; i++) {
    ptrs.Ab[i] = adapt(exec_tag{}, std::cref(m_Ab[i]).get());
    ptrs.Ad[i] = adapt(exec_tag{}, std::cref(m_Ad[i]).get());
  }
  ptrs.ag11dr_e = adapt(exec_tag{}, std::cref(m_ag11dr_e).get());
  ptrs.ag11dr_h = adapt(exec_tag{}, std::cref(m_ag11dr_h).get());
  ptrs.ag13dr_e = adapt(exec_tag{}, std::cref(m_ag13dr_e).get());
  ptrs.ag13dr_h = adapt(exec_tag{}, std::cref(m_ag13dr_h).get());
  ptrs.ag13dr_d = adapt(exec_tag{}, std::cref(m_ag13dr_d).get());
  ptrs.ag13dr_b = adapt(exec_tag{}, std::cref(m_ag13dr_b).get());
  ptrs.ag22dth_e = adapt(exec_tag{}, std::cref(m_ag22dth_e).get());
  ptrs.ag22dth_h = adapt(exec_tag{}, std::cref(m_ag22dth_h).get());
  ptrs.gbetadth_e = adapt(exec_tag{}, std::cref(m_gbetadth_e).get());
  ptrs.gbetadth_h = adapt(exec_tag{}, std::cref(m_gbetadth_h).get());
  ptrs.gbetadth_d = adapt(exec_tag{}, std::cref(m_gbetadth_d).get());
  ptrs.gbetadth_b = adapt(exec_tag{}, std::cref(m_gbetadth_b).get());

  timer::stamp();
  compute_coef();
  timer::show_duration_since_stamp("Computing Kerr-Schild coefficients", "ms");
}

// TODO: This only works for 2D for now?
template <typename Conf>
void
grid_ks_t<Conf>::compute_coef() {
  auto ext = this->extent();

  for (auto idx : m_Ab[0].indices()) {
    using namespace Metric_KS;

    auto pos = get_pos(idx, ext);

    double r = radius(this->template coord<0>(pos[0], false));
    double r_s = radius(this->template coord<0>(pos[0], true));
    double th = theta(this->template coord<1>(pos[1], false));
    double th_s = theta(this->template coord<1>(pos[1], true));

    if (math::abs(th_s) < TINY) th_s = 0.01 * this->delta[1];
    if (math::abs(M_PI - th_s) < TINY) th_s = M_PI - 0.01 * this->delta[1];

    double dth = (theta(this->template coord<1>(pos[1] + 1, true)) - th_s);
    m_Ab[0][idx] =
        gauss_quad([this, r_s](auto x) { return sqrt_gamma(a, r_s, x); }, th_s,
                   theta(this->template coord<1>(pos[1] + 1, true)));
        // sqrt_gamma(a, r_s, th) * dth;
    if (m_Ab[0][idx] != m_Ab[0][idx]) {
      Logger::print_err("m_Ab0 at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    double dr = (radius(this->template coord<0>(pos[0] + 1, true)) - r_s);
    m_Ab[1][idx] =
        gauss_quad([this, th_s](auto x) { return sqrt_gamma(a, x, th_s); }, r_s,
                   radius(this->template coord<0>(pos[0] + 1, true)));
        // sqrt_gamma(a, r, th_s) * dr;
    if (m_Ab[1][idx] != m_Ab[1][idx]) {
      Logger::print_err("m_Ab1 at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    m_Ab[2][idx] =
        gauss_quad(
        [this, r_s, pos](auto x) {
          return gauss_quad([this, x](auto y) { return sqrt_gamma(a, y, x); },
                            r_s,
                            radius(this->template coord<0>(pos[0] + 1, true)));
        },
        th_s, theta(this->template coord<1>(pos[1] + 1, true)));
        // sqrt_gamma(a, r, th) * dr * dth;
    if (m_Ab[2][idx] != m_Ab[2][idx]) {
      Logger::print_err("m_Ab2 at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    dth = th - theta(this->template coord<1>(pos[1] - 1, false));
    if (pos[1] == this->guard[1] && th_s < 0.1 * this->delta[1]) {
      m_Ad[0][idx] =
          2.0 * gauss_quad([this, r](auto x) { return sqrt_gamma(a, r, x); },
                           0.0f, th);
    } else if (pos[1] == this->dims[1] - this->guard[1] &&
               math::abs(th_s - M_PI) < 0.1 * this->delta[1]) {
      m_Ad[0][idx] =
          2.0 * gauss_quad([this, r](auto x) { return sqrt_gamma(a, r, x); },
                           theta(this->template coord<1>(pos[1] - 1, false)),
                           M_PI);
    } else {
      m_Ad[0][idx] =
          gauss_quad([this, r](auto x) { return sqrt_gamma(a, r, x); },
                     theta(this->template coord<1>(pos[1] - 1, false)), th);
          // sqrt_gamma(a, r, th_s) * dth;
    }
    if (m_Ad[0][idx] != m_Ad[0][idx]) {
      Logger::print_err("m_Ad0 at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    dr = r - radius(this->template coord<0>(pos[0] - 1, false));
    m_Ad[1][idx] =
        gauss_quad([this, th](auto x) { return sqrt_gamma(a, x, th); },
                   radius(this->template coord<0>(pos[0] - 1, false)), r);
        // sqrt_gamma(a, r_s, th) * dr;
    if (m_Ad[1][idx] != m_Ad[1][idx]) {
      Logger::print_err("m_Ad1 at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    if (pos[1] == this->guard[1] && th_s < 0.1 * this->delta[1]) {
      m_Ad[2][idx] =
          2.0 * gauss_quad(
                    [this, r, pos](auto x) {
                      return gauss_quad(
                          [this, x](auto y) { return sqrt_gamma(a, y, x); },
                          radius(this->template coord<0>(pos[0] - 1, false)),
                          r);
                    },
                    0.0f, th);
    } else if (pos[1] == this->dims[1] - this->guard[1] &&
               math::abs(th_s - M_PI) < 0.1 * this->delta[1]) {
      m_Ad[2][idx] =
          2.0 * gauss_quad(
                    [this, r, pos](auto x) {
                      return gauss_quad(
                          [this, x](auto y) { return sqrt_gamma(a, y, x); },
                          radius(this->template coord<0>(pos[0] - 1, false)),
                          r);
                    },
                    th, M_PI);
    } else {
      m_Ad[2][idx] =
          gauss_quad(
          [this, r, pos](auto x) {
            return gauss_quad(
                [this, x](auto y) { return sqrt_gamma(a, y, x); },
                radius(this->template coord<0>(pos[0] - 1, false)), r);
          },
          theta(this->template coord<1>(pos[1] - 1, false)), th);
          // sqrt_gamma(a, r_s, th_s) * dr * dth;
    }
    if (m_Ad[2][idx] != m_Ad[2][idx]) {
      Logger::print_err("m_Ad2 at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    m_ag11dr_h[idx] =
        gauss_quad([this, th](auto x) { return ag_11(a, x, th); },
                   radius(this->template coord<0>(pos[0] - 1, false)), r);
    if (m_ag11dr_h[idx] != m_ag11dr_h[idx]) {
      Logger::print_err("m_ag11dr_h at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    m_ag11dr_e[idx] =
        gauss_quad([this, th_s](auto x) { return ag_11(a, x, th_s); }, r_s,
                   radius(this->template coord<0>(pos[0] + 1, true)));
    if (m_ag11dr_e[idx] != m_ag11dr_e[idx]) {
      Logger::print_err("m_ag11dr_e at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    m_ag13dr_h[idx] =
        gauss_quad([this, th](auto x) { return ag_13(a, x, th); },
                   radius(this->template coord<0>(pos[0] - 1, false)), r);
    if (m_ag13dr_h[idx] != m_ag13dr_h[idx]) {
      Logger::print_err("m_ag13dr_h at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    m_ag13dr_b[idx] =
        gauss_quad([this, th](auto x) { return ag_13(a, x, th); },
                   r_s, radius(this->template coord<0>(pos[0] + 1, true)));
    if (m_ag13dr_b[idx] != m_ag13dr_b[idx]) {
      Logger::print_err("m_ag13dr_b at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    m_ag13dr_e[idx] =
        gauss_quad([this, th_s](auto x) { return ag_13(a, x, th_s); }, r_s,
                   radius(this->template coord<0>(pos[0] + 1, true)));
    if (m_ag13dr_e[idx] != m_ag13dr_e[idx]) {
      Logger::print_err("m_ag13dr_e at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    m_ag13dr_d[idx] =
        gauss_quad([this, th_s](auto x) { return ag_13(a, x, th_s); },
                   radius(this->template coord<0>(pos[0] - 1, false)), r);
    if (m_ag13dr_d[idx] != m_ag13dr_d[idx]) {
      Logger::print_err("m_ag13dr_d at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    m_ag22dth_h[idx] =
        gauss_quad([this, r](auto x) { return ag_22(a, r, x); },
                   theta(this->template coord<1>(pos[1] - 1, false)), th);
    if (m_ag22dth_h[idx] != m_ag22dth_h[idx]) {
      Logger::print_err("m_ag22dth_h at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    m_ag22dth_e[idx] =
        gauss_quad([this, r_s](auto x) { return ag_22(a, r_s, x); }, th_s,
                   theta(this->template coord<1>(pos[1] + 1, true)));
    if (m_ag22dth_e[idx] != m_ag22dth_e[idx]) {
      Logger::print_err("m_ag22dth_e at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    m_gbetadth_h[idx] =
        gauss_quad([this, r](auto x) { return sq_gamma_beta(a, r, x); },
                   theta(this->template coord<1>(pos[1] - 1, false)), th);
    if (m_gbetadth_h[idx] != m_gbetadth_h[idx]) {
      Logger::print_err("m_gbetadth_h at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    m_gbetadth_b[idx] =
        gauss_quad([this, r](auto x) { return sq_gamma_beta(a, r, x); },
                   th_s, theta(this->template coord<1>(pos[1] + 1, true)));
    if (m_gbetadth_b[idx] != m_gbetadth_b[idx]) {
      Logger::print_err("m_gbetadth_b at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    m_gbetadth_e[idx] =
        gauss_quad([this, r_s](auto x) { return sq_gamma_beta(a, r_s, x); },
                   th_s, theta(this->template coord<1>(pos[1] + 1, true)));
    if (m_gbetadth_e[idx] != m_gbetadth_e[idx]) {
      Logger::print_err("m_gbetadth_e at ({}, {}) is NaN!", pos[0], pos[1]);
    }

    m_gbetadth_d[idx] =
        gauss_quad([this, r_s](auto x) { return sq_gamma_beta(a, r_s, x); },
                   theta(this->template coord<1>(pos[1] - 1, false)), r);
    if (m_gbetadth_d[idx] != m_gbetadth_d[idx]) {
      Logger::print_err("m_gbetadth_d at ({}, {}) is NaN!", pos[0], pos[1]);
    }

  }

#ifdef GPU_ENABLED
  for (int i = 0; i < 3; i++) {
    m_Ab[i].copy_to_device();
    m_Ad[i].copy_to_device();
  }
  m_ag11dr_e.copy_to_device();
  m_ag11dr_h.copy_to_device();
  m_ag13dr_e.copy_to_device();
  m_ag13dr_h.copy_to_device();
  m_ag13dr_d.copy_to_device();
  m_ag13dr_b.copy_to_device();
  m_ag22dth_e.copy_to_device();
  m_ag22dth_h.copy_to_device();
  m_gbetadth_e.copy_to_device();
  m_gbetadth_h.copy_to_device();
  m_gbetadth_b.copy_to_device();
  m_gbetadth_d.copy_to_device();
#endif
}

}  // namespace Aperture
