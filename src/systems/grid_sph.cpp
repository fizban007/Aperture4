/*
 * Copyright (c) 2020 Alex Chen.
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

#include "grid_sph.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/domain_comm.h"
#include "utils/range.hpp"

namespace Aperture {

template <typename Conf>
grid_sph_t<Conf>::~grid_sph_t() {}

// TODO: This whole thing only works for 2D right now
template <typename Conf>
void
grid_sph_t<Conf>::compute_coef() {
  double r_g = 0.0;
  sim_env().params().get_value("compactness", r_g);

  auto ext = this->extent();

  for (auto idx : range(Conf::begin(ext), Conf::end(ext))) {
    auto pos = idx.get_pos();

    double r = radius(this->template pos<0>(pos[0], false));
    double r_minus = radius(this->template pos<0>(pos[0] - 1, false));
    double rs = radius(this->template pos<0>(pos[0], true));
    double rs_plus = radius(this->template pos<0>(pos[0] + 1, true));

    double th = theta(this->template pos<1>(pos[1], false));
    double th_minus = theta(this->template pos<1>(pos[1] - 1, false));
    double ths = theta(this->template pos<1>(pos[1], true));
    double ths_plus = theta(this->template pos<1>(pos[1] + 1, true));

    // Length elements for E field
    this->m_le[0][idx] = rs_plus - rs;
    this->m_le[1][idx] = rs * this->delta[1];
    if constexpr (Conf::dim == 2) {
      this->m_le[2][idx] = rs * std::sin(ths);
    } else if (Conf::dim == 3) {
      this->m_le[2][idx] = rs * std::sin(ths) * this->delta[2];
    }

    // Length elements for B field
    this->m_lb[0][idx] = r - r_minus;
    this->m_lb[1][idx] = r * this->delta[1];
    if constexpr (Conf::dim == 2) {
      this->m_lb[2][idx] = r * std::sin(th);
    } else if (Conf::dim == 3) {
      this->m_lb[2][idx] = r * std::sin(th) * this->delta[2];
    }

    // Area elements for E field
    this->m_Ae[0][idx] = r * r * (std::cos(th_minus) - std::cos(th));
    if (std::abs(ths) < 0.1 * this->delta[1]) {
      this->m_Ae[0][idx] = r * r * 2.0 * (1.0 - std::cos(0.5 * this->delta[1]));
    } else if (std::abs(ths - M_PI) < 0.1 * this->delta[1]) {
      this->m_Ae[0][idx] = r * r * 2.0 * (1.0 - std::cos(0.5 * this->delta[1]));
    }
    if constexpr (Conf::dim == 3) {
      this->m_Ae[0][idx] *= this->delta[2];
    }

    this->m_Ae[1][idx] = 0.5 * (square(r) - square(r_minus)) * std::sin(th);
    if constexpr (Conf::dim == 3) {
      this->m_Ae[1][idx] *= this->delta[2];
    }

    this->m_Ae[2][idx] =
        (cube(r) - cube(r_minus)) / 3.0 * (std::cos(th_minus) - std::cos(th));

    // Area elements for B field
    this->m_Ab[0][idx] = rs * rs * (std::cos(ths) - std::cos(ths_plus));
    if constexpr (Conf::dim == 3) {
      this->m_Ab[0][idx] *= this->delta[2];
    }

    if (std::abs(ths) > 0.1 * this->delta[1] &&
        std::abs(ths - M_PI) > 0.1 * this->delta[1])
      this->m_Ab[1][idx] = 0.5 * (square(rs_plus) - square(rs)) * std::sin(ths);
    else
      this->m_Ab[1][idx] = TINY;
    if constexpr (Conf::dim == 3) {
      this->m_Ab[1][idx] *= this->delta[2];
    }

    this->m_Ab[2][idx] =
        (cube(rs_plus) - cube(rs)) / 3.0 * (std::cos(ths) - std::cos(ths_plus));

    // Volume element, defined at cell vertices
    this->m_dV[idx] = (cube(r) - cube(r_minus)) / 3.0 *
                      (std::cos(th_minus) - std::cos(th)) /
                      (this->delta[0] * this->delta[1]);

    if (std::abs(ths) < 0.1 * this->delta[1] ||
        std::abs(ths - M_PI) < 0.1 * this->delta[1]) {
      this->m_dV[idx] = (cube(r) - cube(r_minus)) * 2.0 / 3.0 *
                        (1.0 - std::cos(0.5 * this->delta[1])) /
                        (this->delta[0] * this->delta[1]);
    }
    if constexpr (Conf::dim == 3) {
      this->m_dV[idx] /= this->delta[2];
    }
  }

  for (int i = 0; i < 3; i++) {
    this->m_le[i].copy_to_device();
    this->m_lb[i].copy_to_device();
    this->m_Ae[i].copy_to_device();
    this->m_Ab[i].copy_to_device();
  }
  this->m_dV.copy_to_device();
}

template class grid_sph_t<Config<2>>;
template class grid_sph_t<Config<3>>;

}  // namespace Aperture
