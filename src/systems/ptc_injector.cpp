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

#include "ptc_injector.h"
#include "framework/config.h"
#include "framework/environment.h"

namespace Aperture {

template <typename Conf>
ptc_injector<Conf>::ptc_injector(sim_environment& env, const grid_t<Conf>& grid,
                                 const vec_t<value_t, Conf::dim>& lower,
                                 const vec_t<value_t, Conf::dim>& size,
                                 int inj_rate, value_t inj_weight)
    : system_t(env),
      m_grid(grid),
      m_inj_rate(inj_rate),
      m_inj_weight(inj_weight) {
  // Figure out the cell corresponding to the given region
  for (int n = 0; n < Conf::dim; n++) {
    if (lower[n] < m_grid.lower[n] + m_grid.sizes[n] &&
        m_grid.lower[n] <= lower[n] + size[n] ) {
      m_inj_begin[n] = std::round(std::max(lower[n] - m_grid.lower[n], value_t(0.0)) *
                           m_grid.inv_delta[n]) +
                       m_grid.guard[n];
      m_inj_ext[n] =
          std::round(std::min(lower[n] + size[n] - m_grid.lower[n], m_grid.sizes[n]) *
                     m_grid.inv_delta[n]);
    } else {
      m_inj_begin[n] = 0;
      m_inj_ext[n] = 0;
    }
  }
}

template <typename Conf>
void
ptc_injector<Conf>::init() {
  m_env.get_data("particles", &ptc);
  // m_env.get_data("B", &B);

  // m_env.params().get_value("target_sigma", m_target_sigma);
}

template <typename Conf>
void
ptc_injector<Conf>::update(double dt, uint32_t step) {}

template <typename Conf>
void
ptc_injector<Conf>::register_data_components() {}

template class ptc_injector<Config<2>>;

}  // namespace Aperture
