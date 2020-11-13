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
void
ptc_injector<Conf>::add_injector(const vec_t<value_t, Conf::dim> &lower,
                                 const vec_t<value_t, Conf::dim> &size,
                                 value_t inj_rate, value_t inj_weight) {
  injector_params new_injector{};
  new_injector.weight = inj_weight;
  for (int n = 0; n < Conf::dim; n++) {
    if (lower[n] < m_grid.lower[n] + m_grid.sizes[n] &&
        m_grid.lower[n] <= lower[n] + size[n]) {
      new_injector.begin[n] =
          std::round(std::max(lower[n] - m_grid.lower[n], value_t(0.0)) *
          // std::round(std::max(lower[n] - m_grid.lower[n], value_t(-m_grid.guard[n] * m_grid.delta[n])) *
                     m_grid.inv_delta[n]) +
          m_grid.guard[n];
      // FIXME: Ext calculation still has problems
      new_injector.ext[n] =
          std::round(std::min(size[n], m_grid.sizes[n]) * m_grid.inv_delta[n]);
      new_injector.ext[n] = std::min(
          new_injector.ext[n],
          m_grid.reduced_dim(n) + m_grid.guard[n] - new_injector.begin[n]);
    } else {
      new_injector.begin[n] = 0;
      new_injector.ext[n] = 0;
    }
    new_injector.ext.get_strides();
  }
  Logger::print_info_all("Injector begin is ({}, {}), extent is ({}, {})",
                         new_injector.begin[0], new_injector.begin[1],
                         new_injector.ext[0], new_injector.ext[1]);
  if (inj_rate > 1.0f) {
    new_injector.interval = 1;
    new_injector.num = std::round(inj_rate);
  } else {
    new_injector.interval = std::round(1.0 / inj_rate);
    new_injector.num = 1;
  }
  Logger::print_info("Injector interval is {}, num is {}",
                     new_injector.interval, new_injector.num);
  m_injectors.push_back(std::move(new_injector));
  m_weight_funcs.push_back(nullptr);
}

template <typename Conf>
void
ptc_injector<Conf>::init() {
  m_env.get_data("particles", &ptc);

  m_ptc_density = make_multi_array<value_t>(m_grid.extent(), MemType::host_only);
  // m_env.params().get_value("target_sigma", m_target_sigma);
}

template <typename Conf>
void
ptc_injector<Conf>::update(double dt, uint32_t step) {}

template <typename Conf>
void
ptc_injector<Conf>::register_data_components() {}

template class ptc_injector<Config<1>>;
template class ptc_injector<Config<2>>;
template class ptc_injector<Config<3>>;

}  // namespace Aperture
