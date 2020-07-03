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
#include "framework/environment.h"
#include "framework/config.h"

namespace Aperture {

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
ptc_injector<Conf>::register_data_components() {
}

template class ptc_injector<Config<2>>;

}
