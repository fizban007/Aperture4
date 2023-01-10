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

#pragma once

#include "core/multi_array_exp.hpp"
#include "core/ndsubset.hpp"
#include "core/ndsubset_dev.hpp"
#include "field_solver_polar.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/helpers/finite_diff_helper.hpp"
#include "utils/double_buffer.h"
#include "utils/timer.h"

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
field_solver<Conf, ExecPolicy, coord_policy_polar>::field_solver(
    const grid_polar_t<Conf>& grid, const domain_comm<Conf, ExecPolicy>* comm)
    : field_solver_base<Conf>(grid), m_grid_polar(grid), m_comm(comm) {
  ExecPolicy<Conf>::set_grid(this->m_grid);
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_polar>::init() {
  field_solver_base<Conf>::init();

  sim_env().params().get_value("damping_length", m_damping_length);
  sim_env().params().get_value("damping_coef", m_damping_coef);
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_polar>::register_data_components() {
  field_solver_base<Conf>::register_data_components();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_polar>::update_explicit(
    double dt, double time) {}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_polar>::update_semi_implicit(
    double dt, double alpha, double beta, double time) {}

}  // namespace Aperture
