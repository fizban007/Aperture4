/*
 * Copyright (c) 2022 Alex Chen.
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

#include "data/fields.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/domain_comm.h"
// #include "systems/helpers/pml_data.hpp"
#include "systems/grid.h"
#include "utils/nonown_ptr.hpp"
#include <memory>

namespace Aperture {

// Base class for system that updates Maxwell equations
template <typename Conf>
class field_solver_base : public system_t {
 public:
  static std::string name() { return "field_solver"; }

  field_solver_base(const grid_t<Conf>& grid) : m_grid(grid) {}
  virtual ~field_solver_base() {}

  virtual void init() override;
  virtual void update(double dt, uint32_t step) override;
  // virtual void register_data_components() override;
  void register_data_components_impl(MemType type);

  virtual void update_explicit(double dt, double time);
  virtual void update_semi_implicit(double dt, double alpha, double beta,
                                    double time);
  virtual void compute_divs_e_b();
  virtual void compute_flux();
  virtual void compute_EB_sqr();

 protected:
  const grid_t<Conf>& m_grid;

  nonown_ptr<vector_field<Conf>> E, B, Etotal, Btotal, E0, B0, J;
  nonown_ptr<scalar_field<Conf>> divE, divB, EdotB, flux, E_sqr, B_sqr;

  bool m_use_implicit = false;
  double m_alpha = 0.45;
  double m_beta = 0.55;
  int m_data_interval = 100;
  bool m_update_e = true;
  bool m_update_b = true;
  bool m_compute_divs = true;
  bool m_compute_energies = true;
};

}  // namespace Aperture
