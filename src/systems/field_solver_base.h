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

#ifndef _FIELD_SOLVER_BASE_H_
#define _FIELD_SOLVER_BASE_H_

#include "data/fields.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/domain_comm.h"
// #include "systems/helpers/pml_data.hpp"
#include "systems/grid.h"
#include "utils/nonown_ptr.hpp"
#include <memory>

namespace Aperture {

// System that updates Maxwell equations
template <typename Conf, template <class> class ExecPolicy,
          template <class> class CoordPolicy>
class field_solver : public system_t {
 public:
  static std::string name() { return "field_solver"; }

  field_solver(const grid_t<Conf>& grid,
               const domain_comm<Conf, ExecPolicy>* comm = nullptr)
      : m_grid(grid), m_comm(comm) {}

  virtual ~field_solver() {}

  virtual void init() override;
  virtual void update(double dt, uint32_t step) override;
  virtual void register_data_components() override;

  virtual void update_explicit(double dt, double time);
  virtual void update_semi_implicit(double dt, double alpha, double beta,
                                    double time);

 protected:
  const grid_t<Conf>& m_grid;
  const domain_comm<Conf, ExecPolicy>* m_comm;

  // vector_field<Conf> *E, *B, *Etotal, *Btotal, *E0, *B0, *J;
  // scalar_field<Conf> *divE, *divB, *EdotB;
  nonown_ptr<vector_field<Conf>> E, B, Etotal, Btotal, E0, B0, J;
  nonown_ptr<scalar_field<Conf>> divE, divB, EdotB, flux, E_sqr, B_sqr;

  bool m_use_implicit = false;
  double m_alpha = 0.45;
  double m_beta = 0.55;
  int m_data_interval = 100;
  bool m_update_e = true;
  bool m_update_b = true;
  bool m_damping[Conf::dim * 2] = {};
  int m_pml_length = 16;

  // These are temporary fields used in the semi-implicit update
  std::unique_ptr<vector_field<Conf>> m_bnew;
  std::unique_ptr<vector_field<Conf>> m_enew;
  // These are temporary fields required for pml
  nonown_ptr<vector_field<Conf>> m_tmp_b1, m_tmp_b2, m_tmp_e1, m_tmp_e2;

  // PML data structures
  // std::unique_ptr<pml_data<Conf>> m_pml[Conf::dim * 2];

  void init_tmp_fields();

  void register_data_impl(MemType type);
  void compute_e_update_pml(vector_field<Conf>& E, const vector_field<Conf>& B,
                            const vector_field<Conf>& J, double dt);
  void compute_b_update_pml(vector_field<Conf>& B, const vector_field<Conf>& E,
                            double dt);
  void compute_divs_e_b();
};

}  // namespace Aperture

#endif  // _FIELD_SOLVER_BASE_H_
