#ifndef __FIELD_SOLVER_LOGSPH_H_
#define __FIELD_SOLVER_LOGSPH_H_

#include "data/fields.hpp"
#include "framework/environment.hpp"
#include "framework/system.h"
#include "systems/grid_logsph.h"
#include "systems/field_solver_default.h"
#include <memory>

namespace Aperture {

// System that updates Maxwell equations using an explicit scheme in Log-Spherical
// coordinates
template <typename Conf>
class field_solver_logsph : public field_solver_default<Conf> {
 private:
  std::unique_ptr<vector_field<Conf>> m_tmp_b1, m_tmp_b2, m_bnew;
  bool m_use_implicit = true;
  double m_alpha = 0.45;
  double m_beta = 0.55;
  int m_damping_length = 64;
  double m_damping_coef = 0.003;
 
 public:
  static std::string name() { return "field_solver"; }

  typedef field_solver_default<Conf> base_class;

  field_solver_logsph(sim_environment& env, const grid_logsph_t<Conf>& grid,
                       const domain_comm<Conf>* comm)
      : base_class(env, grid, comm) {}

  void init();
  void update(double dt, uint32_t step);

  void update_explicit(double dt, double time);
  void update_semi_impl(double dt, double alpha, double beta, double time);
  void update_b(double dt, double alpha, double beta);
  void update_e(double dt, double alpha, double beta);
};

}

#endif
