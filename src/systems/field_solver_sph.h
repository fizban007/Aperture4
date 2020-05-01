#ifndef __FIELD_SOLVER_SPH_H_
#define __FIELD_SOLVER_SPH_H_

#include "data/fields.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/grid_curv.h"
#include "systems/field_solver_default.h"
#include <memory>

namespace Aperture {

// System that updates Maxwell equations using an explicit scheme in Spherical
// coordinates
template <typename Conf>
class field_solver_sph : public field_solver_default<Conf> {
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

  field_solver_sph(sim_environment& env, const grid_curv_t<Conf>& grid,
                       const domain_comm<Conf>* comm = nullptr)
      : base_class(env, grid, comm) {}

  void init() override;
  void update(double dt, uint32_t step) override;

  void update_explicit(double dt, double time);
  void update_semi_impl(double dt, double alpha, double beta, double time);
};

}

#endif
