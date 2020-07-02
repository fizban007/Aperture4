#ifndef __FIELD_SOLVER_SPH_H_
#define __FIELD_SOLVER_SPH_H_

#include "data/fields.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/grid_curv.h"
#include "systems/field_solver.h"
#include <memory>

namespace Aperture {

// System that updates Maxwell equations using an explicit scheme in Spherical
// coordinates
template <typename Conf>
class field_solver_sph_cu : public field_solver_cu<Conf> {
 private:
  int m_damping_length = 64;
  double m_damping_coef = 0.003;

  scalar_field<Conf>* flux;
 
 public:
  static std::string name() { return "field_solver"; }

  using field_solver_cu<Conf>::field_solver_cu;

  void init() override;
  void update(double dt, uint32_t step) override;
  void register_data_components() override;

  void update_explicit(double dt, double time) override;
  void update_semi_implicit(double dt, double alpha, double beta, double time) override;
};

}

#endif
