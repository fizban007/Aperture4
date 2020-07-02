#ifndef _FIELD_SOLVER_H_
#define _FIELD_SOLVER_H_

#include "data/fields.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/domain_comm.h"
#include "systems/grid.h"
#include <memory>

namespace Aperture {

// System that updates Maxwell equations using an explicit scheme in
// Cartesian coordinates
template <typename Conf>
class field_solver : public system_t {
 public:
  static std::string name() { return "field_solver"; }

  field_solver(sim_environment& env, const grid_t<Conf>& grid,
               const domain_comm<Conf>* comm = nullptr)
      : system_t(env), m_grid(grid), m_comm(comm) {}

  virtual ~field_solver() {}

  virtual void init() override;
  virtual void update(double dt, uint32_t step) override;
  virtual void register_data_components() override;

  virtual void update_explicit(double dt, double time);
  virtual void update_semi_implicit(double dt, double alpha, double beta, double time);

 protected:
  const grid_t<Conf>& m_grid;
  const domain_comm<Conf>* m_comm;

  vector_field<Conf> *E, *B, *Etotal, *Btotal, *E0, *B0, *J;
  scalar_field<Conf> *divE, *divB, *EdotB;

  bool m_use_implicit = true;
  double m_alpha = 0.45;
  double m_beta = 0.55;
  int m_data_interval = 100;

  // These are temporary fields used in the semi-implicit update
  std::unique_ptr<vector_field<Conf>> m_tmp_b1, m_tmp_b2, m_bnew;

  virtual void init_impl_tmp_fields();

  void register_data_impl(MemType type);
};

template <typename Conf>
class field_solver_cu : public field_solver<Conf> {
 public:
  static std::string name() { return "field_solver"; }

  using field_solver<Conf>::field_solver;

  virtual ~field_solver_cu() {}

  // virtual void update(double dt, uint32_t step) override;
  virtual void register_data_components() override;

  virtual void update_explicit(double dt, double time) override;
  virtual void update_semi_implicit(double dt, double alpha, double theta, double time) override;

 protected:
  virtual void init_impl_tmp_fields() override;
};

}

#endif  // _FIELD_SOLVER_H_
