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

  virtual void update(double dt, uint32_t step) override;
  virtual void register_data_components() override;

 protected:
  const grid_t<Conf>& m_grid;
  const domain_comm<Conf>* m_comm;

  vector_field<Conf> *E, *B, *Etotal, *Btotal, *E0, *B0, *J;
  scalar_field<Conf> *divE, *divB, *EdotB;

  void register_data_impl(MemType type);
};

template <typename Conf>
class field_solver_cu : public field_solver<Conf> {
 public:
  static std::string name() { return "field_solver"; }

  field_solver_cu(sim_environment& env, const grid_t<Conf>& grid,
                  const domain_comm<Conf>* comm = nullptr)
      : field_solver<Conf>(env, grid, comm) {}

  virtual ~field_solver_cu() {}

  virtual void update(double dt, uint32_t step) override;
  virtual void register_data_components() override;
};

}

#endif  // _FIELD_SOLVER_H_
