#ifndef __BOUNDARY_CONDITION_H_
#define __BOUNDARY_CONDITION_H_

#include "data/fields.hpp"
#include "framework/environment.hpp"
#include "framework/system.h"
#include "systems/grid_logsph.h"
#include <memory>

namespace Aperture {

template <typename Conf>
class boundary_condition : public system_t {
 protected:
  const grid_logsph_t<Conf>& m_grid;
  double m_omega_0 = 0.0;
  double m_omega_t = 0.0;

  vector_field<Conf> *E, *B, *E0, *B0;

 public:
  static std::string name() { return "boundary_condition"; }

  boundary_condition(sim_environment& env, const grid_logsph_t<Conf>& grid) :
      system_t(env), m_grid(grid) {}

  void init() override;
  void update(double dt, uint32_t step) override;

  void register_dependencies() {
    m_env.get_data("Edelta", &E);
    m_env.get_data("E0", &E0);
    m_env.get_data("Bdelta", &B);
    m_env.get_data("B0", &B0);
  }
};

}

#endif // __BOUNDARY_CONDITION_H_
