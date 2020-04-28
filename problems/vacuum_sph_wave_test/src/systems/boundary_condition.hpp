#ifndef __BOUNDARY_CONDITION_H_
#define __BOUNDARY_CONDITION_H_

#include "data/fields.hpp"
#include "framework/environment.hpp"
#include "framework/system.h"
#include "systems/grid_curv.h"
#include <memory>

namespace Aperture {

template <typename Conf>
class boundary_condition : public system_t {
 protected:
  const grid_curv_t<Conf>& m_grid;
  double m_omega_0 = 0.0;
  double m_omega_t = 0.0;

  vector_field<Conf> *E, *B, *E0, *B0;

 public:
  static std::string name() { return "boundary_condition"; }

  boundary_condition(sim_environment& env, const grid_curv_t<Conf>& grid) :
      system_t(env), m_grid(grid) {}

  void init() override;
  void update(double dt, uint32_t step) override;

  void register_dependencies() {
    E = m_env.register_data<vector_field<Conf>>("Edelta", m_grid,
                                                field_type::edge_centered);
    E0 = m_env.register_data<vector_field<Conf>>("E0", m_grid,
                                                 field_type::edge_centered);
    B = m_env.register_data<vector_field<Conf>>("Bdelta", m_grid,
                                                field_type::face_centered);
    B0 = m_env.register_data<vector_field<Conf>>("B0", m_grid,
                                                 field_type::face_centered);
  }
};

}

#endif // __BOUNDARY_CONDITION_H_
