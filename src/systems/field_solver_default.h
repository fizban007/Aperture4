#ifndef __FIELD_SOLVER_DEFAULT_H_
#define __FIELD_SOLVER_DEFAULT_H_

#include "data/fields.hpp"
#include "framework/environment.hpp"
#include "framework/system.h"
#include "systems/grid.h"
#include <memory>

namespace Aperture {

// System that updates Maxwell equations using an explicit scheme in
// Cartesian coordinates
template <typename Conf>
class field_solver_default : public system_t {
 private:
  const grid_t<Conf>& m_grid;

  std::shared_ptr<vector_field<Conf>> E, B, E0, B0, J;
  std::shared_ptr<scalar_field<Conf>> divE, divB;

 public:
  static std::string name() { return "field_solver"; }

  field_solver_default(sim_environment& env, const grid_t<Conf>& grid) :
      system_t(env), m_grid(grid) {}

  void init() {}

  void update(double dt, uint32_t step);

  void register_dependencies() {
    depends_on("grid");
    depends_on("communicator");
    E = m_env.register_data<vector_field<Conf>>("E", m_grid,
                                              field_type::edge_centered);
    E0 = m_env.register_data<vector_field<Conf>>("E0", m_grid,
                                               field_type::edge_centered);
    B = m_env.register_data<vector_field<Conf>>("B", m_grid,
                                              field_type::face_centered);
    B0 = m_env.register_data<vector_field<Conf>>("B0", m_grid,
                                               field_type::face_centered);
    J = m_env.register_data<vector_field<Conf>>("J", m_grid,
                                              field_type::edge_centered);
    divB = m_env.register_data<scalar_field<Conf>>("divB", m_grid,
                                                 field_type::cell_centered);
    divE = m_env.register_data<scalar_field<Conf>>("divE", m_grid,
                                                 field_type::vert_centered);
  }

  void update_e(double dt);
  void update_b(double dt);
  void compute_divs();
};

}  // namespace Aperture

#endif
