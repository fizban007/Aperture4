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
  const Conf& m_conf;
  const Grid<Conf::dim>* m_grid;

  std::shared_ptr<vector_field<Conf>> E, B, E0, B0, J;
  std::shared_ptr<scalar_field<Conf>> divE, divB;

 public:
  static std::string name() { return "field_solver_default"; }

  field_solver_default(const Conf& conf) : m_conf(conf) {}

  void init() {
    m_grid = m_env->shared_data().get<const Grid<Conf::dim>>("grid");
    if (m_grid == nullptr) throw std::runtime_error("No grid system defined!");
  }

  void update(double dt, uint32_t step);

  void register_dependencies(sim_environment& env) {
    depends_on("grid");
    depends_on("communicator");
    E = env.register_data<vector_field<Conf>>("E", m_conf,
                                              field_type::edge_centered);
    E0 = env.register_data<vector_field<Conf>>("E0", m_conf,
                                               field_type::edge_centered);
    B = env.register_data<vector_field<Conf>>("B", m_conf,
                                              field_type::face_centered);
    B0 = env.register_data<vector_field<Conf>>("B0", m_conf,
                                               field_type::face_centered);
    J = env.register_data<vector_field<Conf>>("J", m_conf,
                                              field_type::edge_centered);
    divB = env.register_data<scalar_field<Conf>>("divB", m_conf,
                                                 field_type::cell_centered);
    divE = env.register_data<scalar_field<Conf>>("divE", m_conf,
                                                 field_type::vert_centered);
  }

  void update_e(double dt);
  void update_b(double dt);
  void compute_divs();
};

}  // namespace Aperture

#endif
