#ifndef __FIELD_SOLVER_DEFAULT_H_
#define __FIELD_SOLVER_DEFAULT_H_

#include "data/fields.hpp"
#include "framework/environment.hpp"
#include "framework/system.h"
#include "systems/grid.hpp"
#include <memory>

namespace Aperture {

// System that updates Maxwell equations using an explicit scheme in Cartesian
// coordinates
template <typename Conf>
class field_solver_default : public system_t {
 private:
  const Conf& m_conf;
  const Grid<Conf::dim>* m_grid;

  std::shared_ptr<vector_field<Conf>> E, B, E0, B0, J;
  std::shared_ptr<scalar_field<Conf>> divE, divB;

 public:
  static std::string name() { return "field_solver_default"; }

  field_solver_default(const Conf& conf)
      : m_conf(conf) {}

  void init() {
    m_grid = m_env->shared_data().get<const Grid<Conf::dim>>("grid");
    if (m_grid == nullptr)
      throw std::runtime_error("No grid system defined!");

    auto ext = m_grid->extent();

    m_env->get_data("E", E);
    m_env->get_data("E0", E0);
    m_env->get_data("B", B);
    m_env->get_data("B0", B0);
    m_env->get_data("J", J);
    m_env->get_data("divE", divE);
    m_env->get_data("divB", divB);
  }

  void update(double dt, uint32_t step);

  void register_dependencies(sim_environment& env) {
    depends_on("grid");
    depends_on("communicator");
    env.register_data<vector_field>(m_conf, "E", field_type::edge_centered);
    env.register_data<vector_field>(m_conf, "E0", field_type::edge_centered);
    env.register_data<vector_field>(m_conf, "B", field_type::face_centered);
    env.register_data<vector_field>(m_conf, "B0", field_type::face_centered);
    env.register_data<vector_field>(m_conf, "J", field_type::edge_centered);
    env.register_data<scalar_field>(m_conf, "divB", field_type::cell_centered);
    env.register_data<scalar_field>(m_conf, "divE", field_type::vert_centered);
  }

  void update_e(double dt);
  void update_b(double dt);
  void compute_divs();
};

}  // namespace Aperture

#endif
