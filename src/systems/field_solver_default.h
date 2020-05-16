#ifndef __FIELD_SOLVER_DEFAULT_H_
#define __FIELD_SOLVER_DEFAULT_H_

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
class field_solver_default : public system_t {
 protected:
  const grid_t<Conf>& m_grid;
  const domain_comm<Conf>* m_comm;

  vector_field<Conf> *E, *B, *Etotal, *Btotal, *E0, *B0, *J;
  scalar_field<Conf> *divE, *divB, *EdotB;

 public:
  static std::string name() { return "field_solver"; }

  field_solver_default(sim_environment& env, const grid_t<Conf>& grid,
                       const domain_comm<Conf>* comm = nullptr)
      : system_t(env), m_grid(grid), m_comm(comm) {}

  void init() {}
  void update(double dt, uint32_t step);

  void register_data_components() {
    // output fields, we don't directly use here
    Etotal = m_env.register_data<vector_field<Conf>>("E", m_grid,
                                            field_type::edge_centered);
    Btotal = m_env.register_data<vector_field<Conf>>("B", m_grid,
                                            field_type::face_centered);

    E = m_env.register_data<vector_field<Conf>>("Edelta", m_grid,
                                                field_type::edge_centered);
    E0 = m_env.register_data<vector_field<Conf>>("E0", m_grid,
                                                 field_type::edge_centered);
    B = m_env.register_data<vector_field<Conf>>("Bdelta", m_grid,
                                                field_type::face_centered);
    B0 = m_env.register_data<vector_field<Conf>>("B0", m_grid,
                                                 field_type::face_centered);
    J = m_env.register_data<vector_field<Conf>>("J", m_grid,
                                                field_type::edge_centered);
    divB = m_env.register_data<scalar_field<Conf>>("divB", m_grid,
                                                   field_type::cell_centered);
    divE = m_env.register_data<scalar_field<Conf>>("divE", m_grid,
                                                   field_type::vert_centered);
    // EdotB = m_env.register_data<scalar_field<Conf>>("EdotB", m_grid,
    //                                                 field_type::vert_centered);
  }

  void update_e(double dt);
  void update_b(double dt);
  void compute_divs();
};

}  // namespace Aperture

#endif
