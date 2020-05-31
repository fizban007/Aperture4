#include "field_solver.h"
#include "framework/config.h"

namespace Aperture {

template <typename Conf>
void
field_solver<Conf>::register_data_components() {
  register_data_impl(MemType::host_only);
}

template <typename Conf>
void
field_solver<Conf>::register_data_impl(MemType type) {
  // output fields, we don't directly use here
  Etotal = m_env.register_data<vector_field<Conf>>(
      "E", m_grid, field_type::edge_centered, type);
  Btotal = m_env.register_data<vector_field<Conf>>(
      "B", m_grid, field_type::face_centered, type);

  E = m_env.register_data<vector_field<Conf>>("Edelta", m_grid,
                                              field_type::edge_centered, type);
  E->skip_output(true);
  E->include_in_snapshot(true);
  E0 = m_env.register_data<vector_field<Conf>>("E0", m_grid,
                                               field_type::edge_centered, type);
  E0->skip_output(true);
  E0->include_in_snapshot(true);
  B = m_env.register_data<vector_field<Conf>>("Bdelta", m_grid,
                                              field_type::face_centered, type);
  B->skip_output(true);
  B->include_in_snapshot(true);
  B0 = m_env.register_data<vector_field<Conf>>("B0", m_grid,
                                               field_type::face_centered, type);
  B0->skip_output(true);
  B0->include_in_snapshot(true);
  J = m_env.register_data<vector_field<Conf>>("J", m_grid,
                                              field_type::edge_centered, type);
  divB = m_env.register_data<scalar_field<Conf>>(
      "divB", m_grid, field_type::cell_centered, type);
  divE = m_env.register_data<scalar_field<Conf>>(
      "divE", m_grid, field_type::vert_centered, type);
  // EdotB = m_env.register_data<scalar_field<Conf>>("EdotB", m_grid,
  //                                                 field_type::vert_centered);
}

template <typename Conf>
void
field_solver<Conf>::update(double dt, uint32_t step) {}

//// Explicit instantiation for 1d, 2d, and 3d
INSTANTIATE_CONFIG(field_solver);

}  // namespace Aperture
