#include "field_solver.h"
#include "framework/config.h"

namespace Aperture {

template <typename Conf>
void
field_solver<Conf>::init() {
  this->m_env.params().get_value("use_implicit", m_use_implicit);
  this->m_env.params().get_value("implicit_beta", m_beta);
  m_alpha = 1.0 - m_beta;
  this->m_env.params().get_value("fld_output_interval", m_data_interval);

  this->init_impl_tmp_fields();
}

template <typename Conf>
void
field_solver<Conf>::init_impl_tmp_fields() {
  if (m_use_implicit) {
    m_tmp_b1 =
        std::make_unique<vector_field<Conf>>(this->m_grid, MemType::host_only);
    m_tmp_b2 =
        std::make_unique<vector_field<Conf>>(this->m_grid, MemType::host_only);
    m_bnew =
        std::make_unique<vector_field<Conf>>(this->m_grid, MemType::host_only);
  }
}

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
field_solver<Conf>::update(double dt, uint32_t step) {
  double time = this->m_env.get_time();
  if (m_use_implicit)
    this->update_semi_implicit(dt, m_alpha, m_beta, time);
  else
    this->update_explicit(dt, time);

  this->Etotal->copy_from(*(this->E0));
  this->Etotal->add_by(*(this->E));
  this->Btotal->copy_from(*(this->B0));
  this->Btotal->add_by(*(this->B));
}

template <typename Conf>
void
field_solver<Conf>::update_explicit(double dt, double time) {}

template <typename Conf>
void
field_solver<Conf>::update_semi_implicit(double dt, double alpha, double beta,
                                         double time) {}

//// Explicit instantiation for 1d, 2d, and 3d
INSTANTIATE_WITH_CONFIG(field_solver);

}  // namespace Aperture
