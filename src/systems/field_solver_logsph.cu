#include "field_solver_logsph.h"
#include "framework/config.h"

namespace Aperture {

template <typename Conf>
void
field_solver_logsph<Conf>::init() {
  m_tmp_b1 = std::make_unique<vector_field<Conf>>(this->m_grid);
  m_tmp_b2 = std::make_unique<vector_field<Conf>>(this->m_grid);
}

template <typename Conf>
void
field_solver_logsph<Conf>::update(double dt, uint32_t step) {}

template <typename Conf>
void
field_solver_logsph<Conf>::update_semi_impl(double dt, double alpha, double beta, double time) {}

template <typename Conf>
void
field_solver_logsph<Conf>::update_b(double dt, double alpha, double beta) {}

template <typename Conf>
void
field_solver_logsph<Conf>::update_e(double dt, double alpha, double beta) {}


template class field_solver_logsph<Config<2>>;

}
