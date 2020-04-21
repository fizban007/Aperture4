#include "grid_logsph.h"
#include "framework/config.h"
#include "framework/environment.hpp"
#include "systems/domain_comm.hpp"

namespace Aperture {

double alpha(double r, double rs) { return std::sqrt(1.0 - rs / r); }
double l1(double r, double rs) {
  double a = alpha(r, rs);
  return r * a + 0.5 * rs * std::log(2.0 * r * (1.0 + a) - rs);
}
double A2(double r, double rs) {
  double a = alpha(r, rs);
  return 0.25 * r * a * (2.0 * r + 3.0 * rs) +
      0.375 * rs * rs * std::log(2.0 * r * (1.0 + a) - rs);
}
double V3(double r, double rs) {
  double a = alpha(r, rs);
  return r * a * (8.0 * r * r + 10.0 * r * rs + 15.0 * rs * rs) / 24.0 +
      0.3125 * rs * rs * rs * std::log(2.0 * r * (1.0 + a) - rs);
}



template <typename Conf>
grid_logsph_t<Conf>::grid_logsph_t(sim_environment& env,
                                   const domain_info_t<Conf::dim>& domain_info)
    : grid_t<Conf>(env, domain_info) {}

template <typename Conf>
grid_logsph_t<Conf>::grid_logsph_t(sim_environment& env,
                                   const domain_comm<Conf>& comm) :
    grid_logsph_t<Conf>(env, comm.domain_info()) {}

template <typename Conf>
grid_logsph_t<Conf>::~grid_logsph_t() {}

template <typename Conf>
void
grid_logsph_t<Conf>::compute_coef() {
  double r_g = 0.0;
  this->m_env.params().get_value("compactness", r_g);

  for (int i = 0; i < 3; i++) {
    m_le[i].resize(extent(this->dims[0], this->dims[1]));
    m_lb[i].resize(extent(this->dims[0], this->dims[1]));
    m_Ae[i].resize(extent(this->dims[0], this->dims[1]));
    m_Ab[i].resize(extent(this->dims[0], this->dims[1]));
  }
  m_dV.resize(extent(this->dims[0], this->dims[1]));

  for (int j = 0; j < this->dims[1]; j++) {
    double x2 = this->pos(1, j, false);
    double x2s = this->pos(1, j, true);
    for (int i = 0; i < this->dims[0]; i++) {
      double x1 = this->pos(0, i, false);
      double x1s = this->pos(0, i, true);
      double r_plus = std::exp(x1 + this->delta[0]);
      double r = std::exp(x1);
      double rs = std::exp(x1s);
      double rs_minus = std::exp(x1s - this->delta[0]);
      auto idx = typename Conf::idx_t({i, j}, this->extent());
      m_le[0][idx] = l1(rs, r_g) - l1(rs_minus, r_g);
      m_le[1][idx] = rs * this->delta[1];
      m_le[2][idx] = rs * std::sin(x2s);
      m_lb[0][idx] = l1(r_plus, r_g) - l1(r, r_g);
      m_lb[1][idx] = r * this->delta[1];
      m_lb[2][idx] = r * std::sin(x2);

      m_Ae[0][idx] =
          r * r * (std::cos(x2) - std::cos(x2 + this->delta[1]));
      if (std::abs(x2s) < TINY) {
        m_Ae[0][idx] =
            r * r * 2.0 * (1.0 - std::cos(0.5 * this->delta[1]));
      } else if (std::abs(x2s - M_PI) < TINY) {
        m_Ae[0][idx] =
            r * r * 2.0 * (1.0 - std::cos(0.5 * this->delta[1]));
      }
      m_Ae[1][idx] = (A2(r_plus, r_g) - A2(r, r_g)) * std::sin(x2);
      // Avoid axis singularity
      // if (std::abs(x2s) < TINY || std::abs(x2s - CONST_PI)
      // < TINY)
      //   m_A2_e(i, j) = 0.5 * std::sin(TINY) *
      //                  (std::exp(2.0 * x1s) -
      //                   std::exp(2.0 * (x1s - this->delta[0])));

      m_Ae[2][idx] = (A2(r_plus, r_g) - A2(r, r_g)) * this->delta[1];

      m_Ab[0][idx] =
          rs * rs * (std::cos(x2s - this->delta[1]) - std::cos(x2s));
      m_Ab[1][idx] =
          (A2(rs, r_g) - A2(rs_minus, r_g)) * std::sin(x2s);
      m_Ab[2][idx] =
          (A2(rs, r_g) - A2(rs_minus, r_g)) * this->delta[1];

      m_dV[idx] = (V3(r_plus, r_g) - V3(r, r_g)) *
          (std::cos(x2) - std::cos(x2 + this->delta[1])) /
          (this->delta[0] * this->delta[1]);

      if (std::abs(x2s) < TINY ||
          std::abs(x2s - M_PI) < TINY) {
        m_dV(i, j) = (V3(r_plus, r_g) - V3(r, r_g)) * 2.0 *
            (1.0 - std::cos(0.5 * this->delta[1])) /
            (this->delta[0] * this->delta[1]);
        // if (i == 100)
        //   Logger::print_info("dV is {}", m_dV(i, j));
      }
    }
  }

  for (int i = 0; i < 3; i++) {
    m_le[i].copy_to_device();
    m_lb[i].copy_to_device();
    m_Ae[i].copy_to_device();
    m_Ab[i].copy_to_device();
  }
  m_dV.copy_to_device();
}

template <typename Conf>
auto
grid_logsph_t<Conf>::get_grid_ptrs() const -> grid_ptrs_t {
  grid_ptrs_t result;

  for (int i = 0; i < 3; i++) {
    result.le[i] = m_le[i].get_const_ptr();
    result.lb[i] = m_lb[i].get_const_ptr();
    result.Ae[i] = m_Ae[i].get_const_ptr();
    result.Ab[i] = m_Ab[i].get_const_ptr();
  }
  result.dV = m_dV.get_const_ptr();

  return result;
}

template class grid_logsph_t<Config<2>>;

}  // namespace Aperture
