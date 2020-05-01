#include "grid_sph.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/domain_comm.hpp"

namespace Aperture {

double l1(double r, double rs) {
  return r;
}

double A2(double r, double rs) {
  return 0.5 * r * r;
}

double V3(double r, double rs) {
  return r * r * r / 3.0;
}

template <typename Conf>
grid_sph_t<Conf>::grid_sph_t(sim_environment& env,
                                   const domain_info_t<Conf::dim>& domain_info)
    : grid_curv_t<Conf>(env, domain_info) {}

template <typename Conf>
grid_sph_t<Conf>::grid_sph_t(sim_environment& env,
                                   const domain_comm<Conf>& comm) :
    grid_sph_t<Conf>(env, comm.domain_info()) {}

template <typename Conf>
grid_sph_t<Conf>::~grid_sph_t() {}

template <typename Conf>
void
grid_sph_t<Conf>::compute_coef() {
  double r_g = 0.0;
  this->m_env.params().get_value("compactness", r_g);

  for (int j = 0; j < this->dims[1]; j++) {
    double x2 = this->pos(1, j, false);
    double x2s = this->pos(1, j, true);
    for (int i = 0; i < this->dims[0]; i++) {
      double x1 = this->pos(0, i, false);
      double x1s = this->pos(0, i, true);
      double r_minus = x1 - this->delta[0];
      double r = x1;
      double rs = x1s;
      double rs_plus = x1s + this->delta[0];
      auto idx = typename Conf::idx_t({i, j}, this->extent());
      auto pos = idx.get_pos();
      this->m_le[0][idx] = l1(rs_plus, r_g) - l1(rs, r_g);
      this->m_le[1][idx] = rs * this->delta[1];
      this->m_le[2][idx] = rs * std::sin(x2s);
      this->m_lb[0][idx] = l1(r, r_g) - l1(r_minus, r_g);
      this->m_lb[1][idx] = r * this->delta[1];
      this->m_lb[2][idx] = r * std::sin(x2);

      this->m_Ae[0][idx] =
          r * r * (std::cos(x2 - this->delta[1]) - std::cos(x2));
      if (std::abs(x2s) < 0.1 * this->delta[1]) {
        this->m_Ae[0][idx] =
            r * r * 2.0 * (1.0 - std::cos(0.5 * this->delta[1]));
      } else if (std::abs(x2s - M_PI) < 0.1 * this->delta[1]) {
        this->m_Ae[0][idx] =
            r * r * 2.0 * (1.0 - std::cos(0.5 * this->delta[1]));
      }
      this->m_Ae[1][idx] = (A2(r, r_g) - A2(r_minus, r_g)) * std::sin(x2);
      // Avoid axis singularity
      // if (std::abs(x2s) < TINY || std::abs(x2s - CONST_PI)
      // < TINY)
      //   m_A2_e(i, j) = 0.5 * std::sin(TINY) *
      //                  (std::exp(2.0 * x1s) -
      //                   std::exp(2.0 * (x1s - this->delta[0])));

      this->m_Ae[2][idx] = (A2(r, r_g) - A2(r_minus, r_g)) * this->delta[1];

      this->m_Ab[0][idx] =
          rs * rs * (std::cos(x2s) - std::cos(x2s + this->delta[1]));
      if (std::abs(x2s) > 0.1 * this->delta[1] &&
          std::abs(x2s - M_PI) > 0.1 * this->delta[1])
        this->m_Ab[1][idx] =
            (A2(rs_plus, r_g) - A2(rs, r_g)) * std::sin(x2s);
      else
        this->m_Ab[1][idx] = TINY;
      this->m_Ab[2][idx] =
          (A2(rs_plus, r_g) - A2(rs, r_g)) * this->delta[1];

      this->m_dV[idx] = (V3(r, r_g) - V3(r_minus, r_g)) *
          (std::cos(x2 - this->delta[1]) - std::cos(x2)) /
          (this->delta[0] * this->delta[1]);

      if (std::abs(x2s) < 0.1 * this->delta[1] ||
          std::abs(x2s - M_PI) < 0.1 * this->delta[1]) {
        this->m_dV[idx] = (V3(r, r_g) - V3(r_minus, r_g)) * 2.0 *
            (1.0 - std::cos(0.5 * this->delta[1])) /
            (this->delta[0] * this->delta[1]);
        // if (i == 100)
        //   Logger::print_info("dV is {}", m_dV(i, j));
      }
    }
  }

  for (int i = 0; i < 3; i++) {
    this->m_le[i].copy_to_device();
    this->m_lb[i].copy_to_device();
    this->m_Ae[i].copy_to_device();
    this->m_Ab[i].copy_to_device();
  }
  this->m_dV.copy_to_device();
}

template class grid_sph_t<Config<2>>;



}
