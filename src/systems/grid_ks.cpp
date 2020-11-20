#include "grid_ks.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "utils/gauss_quadrature.h"
#include "utils/logger.h"
#include "utils/timer.h"

namespace Aperture {

template <typename Conf>
grid_ks_t<Conf>::grid_ks_t(sim_environment& env, const domain_comm<Conf>* comm)  :
    grid_t<Conf>(env, comm) {
  env.params().get_value("bh_spin", a);

  Logger::print_info("In grid, a is {}", a);

  for (int i = 0; i < 3; i++) {
    m_Ab[i].resize(this->extent());
    m_Ad[i].resize(this->extent());
  }
  m_ag11dr_e.resize(this->extent());
  m_ag11dr_h.resize(this->extent());
  m_ag13dr_e.resize(this->extent());
  m_ag13dr_h.resize(this->extent());
  m_ag22dth_e.resize(this->extent());
  m_ag22dth_h.resize(this->extent());
  m_gbetadth_e.resize(this->extent());
  m_gbetadth_h.resize(this->extent());

  timer::stamp();
  compute_coef();
  timer::show_duration_since_stamp("Computing KS coefficients", "ms");
}

template <typename Conf>
void
grid_ks_t<Conf>::compute_coef() {
  auto ext = this->extent();

  for (auto idx : m_Ab[0].indices()) {
    using namespace Metric_KS;

    auto pos = get_pos(idx, ext);

    double r = radius(this->template pos<0>(pos[0], false));
    double r_s = radius(this->template pos<0>(pos[0], true));
    double th = theta(this->template pos<1>(pos[1], false));
    double th_s = theta(this->template pos<1>(pos[1], true));

    m_Ab[0][idx] = gauss_quad([this, r_s] (auto x) {
      return sqrt_gamma(a, r_s, x);
    }, th_s, theta(this->template pos<1>(pos[1] + 1, true)));

    m_Ab[1][idx] = gauss_quad([this, th_s] (auto x) {
      return sqrt_gamma(a, x, th_s);
    }, r_s, radius(this->template pos<0>(pos[0] + 1, true)));

    m_Ab[2][idx] = gauss_quad([this, r_s, pos] (auto x) {
      return gauss_quad([this, x] (auto y) {
        return sqrt_gamma(a, y, x);
      }, r_s, radius(this->template pos<0>(pos[0] + 1, true)));
    }, th_s, theta(this->template pos<1>(pos[1] + 1, true)));

    m_Ad[0][idx] = gauss_quad([this, r] (auto x) {
      return sqrt_gamma(a, r, x);
    }, theta(this->template pos<1>(pos[1] - 1, false)), th);

    m_Ad[1][idx] = gauss_quad([this, th] (auto x) {
      return sqrt_gamma(a, x, th);
    }, radius(this->template pos<0>(pos[0] - 1, false)), r);

    m_Ad[2][idx] = gauss_quad([this, r, pos] (auto x) {
      return gauss_quad([this, x] (auto y) {
        return sqrt_gamma(a, y, x);
      }, radius(this->template pos<0>(pos[0] - 1, false)), r);
    }, theta(this->template pos<1>(pos[1] - 1, false)), th);

    m_ag11dr_h[idx] = gauss_quad([this, th] (auto x) {
      return ag_11(a, x, th);
    }, radius(this->template pos<0>(pos[0] - 1, false)), r);

    m_ag11dr_e[idx] = gauss_quad([this, th_s] (auto x) {
      return ag_11(a, x, th_s);
    }, r_s, radius(this->template pos<0>(pos[0] + 1, true)));

    m_ag13dr_h[idx] = gauss_quad([this, th] (auto x) {
      return ag_13(a, x, th);
    }, radius(this->template pos<0>(pos[0] - 1, false)), r);

    m_ag13dr_e[idx] = gauss_quad([this, th_s] (auto x) {
      return ag_13(a, x, th_s);
    }, r_s, radius(this->template pos<0>(pos[0] + 1, true)));

    m_ag22dth_h[idx] = gauss_quad([this, r] (auto x) {
      return ag_22(a, r, x);
    }, theta(this->template pos<1>(pos[1] - 1, false)), th);

    m_ag22dth_e[idx] = gauss_quad([this, r_s] (auto x) {
      return ag_22(a, r_s, x);
    }, th_s, theta(this->template pos<1>(pos[1] + 1, true)));

    m_gbetadth_h[idx] = gauss_quad([this, r] (auto x) {
      return sq_gamma_beta(a, r, x);
    }, theta(this->template pos<1>(pos[1] - 1, false)), th);

    m_gbetadth_e[idx] = gauss_quad([this, r_s] (auto x) {
      return sq_gamma_beta(a, r_s, x);
    }, th_s, theta(this->template pos<1>(pos[1] + 1, true)));


  }

  for (int i = 0; i < 3; i++) {
    m_Ab[i].copy_to_device();
    m_Ad[i].copy_to_device();
  }
  m_ag11dr_e.copy_to_device();
  m_ag11dr_h.copy_to_device();
  m_ag13dr_e.copy_to_device();
  m_ag13dr_h.copy_to_device();
  m_ag22dth_e.copy_to_device();
  m_ag22dth_h.copy_to_device();
  m_gbetadth_e.copy_to_device();
  m_gbetadth_h.copy_to_device();
}

template class grid_ks_t<Config<2>>;

}
