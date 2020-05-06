#include "boundary_condition.hpp"
#include "core/math.hpp"
#include "framework/config.h"
#include "systems/grid_sph.h"
#include "utils/kernel_helper.hpp"
#include "utils/util_functions.h"

namespace Aperture {

struct wpert_sph_t {
  float rpert1, rpert2;
  float tp_start, tp_end, nT, dw0;

  HD_INLINE wpert_sph_t(float rp1, float rp2, float tp_s, float tp_e, float nT_,
              float dw0_) :
      rpert1(rp1), rpert2(rp2), tp_start(tp_s), tp_end(tp_e), nT(nT_),
      dw0(dw0_) {}

  HD_INLINE Scalar operator()(Scalar t, Scalar r, Scalar th) {
    Scalar th1 = math::acos(math::sqrt(1.0f - 1.0f / rpert1));
    Scalar th2 = math::acos(math::sqrt(1.0f - 1.0f / rpert2));
    if (th1 > th2) {
      Scalar tmp = th1;
      th1 = th2;
      th2 = tmp;
    }
    Scalar mu = (th1 + th2) / 2.0;
    Scalar s = (mu - th1) / 3.0;
    if (t >= tp_start && t <= tp_end && th >= th1 &&
        th <= th2) {
      Scalar omega = dw0 * math::exp(-0.5 * square((th - mu) / s)) *
          math::sin((t - tp_start) * 2.0 * M_PI * nT / (tp_end - tp_start));
      return omega;
    } else {
      return 0.0;
    }
  }
};

template <typename Conf>
void
boundary_condition<Conf>::init() {
  m_env.get_data("Edelta", &E);
  m_env.get_data("E0", &E0);
  m_env.get_data("Bdelta", &B);
  m_env.get_data("B0", &B0);

  m_env.params().get_value("rpert1", m_rpert1);
  m_env.params().get_value("rpert2", m_rpert2);
  m_env.params().get_value("tp_start", m_tp_start);
  m_env.params().get_value("tp_end", m_tp_end);
  m_env.params().get_value("nT", m_nT);
  m_env.params().get_value("dw0", m_dw0);
  Logger::print_info("{}, {}, {}, {}, {}, {}", m_rpert1, m_rpert2,
                     m_tp_start, m_tp_end, m_nT, m_dw0);
}

template <typename Conf>
void
boundary_condition<Conf>::update(double dt, uint32_t step) {
  auto ext = m_grid.extent();
  typedef typename Conf::idx_t idx_t;
  typedef typename Conf::value_t value_t;

  value_t time = m_env.get_time();
  wpert_sph_t wpert(m_rpert1, m_rpert2, m_tp_start, m_tp_end, m_nT, m_dw0);

  kernel_launch([ext, time] __device__ (auto e, auto b, auto e0, auto b0, auto wpert) {
      auto& grid = dev_grid<Conf::dim>();
      for (auto n1 : grid_stride_range(0, grid.dims[1])) {
        value_t theta = grid_sph_t<Conf>::theta(grid.template pos<1>(n1, false));
        value_t theta_s = grid_sph_t<Conf>::theta(grid.template pos<1>(n1, true));

        // For quantities that are not continuous across the surface
        for (int n0 = 0; n0 < grid.skirt[0]; n0++) {
          auto idx = idx_t(index_t<2>(n0, n1), ext);
          value_t r = grid_sph_t<Conf>::radius(grid.template pos<0>(n0, false));
          value_t omega = wpert(time, r, theta_s);
          // printf("omega is %f\n", omega);
          e[0][idx] = omega * sin(theta_s) * r * b0[1][idx];
          b[1][idx] = 0.0;
          b[2][idx] = 0.0;
        }
        // For quantities that are continuous across the surface
        for (int n0 = 0; n0 < grid.skirt[0] + 1; n0++) {
          auto idx = idx_t(index_t<2>(n0, n1), ext);
          value_t r_s = grid_sph_t<Conf>::radius(grid.template pos<0>(n0, true));
          value_t omega = wpert(time, r_s, theta);
          b[0][idx] = 0.0;
          e[1][idx] = -omega * sin(theta) * r_s * b0[0][idx];
          e[2][idx] = 0.0;
        }
      }
    }, E->get_ptrs(), B->get_ptrs(), E0->get_ptrs(), B0->get_ptrs(), wpert);
  CudaSafeCall(cudaDeviceSynchronize());
}


template class boundary_condition<Config<2>>;

}
