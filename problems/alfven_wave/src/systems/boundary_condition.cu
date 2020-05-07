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
                        float dw0_)
      : rpert1(rp1),
        rpert2(rp2),
        tp_start(tp_s),
        tp_end(tp_e),
        nT(nT_),
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
    if (t >= tp_start && t <= tp_end && th >= th1 && th <= th2) {
      Scalar omega =
          dw0 * math::exp(-0.5 * square((th - mu) / s)) *
          math::sin((t - tp_start) * 2.0 * M_PI * nT / (tp_end - tp_start));
      return omega;
    } else {
      return 0.0;
    }
  }
};

template <typename Conf>
void
inject_particles(particle_data_t& ptc, curand_states_t& rand_states,
                 buffer<float>& surface_n, int num_per_cell,
                 typename Conf::value_t weight,
                 const grid_curv_t<Conf>& grid) {
  surface_n.assign_dev(0.0f);

  auto ptc_num = ptc.number();
  // First measure surface density
  kernel_launch(
      [ptc_num] __device__(auto ptc, auto surface_n) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        for (auto n : grid_stride_range(0, ptc_num)) {
          auto c = ptc.cell[n];
          if (c == empty_cell) continue;

          auto idx = typename Conf::idx_t(c, ext);
          auto pos = idx.get_pos();

          if (pos[0] == grid.guard[0]) {
            auto flag = ptc.flag[n];
            auto sp = get_ptc_type(flag);
            // surface_n[pos[1]] += ptc.weight[n];
            atomicAdd(&surface_n[pos[1]],
                      ptc.weight[n] * math::abs(dev_charges[sp]));
          }
        }
      },
      ptc.get_dev_ptrs(), surface_n.dev_ptr());
  CudaSafeCall(cudaDeviceSynchronize());

  kernel_launch(
      [ptc_num, weight] __device__(auto ptc, auto surface_n, auto num_inj,
                                   auto states) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        int inj_n0 = grid.guard[0];
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        cuda_rng_t rng(&states[id]);
        for (auto n1 :
             grid_stride_range(grid.guard[1], grid.dims[1] - grid.guard[1])) {
          size_t offset = ptc_num + n1 * num_inj * 2;
          auto pos = index_t<Conf::dim>(inj_n0, n1);
          auto idx = typename Conf::idx_t(pos, ext);
          Scalar theta = grid.template pos<1>(n1, false);
          if (surface_n[pos[1]] >
              square(0.5f / grid.delta[1]) * math::sin(theta))
            continue;
          for (int i = 0; i < num_inj; i++) {
            float x2 = rng();
            theta = grid.template pos<1>(n1, x2);
            ptc.x1[offset + i * 2] = ptc.x1[offset + i * 2 + 1] = 0.5f;
            ptc.x2[offset + i * 2] = ptc.x2[offset + i * 2 + 1] = x2;
            ptc.x3[offset + i * 2] = ptc.x3[offset + i * 2 + 1] = 0.0f;
            ptc.p1[offset + i * 2] = ptc.p1[offset + i * 2 + 1] = 0.0f;
            ptc.p2[offset + i * 2] = ptc.p2[offset + i * 2 + 1] = 0.0f;
            ptc.p3[offset + i * 2] = ptc.p3[offset + i * 2 + 1] = 0.0f;
            ptc.E[offset + i * 2] = ptc.E[offset + i * 2 + 1] = 1.0f;
            ptc.cell[offset + i * 2] = ptc.cell[offset + i * 2 + 1] =
                idx.linear;
            ptc.weight[offset + i * 2] = ptc.weight[offset + i * 2 + 1] =
                weight * math::sin(theta);
            ptc.flag[offset + i * 2] = set_ptc_type_flag(0, PtcType::electron);
            ptc.flag[offset + i * 2 + 1] =
                set_ptc_type_flag(0, PtcType::positron);
          }
        }
      },
      ptc.get_dev_ptrs(), surface_n.dev_ptr(), num_per_cell,
      rand_states.states());
  CudaSafeCall(cudaDeviceSynchronize());

  ptc.add_num(num_per_cell * 2 * grid.dims[1]);
}

template <typename Conf>
void
boundary_condition<Conf>::init() {
  m_env.get_data("Edelta", &E);
  m_env.get_data("E0", &E0);
  m_env.get_data("Bdelta", &B);
  m_env.get_data("B0", &B0);
  m_env.get_data("rand_states", &rand_states);
  m_env.get_data("particles", &ptc);

  m_env.params().get_value("rpert1", m_rpert1);
  m_env.params().get_value("rpert2", m_rpert2);
  m_env.params().get_value("tp_start", m_tp_start);
  m_env.params().get_value("tp_end", m_tp_end);
  m_env.params().get_value("nT", m_nT);
  m_env.params().get_value("dw0", m_dw0);
  Logger::print_info("{}, {}, {}, {}, {}, {}", m_rpert1, m_rpert2, m_tp_start,
                     m_tp_end, m_nT, m_dw0);

  m_surface_n.resize(m_grid.dims[1]);
}

template <typename Conf>
void
boundary_condition<Conf>::update(double dt, uint32_t step) {
  auto ext = m_grid.extent();
  typedef typename Conf::idx_t idx_t;
  typedef typename Conf::value_t value_t;

  value_t time = m_env.get_time();
  wpert_sph_t wpert(m_rpert1, m_rpert2, m_tp_start, m_tp_end, m_nT, m_dw0);

  // Apply twist on the stellar surface
  kernel_launch(
      [ext, time] __device__(auto e, auto b, auto e0, auto b0, auto wpert) {
        auto& grid = dev_grid<Conf::dim>();
        for (auto n1 : grid_stride_range(0, grid.dims[1])) {
          value_t theta =
              grid_sph_t<Conf>::theta(grid.template pos<1>(n1, false));
          value_t theta_s =
              grid_sph_t<Conf>::theta(grid.template pos<1>(n1, true));

          // For quantities that are not continuous across the surface
          for (int n0 = 0; n0 < grid.skirt[0]; n0++) {
            auto idx = idx_t(index_t<2>(n0, n1), ext);
            value_t r =
                grid_sph_t<Conf>::radius(grid.template pos<0>(n0, false));
            value_t omega = wpert(time, r, theta_s);
            // printf("omega is %f\n", omega);
            e[0][idx] = omega * sin(theta_s) * r * b0[1][idx];
            b[1][idx] = 0.0;
            b[2][idx] = 0.0;
          }
          // For quantities that are continuous across the surface
          for (int n0 = 0; n0 < grid.skirt[0] + 1; n0++) {
            auto idx = idx_t(index_t<2>(n0, n1), ext);
            value_t r_s =
                grid_sph_t<Conf>::radius(grid.template pos<0>(n0, true));
            value_t omega = wpert(time, r_s, theta);
            b[0][idx] = 0.0;
            e[1][idx] = -omega * sin(theta) * r_s * b0[0][idx];
            e[2][idx] = 0.0;
          }
        }
      },
      E->get_ptrs(), B->get_ptrs(), E0->get_ptrs(), B0->get_ptrs(), wpert);
  CudaSafeCall(cudaDeviceSynchronize());

  // Inject particles
  if (step % 2 == 0) {
    inject_particles<Conf>(*ptc, *rand_states, m_surface_n, 1, 1.0, m_grid);
  }
}

template class boundary_condition<Config<2>>;

}  // namespace Aperture
