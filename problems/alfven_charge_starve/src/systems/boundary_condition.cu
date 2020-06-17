#include "boundary_condition.h"
#include "core/math.hpp"
#include "framework/config.h"
#include "systems/grid.h"
#include "utils/kernel_helper.hpp"
#include "utils/util_functions.h"

namespace Aperture {

struct wpert_cart_t {
  float tp_start, tp_end, nT, dw0, y_start, y_end, y_m;

  HD_INLINE wpert_cart_t(float tp_s, float tp_e, float nT_, float dw0_)
      : tp_start(tp_s), tp_end(tp_e), nT(nT_), dw0(dw0_) {
    y_start = 0.5f;
    y_m = 1.0f;
    y_end = 1.5f;
  }

  HD_INLINE Scalar operator()(Scalar t, Scalar x, Scalar y) {
    if (t >= tp_start && t <= tp_end && y > y_start && y < y_end) {
      Scalar omega =
          dw0 *
          math::sin((t - tp_start) * 2.0f * M_PI * nT / (tp_end - tp_start)) *
          math::sin(M_PI * (y - y_start) / (y_end - y_start));
      return omega;
    } else {
      return 0.0;
    }
  }
};

HOST_DEVICE Scalar
pml_sigma(Scalar x, Scalar xh, Scalar pmlscale, Scalar sig0) {
  if (x > xh)
    return sig0 * cube((x - xh) / pmlscale);
  else
    return 0.0;
}

template <typename Conf>
void
inject_particles(particle_data_t& ptc, curand_states_t& rand_states,
                 buffer<float>& surface_ne, buffer<float>& surface_np,
                 int num_per_cell, typename Conf::value_t weight,
                 const grid_t<Conf>& grid) {
  surface_ne.assign_dev(0.0f);
  surface_np.assign_dev(0.0f);

  auto ptc_num = ptc.number();
  // First measure surface density
  kernel_launch(
      [ptc_num] __device__(auto ptc, auto surface_ne, auto surface_np) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        for (auto n : grid_stride_range(0, ptc_num)) {
          auto c = ptc.cell[n];
          if (c == empty_cell) continue;

          auto idx = typename Conf::idx_t(c, ext);
          auto pos = idx.get_pos();
          if (pos[0] == grid.skirt[0]) {
            auto flag = ptc.flag[n];
            auto sp = get_ptc_type(flag);

            if (sp == 0)
              atomicAdd(&surface_ne[pos[1]],
                        ptc.weight[n] * math::abs(dev_charges[sp]));
            else if (sp == 1)
              atomicAdd(&surface_np[pos[1]],
                        ptc.weight[n] * math::abs(dev_charges[sp]));
          }
        }
      },
      ptc.get_dev_ptrs(), surface_ne.dev_ptr(), surface_np.dev_ptr());
  CudaSafeCall(cudaDeviceSynchronize());

  // Then inject particles
  kernel_launch(
      [ptc_num, weight] __device__(auto ptc, auto surface_ne, auto surface_np,
                                   auto num_inj, auto states) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        int inj_n0 = grid.skirt[0];
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        cuda_rng_t rng(&states[id]);
        for (auto n1 :
             grid_stride_range(grid.skirt[1], grid.dims[1] - grid.skirt[1])) {
          size_t offset = ptc_num + n1 * num_inj * 2;
          auto pos = index_t<Conf::dim>(inj_n0, n1);
          auto idx = typename Conf::idx_t(pos, ext);
          if (std::min(surface_ne[pos[1]], surface_np[pos[1]]) >
              square(2.0f / grid.delta[0]))
            continue;
          for (int i = 0; i < num_inj; i++) {
            float x2 = rng();
            ptc.x1[offset + i * 2] = ptc.x1[offset + i * 2 + 1] = 1.0f;
            ptc.x2[offset + i * 2] = ptc.x2[offset + i * 2 + 1] = x2;
            ptc.x3[offset + i * 2] = ptc.x3[offset + i * 2 + 1] = 0.0f;
            ptc.p1[offset + i * 2] = ptc.p1[offset + i * 2 + 1] = 0.0f;
            ptc.p2[offset + i * 2] = ptc.p2[offset + i * 2 + 1] = 0.0f;
            ptc.p3[offset + i * 2] = ptc.p3[offset + i * 2 + 1] = 0.0f;
            ptc.E[offset + i * 2] = ptc.E[offset + i * 2 + 1] = 1.0f;
            ptc.cell[offset + i * 2] = ptc.cell[offset + i * 2 + 1] =
                idx.linear;
            ptc.weight[offset + i * 2] = ptc.weight[offset + i * 2 + 1] =
                weight;
            ptc.flag[offset + i * 2] = set_ptc_type_flag(0, PtcType::electron);
            ptc.flag[offset + i * 2 + 1] =
                set_ptc_type_flag(0, PtcType::positron);
          }
        }
      },
      ptc.get_dev_ptrs(), surface_ne.dev_ptr(), surface_np.dev_ptr(),
      num_per_cell, rand_states.states());
  CudaSafeCall(cudaDeviceSynchronize());

  ptc.add_num(num_per_cell * 2 * grid.dims[1]);
}

template <typename Conf>
boundary_condition<Conf>::boundary_condition(sim_environment& env,
                                             const grid_t<Conf>& grid)
    : system_t(env), m_grid(grid) {
  using multi_array_t = typename Conf::multi_array_t;
  m_env.params().get_value("damping_length", m_damping_length);
  m_env.params().get_value("pmllen", m_pmllen);
  m_env.params().get_value("sigpml", m_sigpml);

  m_prev_E1 = std::make_unique<multi_array_t>(
      extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  m_prev_E2 = std::make_unique<multi_array_t>(
      extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  m_prev_E3 = std::make_unique<multi_array_t>(
      extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  m_prev_B1 = std::make_unique<multi_array_t>(
      extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  m_prev_B2 = std::make_unique<multi_array_t>(
      extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  m_prev_B3 = std::make_unique<multi_array_t>(
      extent(m_damping_length, m_grid.dims[1]), MemType::device_only);

  m_prev_E1->assign_dev(0.0f);
  m_prev_E2->assign_dev(0.0f);
  m_prev_E3->assign_dev(0.0f);
  m_prev_B1->assign_dev(0.0f);
  m_prev_B2->assign_dev(0.0f);
  m_prev_B3->assign_dev(0.0f);

  m_prev_E.set_memtype(MemType::host_device);
  m_prev_B.set_memtype(MemType::host_device);
  m_prev_E.resize(3);
  m_prev_B.resize(3);
  m_prev_E[0] = m_prev_E1->dev_ptr();
  m_prev_E[1] = m_prev_E2->dev_ptr();
  m_prev_E[2] = m_prev_E3->dev_ptr();
  m_prev_B[0] = m_prev_B1->dev_ptr();
  m_prev_B[1] = m_prev_B2->dev_ptr();
  m_prev_B[2] = m_prev_B3->dev_ptr();
  m_prev_E.copy_to_device();
  m_prev_B.copy_to_device();
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

  m_env.params().get_value("tp_start", m_tp_start);
  m_env.params().get_value("tp_end", m_tp_end);
  m_env.params().get_value("nT", m_nT);
  m_env.params().get_value("dw0", m_dw0);

  m_surface_ne.set_memtype(MemType::host_device);
  m_surface_ne.resize(m_grid.dims[1]);
  m_surface_np.set_memtype(MemType::host_device);
  m_surface_np.resize(m_grid.dims[1]);
}

template <typename Conf>
void
boundary_condition<Conf>::update(double dt, uint32_t step) {
  typedef typename Conf::idx_t idx_t;
  typedef typename Conf::value_t value_t;

  value_t time = m_env.get_time();
  wpert_cart_t wpert(m_tp_start, m_tp_end, m_nT, m_dw0);

  // Apply twist on the stellar surface
  kernel_launch(
      [time] __device__(auto e, auto b, auto e0, auto b0, auto wpert) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        for (auto n1 : grid_stride_range(0, grid.dims[1])) {
          value_t y = grid.template pos<1>(n1, false);
          value_t y_s = grid.template pos<1>(n1, true);

          // For quantities that are not continuous across the surface
          for (int n0 = 0; n0 < grid.skirt[0]; n0++) {
            auto idx = idx_t(index_t<2>(n0, n1), ext);
            value_t x = grid.template pos<0>(n0, false);
            value_t omega = wpert(time, x, y_s);
            // printf("omega is %f\n", omega);
            e[0][idx] = omega * b0[1][idx];
            b[1][idx] = 0.0;
            b[2][idx] = 0.0;
          }
          // For quantities that are continuous across the surface
          for (int n0 = 0; n0 < grid.skirt[0] + 1; n0++) {
            auto idx = idx_t(index_t<2>(n0, n1), ext);
            value_t x_s = grid.template pos<0>(n0, true);
            value_t omega = wpert(time, x_s, y);
            b[0][idx] = 0.0;
            e[1][idx] = -omega * b0[0][idx];
            e[2][idx] = 0.0;
          }
        }
      },
      E->get_ptrs(), B->get_ptrs(), E0->get_ptrs(), B0->get_ptrs(), wpert);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  // Apply damping boundary condition on the other side
  kernel_launch(
      [] __device__(auto e, auto b, auto prev_e, auto prev_b, auto damping_length,
                    auto pmllen, auto sigpml) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        auto ext_damping = extent(damping_length, grid.dims[1]);
        for (auto n1 : grid_stride_range(0, grid.dims[1])) {
          auto n0_start = grid.dims[0] - damping_length;
          auto xh = grid.template pos<0>(n0_start, true);
          for (int n0 = n0_start; n0 < grid.dims[0]; n0++) {
            auto idx = idx_t(index(n0, n1), ext);
            auto idx_damping = idx_t(index(n0 - n0_start, n1), ext_damping);
            auto x = grid.template pos<0>(n0, false);
            auto x_s = grid.template pos<0>(n0, true);
            auto sig = pml_sigma(x, xh, pmllen, sigpml);
            if (sig > TINY) {
              auto exp_sig = math::exp(-sig);
              e[0][idx] = exp_sig * prev_e[0][idx_damping] +
                  (1.0f - exp_sig) / sig * (e[0][idx] - prev_e[0][idx_damping]);
              b[1][idx] = exp_sig * prev_b[1][idx_damping] +
                  (1.0f - exp_sig) / sig * (b[1][idx] - prev_b[1][idx_damping]);
              b[2][idx] = exp_sig * prev_b[2][idx_damping] +
                  (1.0f - exp_sig) / sig * (b[2][idx] - prev_b[2][idx_damping]);
            }
            auto sig_s = pml_sigma(x_s, xh, pmllen, sigpml);
            if (sig_s > TINY) {
              auto exp_sig = math::exp(-sig_s);
              e[1][idx] = exp_sig * prev_e[1][idx_damping] +
                  (1.0f - exp_sig) / sig * (e[1][idx] - prev_e[1][idx_damping]);
              e[2][idx] = exp_sig * prev_e[2][idx_damping] +
                  (1.0f - exp_sig) / sig * (e[2][idx] - prev_e[2][idx_damping]);
              b[0][idx] = exp_sig * prev_b[0][idx_damping] +
                  (1.0f - exp_sig) / sig * (b[0][idx] - prev_b[0][idx_damping]);
            }
          }
        }
      },
      E->get_ptrs(), B->get_ptrs(), m_prev_E.dev_ptr(), m_prev_B.dev_ptr(),
      m_damping_length, m_pmllen, m_sigpml);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  // Store the current values of the field for the damping boundary of next time
  // step
  kernel_launch([] __device__(auto e, auto prev_e, auto b, auto prev_b,
                              auto damping_length) {
      auto& grid = dev_grid<Conf::dim>();
      auto ext = grid.extent();
      auto ext_damping = extent(damping_length, grid.dims[1]);
      for (auto n1 : grid_stride_range(0, grid.dims[1])) {
        auto n0_start = grid.dims[0] - damping_length;
        for (int n0 = n0_start; n0 < grid.dims[0]; n0++) {
          auto pos = index(n0, n1);
          auto idx = idx_t(pos, ext);
          auto idx_damping = idx_t(index(n0 - n0_start, n1), ext_damping);

          prev_e[0][idx_damping] = e[0][idx];
          prev_e[1][idx_damping] = e[1][idx];
          prev_e[2][idx_damping] = e[2][idx];
          prev_b[0][idx_damping] = b[0][idx];
          prev_b[1][idx_damping] = b[1][idx];
          prev_b[2][idx_damping] = b[2][idx];
        }
      }
    }, E->get_ptrs(), m_prev_E.dev_ptr(), B->get_ptrs(), m_prev_B.dev_ptr(), m_damping_length);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  // Inject particles
  if (step % 1 == 0 && time > m_tp_start && time < m_tp_end) {
    inject_particles<Conf>(*ptc, *rand_states, m_surface_ne, m_surface_np, 2,
                           1.0, m_grid);
  }
}

template class boundary_condition<Config<2>>;

}  // namespace Aperture
