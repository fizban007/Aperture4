#include "core/math.hpp"
#include "core/particle_structs.h"
#include "framework/config.h"
#include "systems/radiative_transfer_cu_impl.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
struct rt_magnetar_dev_t {
  typedef typename Conf::value_t value_t;

  float BQ = 1.0f;
  float photon_path = 0.0f;

  vec_t<typename Conf::ndptr_const_t, 3> B;

  HOST_DEVICE rt_magnetar_dev_t() {}
  rt_magnetar_dev_t(sim_environment& env) {
    env.params().get_value("BQ", BQ);
    env.params().get_value("photon_path", photon_path);

    const vector_field<Conf>* B_data;
    env.get_data("B", &B_data);
    B = B_data->get_ptrs();
  }

  HOST_DEVICE rt_magnetar_dev_t(const rt_magnetar_dev_t& other) = default;

  __device__ bool check_emit_photon(ptc_ptrs& ptc, uint32_t tid,
                                    cuda_rng_t& rng) {
    bool emit = check_flag(ptc.flag[tid], PtcFlag::emit_photon);

    if (emit) {
      // Reset the flag of the emitting particle
      ptc.flag[tid] &= ~flag_or(PtcFlag::emit_photon);
    }
    return emit;
  }

  __device__ void emit_photon(ptc_ptrs& ptc, uint32_t tid, ph_ptrs& ph,
                              uint32_t offset, cuda_rng_t& rng) {
    auto& grid = dev_grid<Conf::dim>();
    auto ext = grid.extent();
    auto c = ptc.cell[tid];
    auto idx = Conf::idx(c, ext);
    auto pos = idx.get_pos();
    value_t p1 = ptc.p1[tid];
    value_t p2 = ptc.p2[tid];
    value_t p3 = ptc.p3[tid];
    auto x = vec_t<Pos_t, 3>(ptc.x1[tid], ptc.x2[tid], ptc.x3[tid]);
    // value_t gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
    value_t gamma = ptc.E[tid];
    // value_t pi = math::sqrt(gamma * gamma - 1.0f);
    value_t pi = math::sqrt(p1 * p1 + p2 * p2 + p3 * p3);
    // value_t u = rng();

    auto interp = interpolator<typename Conf::spline_t, Conf::dim>{};
    value_t B1 = interp(B[0], x, idx, stagger_t(0b001));
    value_t B2 = interp(B[1], x, idx, stagger_t(0b010));
    value_t B3 = interp(B[2], x, idx, stagger_t(0b100));
    value_t B = sqrt(B1 * B1 + B2 * B2 + B3 * B3);
    value_t p = sqrt(p1 * p1 + p2 * p2 + p3 * p3);
    // value_t pB1 = p1 / p;
    // value_t pB2 = p2 / p;
    // value_t pB3 = p3 / p;

    value_t beta = math::abs(p) / gamma;
    // float theta_p = CONST_PI * rng();
    float u = rng();
    value_t phi_p = 2.0f * M_PI * rng();
    value_t cphi = cos(phi_p);
    value_t sphi = sin(phi_p);

    value_t Eph =
        math::abs(gamma * (1.0f + beta * u) *
                 (1.0f - 1.0f / math::sqrt(1.0f + 2.0f * B / BQ)));
    value_t pf = math::sqrt(square(gamma - Eph) - 1.0f);
    p1 *= pf / pi;
    p2 *= pf / pi;
    p3 *= pf / pi;
    ptc.p1[tid] = p1;
    ptc.p2[tid] = p2;
    ptc.p3[tid] = p3;
    ptc.E[tid] = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
  }

  __device__ bool check_produce_pair(ph_ptrs& ph, uint32_t tid,
                                     cuda_rng_t& rng) {
    return ph.path_left[tid] < photon_path;
  }

  __device__ void produce_pair(ph_ptrs& ph, uint32_t tid, ptc_ptrs& ptc,
                               uint32_t offset, cuda_rng_t& rng) {
  }
};

template class radiative_transfer_cu<Config<2>, rt_magnetar_dev_t<Config<2>>>;

}  // namespace Aperture
