#include "core/math.hpp"
#include "core/particle_structs.h"
#include "framework/config.h"
#include "systems/radiative_transfer_cu_impl.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
struct ph_freepath_dev_t {
  typedef typename Conf::value_t value_t;

  float gamma_thr = 30.0f;
  float E_s = 2.0f;
  float photon_path = 0.0f;

  HOST_DEVICE ph_freepath_dev_t() {}
  ph_freepath_dev_t(sim_environment& env) {
    env.params().get_value("gamma_thr", gamma_thr);
    env.params().get_value("E_s", E_s);
    env.params().get_value("photon_path", photon_path);
  }
  HOST_DEVICE ph_freepath_dev_t(const ph_freepath_dev_t& other) = default;

  __device__ bool check_emit_photon(ptc_ptrs& ptc, uint32_t tid,
                                    cuda_rng_t& rng) {
    return ptc.E[tid] > gamma_thr;
  }

  __device__ void emit_photon(ptc_ptrs& ptc, uint32_t tid, ph_ptrs& ph,
                              uint32_t offset, cuda_rng_t& rng) {
    value_t p1 = ptc.p1[tid];
    value_t p2 = ptc.p2[tid];
    value_t p3 = ptc.p3[tid];
    // value_t gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
    value_t gamma = ptc.E[tid];
    value_t pi = std::sqrt(gamma * gamma - 1.0f);

    value_t u = rng();
    value_t Eph = 2.5f + u * (E_s - 1.0f) * 2.0f;
    value_t pf = std::sqrt(square(gamma - Eph) - 1.0f);

    ptc.p1[tid] = p1 * pf / pi;
    ptc.p2[tid] = p2 * pf / pi;
    ptc.p3[tid] = p3 * pf / pi;
    ptc.E[tid] = gamma - Eph;

    auto c = ptc.cell[tid];
    auto& grid = dev_grid<Conf::dim>();
    auto idx = typename Conf::idx_t(c, grid.extent());
    auto pos = idx.get_pos();
    value_t theta = grid.pos<1>(pos[0], ptc.x2[tid]);
    value_t lph = min(10.0f, (1.0f / std::sin(theta) - 1.0f) * photon_path);
    // If photon energy is too low, do not track it, but still
    // subtract its energy as done above
    // if (std::abs(Eph) < dev_params.E_ph_min) continue;
    if (theta < 0.005f || theta > M_PI - 0.005f) return;

    u = rng();
    // Add the new photo
    value_t path = lph * (0.5f + 0.5f * u);
    // if (path > dev_params.r_cutoff) return;
    // printf("Eph is %f, path is %f\n", Eph, path);
    ph.x1[offset] = ptc.x1[tid];
    ph.x2[offset] = ptc.x2[tid];
    ph.x3[offset] = ptc.x3[tid];
    ph.p1[offset] = Eph * p1 / pi;
    ph.p2[offset] = Eph * p2 / pi;
    ph.p3[offset] = Eph * p3 / pi;
    ph.weight[offset] = ptc.weight[tid];
    ph.path_left[offset] = path;
    ph.cell[offset] = ptc.cell[tid];
  }

  __device__ bool check_produce_pair(ph_ptrs& ph, uint32_t tid,
                                     cuda_rng_t& rng) {
    return ph.path_left[tid] < photon_path;
  }

  __device__ void produce_pair(ph_ptrs& ph, uint32_t tid, ptc_ptrs& ptc,
                               uint32_t offset, cuda_rng_t& rng) {
    value_t p1 = ph.p1[tid];
    value_t p2 = ph.p2[tid];
    value_t p3 = ph.p3[tid];
    value_t Eph2 = p1 * p1 + p2 * p2 + p3 * p3;
    if (Eph2 < 4.01f) Eph2 = 4.01f;

    value_t ratio = math::sqrt(0.25f - 1.0f / Eph2);
    value_t gamma = math::sqrt(1.0f + ratio * ratio * Eph2);
    uint32_t offset_e = offset;
    uint32_t offset_p = offset + 1;

    ptc.x1[offset_e] = ptc.x1[offset_p] = ph.x1[tid];
    ptc.x2[offset_e] = ptc.x2[offset_p] = ph.x2[tid];
    ptc.x3[offset_e] = ptc.x3[offset_p] = ph.x3[tid];
    // printf("x1 = %f, x2 = %f, x3 = %f\n",
    // ptc.x1[offset_e],
    // ptc.x2[offset_e], ptc.x3[offset_e]);

    ptc.p1[offset_e] = ptc.p1[offset_p] = ratio * p1;
    ptc.p2[offset_e] = ptc.p2[offset_p] = ratio * p2;
    ptc.p3[offset_e] = ptc.p3[offset_p] = ratio * p3;
    ptc.E[offset_e] = ptc.E[offset_p] = gamma;

#ifndef NDEBUG
    assert(ptc.cell[offset_e] == empty_cell);
    assert(ptc.cell[offset_p] == empty_cell);
#endif
    ptc.weight[offset_e] = ptc.weight[offset_p] = ph.weight[tid];
    ptc.cell[offset_e] = ptc.cell[offset_p] = ph.cell[tid];
    ptc.flag[offset_e] =
        set_ptc_type_flag(flag_or(PtcFlag::secondary), PtcType::electron);
    ptc.flag[offset_p] =
        set_ptc_type_flag(flag_or(PtcFlag::secondary), PtcType::positron);

    // Set this photon to be empty
    ph.cell[tid] = empty_cell;
  }
};

template class radiative_transfer_cu<Config<2>, ph_freepath_dev_t<Config<2>>>;

}  // namespace Aperture
