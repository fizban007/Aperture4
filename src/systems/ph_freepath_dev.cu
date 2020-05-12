#include "systems/radiative_transfer_cu_impl.hpp"
#include "framework/config.h"

namespace Aperture {

template <typename Conf>
struct ph_freepath_dev_t {
  float gamma_thr = 30.0f;
  float E_s = 2.0f;
  float photon_path = 0.0f;

  HOST_DEVICE ph_freepath_dev_t() {}
  ph_freepath_dev_t(sim_environment& env) {}
  HOST_DEVICE ph_freepath_dev_t(const ph_freepath_dev_t& other) = default;

  template <typename Ptc>
  __device__ bool check_emit_photon(Ptc& ptc, uint32_t tid, cuda_rng_t& rng) {
    return ptc.E[tid] > gamma_thr;
  }

  template <typename Ptc, typename Ph>
  __device__ void emit_photon(Ptc& ptc, uint32_t tid, Ph& ph, uint32_t offset,
                              cuda_rng_t& rng) {
    Scalar p1 = ptc.p1[tid];
    Scalar p2 = ptc.p2[tid];
    Scalar p3 = ptc.p3[tid];
    //   // Scalar gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
    Scalar gamma = ptc.E[tid];
    Scalar pi = std::sqrt(gamma * gamma - 1.0f);

    Scalar u = rng();
    Scalar Eph = 2.5f + u * (E_s - 1.0f) * 2.0f;
    Scalar pf = std::sqrt(square(gamma - Eph) - 1.0f);

    ptc.p1[tid] = p1 * pf / pi;
    ptc.p2[tid] = p2 * pf / pi;
    ptc.p3[tid] = p3 * pf / pi;
    ptc.E[tid] = gamma - Eph;

    auto c = ptc.cell[tid];
    auto& grid = dev_grid<Conf::dim>();
    auto idx = typename Conf::idx_t(c, grid.extent());
    auto pos = idx.get_pos();
    Scalar theta = grid.pos<1>(pos[0], ptc.x2[tid]);
    Scalar lph = min(
        10.0f, (1.0f / std::sin(theta) - 1.0f) * photon_path);
    // If photon energy is too low, do not track it, but still
    // subtract its energy as done above
    // if (std::abs(Eph) < dev_params.E_ph_min) continue;
    if (theta < 0.005f || theta > M_PI - 0.005f) return;

    u = rng();
    // Add the new photo
    Scalar path = lph * (0.5f + 0.5f * u);
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

  template <typename Ph>
  __device__ bool check_produce_pair(Ph& ph, uint32_t tid, cuda_rng_t& rng) {
    return false;
  }

  template <typename Ph, typename Ptc>
  __device__ void produce_pair(Ph& ph, uint32_t tid, Ptc& ptc, uint32_t offset,
                               cuda_rng_t& rng) {

  }
};


template class radiative_transfer_cu<Config<2>, ph_freepath_dev_t<Config<2>>>;

}
