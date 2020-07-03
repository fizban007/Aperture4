#include "core/math.hpp"
#include "core/particle_structs.h"
#include "data/multi_array_data.hpp"
#include "framework/config.h"
#include "systems/radiative_transfer_cu_impl.hpp"
#include "systems/grid_sph.h"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
struct rt_magnetar_impl_t {
  typedef typename Conf::value_t value_t;

  float BQ = 1.0f;
  float photon_path = 0.0f;

  vec_t<typename Conf::ndptr_const_t, 3> B;
  ndptr<float, 2> ph_flux;
  int flux_n_th;
  int flux_n_E;

  HOST_DEVICE rt_magnetar_impl_t() {}
  rt_magnetar_impl_t(sim_environment& env) {
    env.params().get_value("BQ", BQ);
    env.params().get_value("photon_path", photon_path);

    const vector_field<Conf>* B_data;
    env.get_data("B", &B_data);
    B = B_data->get_ptrs();

    multi_array_data<float, 2>* ph_flux_data;
    env.get_data("ph_flux", &ph_flux_data);
    ph_flux = ph_flux_data->dev_ndptr();
    auto ext = ph_flux_data->extent();
    flux_n_E = ext[0];
    flux_n_th = ext[1];
  }

  HOST_DEVICE rt_magnetar_impl_t(const rt_magnetar_impl_t& other) = default;

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
    auto c = ptc.cell[tid];
    auto idx = Conf::idx(c, grid.extent());
    // auto pos = idx.get_pos();
    value_t p1 = ptc.p1[tid];
    value_t p2 = ptc.p2[tid];
    value_t p3 = ptc.p3[tid];
    auto x = vec_t<Pos_t, 3>(ptc.x1[tid], ptc.x2[tid], ptc.x3[tid]);
    // value_t gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
    value_t gamma = ptc.E[tid];
    // value_t pi = math::sqrt(gamma * gamma - 1.0f);
    // value_t u = rng();

    auto interp = interpolator<typename Conf::spline_t, Conf::dim>{};
    value_t B1 = interp(B[0], x, idx, stagger_t(0b001));
    value_t B2 = interp(B[1], x, idx, stagger_t(0b010));
    value_t B3 = interp(B[2], x, idx, stagger_t(0b100));
    value_t B = math::sqrt(B1 * B1 + B2 * B2 + B3 * B3);
    value_t p = math::sqrt(p1 * p1 + p2 * p2 + p3 * p3);

    value_t beta = math::abs(p) / gamma;
    // float theta_p = CONST_PI * rng();
    float u = 2.0f * rng() - 1.0f;
    value_t phi_p = 2.0f * M_PI * rng();
    value_t cphi = math::cos(phi_p);
    value_t sphi = math::sin(phi_p);

    value_t Eph =
        math::abs(gamma * (1.0f + beta * u) *
                 (1.0f - 1.0f / math::sqrt(1.0f + 2.0f * B / BQ)));

    // Lorentz transform u to the lab frame
    u = (u + beta) / (1 + beta * u);
    value_t n1 = p1 / p;
    value_t n2 = p2 / p;
    value_t n3 = p3 / p;
    value_t np = math::sqrt(n1 * n1 + n2 * n2);

    value_t sth = sqrt(1.0f - u * u);
    value_t ph1 = Eph * (n1 * u + sth * (n2 * cphi + n1 * n3 * sphi) / np);
    value_t ph2 = Eph * (n2 * u + sth * (-n1 * cphi + n2 * n3 * sphi) / np);
    value_t ph3 = Eph * (n3 * u - sth * (-np * sphi));

    ptc.p1[tid] = p1 - ph1;
    ptc.p2[tid] = p2 - ph2;
    ptc.p3[tid] = p3 - ph3;
    ptc.E[tid] = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

    // Actually produce the photons
    ph.x1[offset] = ptc.x1[tid];
    ph.x2[offset] = ptc.x2[tid];
    ph.x3[offset] = ptc.x3[tid];
    ph.p1[offset] = ph1;
    ph.p2[offset] = ph2;
    ph.p3[offset] = ph3;
    ph.E[offset] = Eph;
    ph.weight[offset] = ptc.weight[tid];
    ph.path_left[offset] = photon_path;
    ph.cell[offset] = ptc.cell[tid];
  }

  __device__ bool check_produce_pair(ph_ptrs& ph, uint32_t tid,
                                     cuda_rng_t& rng) {
    // return ph.path_left[tid] < photon_path;
    auto& grid = dev_grid<Conf::dim>();
    uint32_t cell = ph.cell[tid];
    auto idx = Conf::idx(cell, grid.extent());
    auto pos = idx.get_pos();
    auto x = vec_t<Pos_t, 3>(ph.x1[tid], ph.x2[tid], ph.x3[tid]);
    auto p1 = ph.p1[tid];
    auto p2 = ph.p2[tid];
    auto p3 = ph.p3[tid];
    auto Eph = ph.E[tid];
    value_t theta = grid_sph_t<Conf>::theta(grid.template pos<1>(pos[1], x[1]));
    // Do not care about photons in the first and last theta cell
    if (theta < grid.delta[1] || theta > M_PI - grid.delta[1]) {
      ph.cell[tid] = empty_cell;
      return false;
    }

    auto interp = interpolator<typename Conf::spline_t, Conf::dim>{};
    value_t B1 = interp(B[0], x, idx, stagger_t(0b001));
    value_t B2 = interp(B[1], x, idx, stagger_t(0b010));
    value_t B3 = interp(B[2], x, idx, stagger_t(0b100));

    value_t B = sqrt(B1 * B1 + B2 * B2 + B3 * B3);
    if (Eph * B / BQ < 2.0f && p1 > 0.0f) {
      // TODO: destroy the photon and deposit it to flux
      value_t cth_p = (p1 * math::cos(theta) - p2 * math::sin(theta)) / Eph;
      Eph = math::log(math::abs(Eph)) / math::log(10.0f);
      if (Eph > 2.0f)
        Eph = 2.0f;
      if (Eph < -6.0f)
        Eph = -6.0f;
      int n0 = ((Eph + 6.0f) / 8.02f * (flux_n_E - 1));
      if (n0 < 0)
        n0 = 0;
      if (n0 >= flux_n_E)
        n0 = flux_n_E - 1;
      int n1 =
          (0.5f * (cth_p + 1.0f)) * (flux_n_th - 1);
      if (n1 < 0)
        n1 = 0;
      if (n1 >= flux_n_th)
        n1 = flux_n_th - 1;
      auto ph_idx = idx_col_major_t<2>(index(n0, n1), extent(flux_n_E, flux_n_th));
      atomicAdd(&ph_flux[ph_idx], ph.weight[tid]);
    }

    // Otherwise compute chi and convert to pair according to chi
    value_t cth = (B1 * p1 + B2 * p2 + B3 * p3) / (B * Eph);
    value_t chi = Eph * B * sqrt(1.0 - cth * cth) / BQ;
    return chi > 0.12;
  }

  __device__ void produce_pair(ph_ptrs& ph, uint32_t tid, ptc_ptrs& ptc,
                               uint32_t offset, cuda_rng_t& rng) {
    Scalar p1 = ph.p1[tid];
    Scalar p2 = ph.p2[tid];
    Scalar p3 = ph.p3[tid];
    Scalar E_ph2 = p1 * p1 + p2 * p2 + p3 * p3;
    if (E_ph2 <= 4.01f)
      E_ph2 = 4.01f;

    Scalar ratio = std::sqrt(0.25f - 1.0f / E_ph2);
    Scalar gamma = sqrt(1.0f + ratio * ratio * E_ph2);

    if (gamma != gamma) {
      ph.cell[tid] = empty_cell;
      return;
    }
    // Add the two new particles
    int offset_e = offset;
    int offset_p = offset + 1;

    ptc.x1[offset_e] = ptc.x1[offset_p] = ph.x1[tid];
    ptc.x2[offset_e] = ptc.x2[offset_p] = ph.x2[tid];
    ptc.x3[offset_e] = ptc.x3[offset_p] = ph.x3[tid];
    // printf("x1 = %f, x2 = %f, x3 = %f\n", ptc.x1[offset_e],
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
    ptc.flag[offset_e] = set_ptc_type_flag(flag_or(PtcFlag::secondary),
                                           PtcType::electron);
    ptc.flag[offset_p] = set_ptc_type_flag(flag_or(PtcFlag::secondary),
                                           PtcType::positron);

    // Set this photon to be empty
    ph.cell[tid] = empty_cell;
  }
};

template class radiative_transfer_cu<Config<2>, rt_magnetar_impl_t<Config<2>>>;

}  // namespace Aperture
