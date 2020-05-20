#include "ptc_updater_magnetar.h"
#include "data/curand_states.h"
#include "framework/config.h"
#include "systems/grid_sph.h"
#include "systems/forces/sync_cooling.hpp"
#include "systems/forces/gravity.hpp"
#include "systems/helpers/ptc_update_helper.hpp"
#include "utils/kernel_helper.hpp"

namespace Aperture {

template <typename Pusher>
struct pusher_impl_magnetar {
  Pusher pusher;
  double cooling_coef = 0.0, B0 = 1.0, g0 = 0.1;
  double res_drag_coef = 0.0, BQ = 1.0, star_kT = 0.01;
  bool gravity_on = false;
  curandState* rand_states;

  pusher_impl_magnetar(sim_environment& env) {
    env.params().get_value("sync_cooling_coef", cooling_coef);
    env.params().get_value("Bp", B0);
    env.params().get_value("gravity", g0);
    env.params().get_value("gravity_on", gravity_on);
    env.params().get_value("res_drag_coef", res_drag_coef);
    env.params().get_value("BQ", BQ);
    env.params().get_value("star_kT", star_kT);

    curand_states_t* rand;
    env.get_data("rand_states", &rand);
    rand_states = rand->states();
  }

  HOST_DEVICE pusher_impl_magnetar(const pusher_impl_magnetar<Pusher>& other) = default;

  template <typename Scalar>
  __device__ void operator()(ptc_ptrs& ptc, uint32_t n, EB_t<Scalar>& EB,
                            Scalar qdt_over_2m, Scalar dt) {
    using Conf = Config<2>;
    auto& grid = dev_grid<2>();
    auto ext = grid.extent();
    auto idx = Conf::idx(ptc.cell[n], ext);
    auto pos = idx.get_pos();

    Scalar p1 = ptc.p1[n], p2 = ptc.p2[n], p3 = ptc.p3[n];
    Scalar gamma = ptc.E[n];
    Scalar r = grid_sph_t<Conf>::radius(grid.template pos<0>(pos[0], ptc.x1[n]));

    pusher(p1, p2, p3, gamma, EB.E1, EB.E2,
           EB.E3, EB.B1, EB.B2, EB.B3, qdt_over_2m, dt);

    if (gravity_on) {
      gravity(p1, p2, p3, gamma, r, dt, (Scalar)g0, qdt_over_2m * 2.0f / dt);
    }

    sync_kill_perp(p1, p2, p3, gamma, EB.E1, EB.E2, EB.E3,
                   EB.B1, EB.B2, EB.B3, qdt_over_2m * 2.0f / dt,
                   (Scalar)cooling_coef, (Scalar)B0);

    auto flag = ptc.flag[n];
    int sp = get_ptc_type(flag);
    if (sp != (int)PtcType::ion) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      cuda_rng_t rng(&rand_states[tid]);

      // Compute resonant drag
      Scalar p = sqrt(p1 * p1 + p2 * p2 + p3 * p3);
      Scalar B = sqrt(EB.B1 * EB.B1 + EB.B2 * EB.B2 + EB.B3 * EB.B3);
      Scalar pdotB = (p1 * EB.B1 + p2 * EB.B2 + p3 * EB.B3) / B;

      Scalar pB1 = p1 / p;
      Scalar pB2 = p2 / p;
      Scalar pB3 = p3 / p;

      Scalar mu = std::abs(EB.B1 / B);
      Scalar p_mag_signed = sgn(pdotB) * sgn(EB.B1) * std::abs(pdotB);
      Scalar g = sqrt(1.0f + p_mag_signed * p_mag_signed);

      Scalar beta = sqrt(1.0f - 1.0f / (g * g));
      Scalar y = std::abs((B / BQ) /
                          (star_kT * (g - p_mag_signed * mu)));
      if (y < 30.0f && y > 0.0f) {
        Scalar coef = res_drag_coef * square(star_kT) * y *
                      y / (r * r * (std::exp(y) - 1.0f));
        Scalar Nph = std::abs(coef / gamma) * dt;
        Scalar Eph =
            min(g - 1.0f,
                g * (1.0f - 1.0f / std::sqrt(1.0f + 2.0f * B / BQ)));

        if (Eph > 2.0f) {
          // Produce individual tracked photons
          if (Nph < 1.0f) {
            float u = rng();
            if (u < Nph)
              set_flag(ptc.flag[n], PtcFlag::emit_photon);
          } else {
            set_flag(ptc.flag[n], PtcFlag::emit_photon);
          }
        } else {
          // Compute analytically the drag force on the particle
          Scalar drag_coef =
              coef * star_kT * y * (g * mu - p_mag_signed);
          if (EB.B1 < 0.0f)
            drag_coef = -drag_coef;
          p1 += EB.B1 * dt * drag_coef / B;
          p2 += EB.B2 * dt * drag_coef / B;
          p3 += EB.B3 * dt * drag_coef / B;
        }
      }
    }

    ptc.p1[n] = p1;
    ptc.p2[n] = p2;
    ptc.p3[n] = p3;
    ptc.E[n] = gamma;
  }
};

template <typename Conf>
ptc_updater_magnetar<Conf>::ptc_updater_magnetar(sim_environment& env,
                                                 const grid_sph_t<Conf>& grid,
                                                 const domain_comm<Conf>* comm)
    : ptc_updater_sph_cu<Conf>(env, grid, comm) {}

template <typename Conf>
void
ptc_updater_magnetar<Conf>::init() {
  ptc_updater_sph_cu<Conf>::init();

  m_impl_boris = std::make_unique<pusher_impl_magnetar<boris_pusher>>(this->m_env);
  m_impl_vay = std::make_unique<pusher_impl_magnetar<vay_pusher>>(this->m_env);
  m_impl_higuera = std::make_unique<pusher_impl_magnetar<higuera_pusher>>(this->m_env);
}

template <typename Conf>
void
ptc_updater_magnetar<Conf>::register_data_components() {
  ptc_updater_sph_cu<Conf>::register_data_components();

  int ph_flux_n_th = 256, ph_flux_n_E = 100;
  this->m_env.params().get_value("ph_flux_n_th", ph_flux_n_th);
  this->m_env.params().get_value("ph_flux_n_E", ph_flux_n_E);
  m_ph_flux = this->m_env.template register_data<multi_array_data<float, 2>>(
      "ph_flux", extent(ph_flux_n_E, ph_flux_n_th), MemType::host_device);
}

template <typename Conf>
void
ptc_updater_magnetar<Conf>::push_default(double dt) {
  // dispatch according to enum. This will also instantiate all the versions of
  // push
  if (this->m_pusher == Pusher::boris) {
    this->push(dt, *m_impl_boris);
  } else if (this->m_pusher == Pusher::vay) {
    this->push(dt, *m_impl_vay);
  } else if (this->m_pusher == Pusher::higuera) {
    this->push(dt, *m_impl_higuera);
  }
}

#include "systems/ptc_updater_cu_impl.hpp"

template class ptc_updater_magnetar<Config<2>>;

}  // namespace Aperture
