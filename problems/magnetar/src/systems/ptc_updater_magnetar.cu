/*
 * Copyright (c) 2020 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "data/curand_states.h"
#include "framework/config.h"
#include "ptc_updater_magnetar.h"
#include "systems/grid_sph.hpp"
#include "systems/helpers/ptc_update_helper.hpp"
#include "systems/physics/gravity.hpp"
#include "systems/physics/sync_cooling.hpp"
#include "utils/kernel_helper.hpp"

namespace Aperture {

template <typename Pusher>
struct pusher_impl_magnetar {
  Pusher pusher;
  double cooling_coef = 0.0, B0 = 1.0, g0 = 0.1;
  double res_drag_coef = 0.0, BQ = 1.0, star_kT = 0.01;
  bool gravity_on = false;
  curandState* rand_states;
  ndptr<float, 2> ph_flux;
  int flux_n_th;
  int flux_n_E;

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

    multi_array_data<float, 2>* ph_flux_data;
    env.get_data("ph_flux", &ph_flux_data);
    ph_flux = ph_flux_data->dev_ndptr();
    auto ext = ph_flux_data->extent();
    flux_n_E = ext[0];
    flux_n_th = ext[1];
  }
  HOST_DEVICE pusher_impl_magnetar(const pusher_impl_magnetar<Pusher>& other) =
      default;
  HOST_DEVICE pusher_impl_magnetar() {}

  template <typename Scalar>
  __device__ void operator()(ptc_ptrs& ptc, uint32_t n, EB_t<Scalar>& EB,
                             Scalar qdt_over_2m, Scalar dt) {
    using Conf = Config<2>;
    auto& grid = dev_grid<2>();
    auto idx = Conf::idx(ptc.cell[n], grid.extent());
    auto pos = idx.get_pos();

    Scalar p1 = ptc.p1[n], p2 = ptc.p2[n], p3 = ptc.p3[n];
    Scalar gamma = ptc.E[n];
    Scalar r =
        grid_sph_t<Conf>::radius(grid.template pos<0>(pos[0], ptc.x1[n]));

    pusher(p1, p2, p3, gamma, EB.E1, EB.E2, EB.E3, EB.B1, EB.B2, EB.B3,
           qdt_over_2m, dt);

    // printf("After pusher, p1, p2, p3 are %f, %f, %f\n", p1, p2, p3);

    if (gravity_on) {
      gravity(p1, p2, p3, gamma, r, dt, (Scalar)g0, qdt_over_2m * 2.0f / dt);
    }

    // printf("cooling_coef is %f, B0 is %f\n", (Scalar)cooling_coef,
    // (Scalar)B0);
    sync_kill_perp(p1, p2, p3, gamma, EB.E1, EB.E2, EB.E3, EB.B1, EB.B2, EB.B3,
                   qdt_over_2m * 2.0f / dt, (Scalar)cooling_coef, (Scalar)B0);

    // printf("After sync, p1, p2, p3 are %f, %f, %f\n", p1, p2, p3);

    auto flag = ptc.flag[n];
    int sp = get_ptc_type(flag);
    if (sp != (int)PtcType::ion) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      cuda_rng_t rng(&rand_states[tid]);

      // Compute resonant drag
      Scalar p = math::sqrt(p1 * p1 + p2 * p2 + p3 * p3);
      Scalar B = math::sqrt(EB.B1 * EB.B1 + EB.B2 * EB.B2 + EB.B3 * EB.B3);
      Scalar pdotB = (p1 * EB.B1 + p2 * EB.B2 + p3 * EB.B3) / B;

      Scalar mu = math::abs(EB.B1 / B);
      // Scalar p_mag_signed = sgn(pdotB) * sgn(EB.B1) * math::abs(pdotB);
      Scalar p_mag_signed = sgn(p1) * math::abs(pdotB);
      Scalar g = sqrt(1.0f + p_mag_signed * p_mag_signed);

      Scalar beta = sqrt(1.0f - 1.0f / (g * g));
      Scalar y = math::abs((B / BQ) / (star_kT * (g - p_mag_signed * mu)));
      // printf("y is %f, res_drag_coef is %f, star_kT is %f\n", y,
      //        res_drag_coef, star_kT);
      if (y < 30.0f && y > 0.0f) {
        Scalar coef = res_drag_coef * square(star_kT) * y * y /
                      (r * r * (math::exp(y) - 1.0f));
        Scalar Nph = math::abs(coef / gamma) * dt;
        Scalar Eph =
            min(g - 1.0f, g * (1.0f - 1.0f / math::sqrt(1.0f + 2.0f * B / BQ)));

        if (Eph > 2.0f) {
          // Produce individual tracked photons
          if (Nph < 1.0f) {
            float u = rng();
            if (u < Nph) set_flag(ptc.flag[n], PtcFlag::emit_photon);
          } else {
            set_flag(ptc.flag[n], PtcFlag::emit_photon);
          }
        } else {
          // Compute analytically the drag force on the particle and apply it
          Scalar drag_coef = coef * star_kT * y * (g * mu - p_mag_signed);
          // printf("coef is %f, drag coef is %f\n", coef, drag_coef);
          // printf("r is %f, theta is %f, gamma is %f, p_par is %f\n", r,
          // theta,
          //        gamma, p_mag_signed);
          if (EB.B1 < 0.0f) drag_coef = -drag_coef;
          p1 += EB.B1 * dt * drag_coef / B;
          p2 += EB.B2 * dt * drag_coef / B;
          p3 += EB.B3 * dt * drag_coef / B;

          // printf("After resonant drag, p1, p2, p3 are %f, %f, %f\n", p1, p2,
          // p3);

          // Draw emission direction in the particle rest frame,
          // direction is the particle moving direction
          Scalar phi_p = 2.0f * M_PI * rng();
          Scalar u = 2.0f * rng() - 1.0f;
          Scalar cphi = math::cos(phi_p);
          Scalar sphi = math::sin(phi_p);

          Eph = g * (1.0f + math::abs(beta) * u) *
                (1.0f - 1.0f / sqrt(1.0f + 2.0f * B / BQ));

          // Lorentz transform u to the lab frame
          u = (u + beta) / (1 + beta * u);
          Scalar n1 = p1 / p;
          Scalar n2 = p2 / p;
          Scalar n3 = p3 / p;
          Scalar np = math::sqrt(n1 * n1 + n2 * n2);

          Scalar sth = sqrt(1.0f - u * u);
          Scalar ph1 = (n1 * u + sth * (n2 * cphi + n1 * n3 * sphi) / np);
          Scalar ph2 = (n2 * u + sth * (-n1 * cphi + n2 * n3 * sphi) / np);
          // ph3 = (n3 * u - sth * (-np * sphi));

          // Compute the theta of the photon outgoing direction
          if (ph1 > 0.0f) {
            Scalar theta = grid_sph_t<Conf>::theta(
                grid.template pos<1>(pos[1], ptc.x2[n]));
            Scalar cth_p = ph1 * math::cos(theta) - ph2 * math::sin(theta);
            Eph = math::log(math::abs(Eph)) / math::log(10.0f);
            if (Eph > 2.0f) Eph = 2.0f;
            if (Eph < -6.0f) Eph = -6.0f;
            int n0 = ((Eph + 6.0f) / 8.02f * (flux_n_E - 1));
            if (n0 < 0) n0 = 0;
            if (n0 >= flux_n_E) n0 = flux_n_E - 1;
            int n1 = (0.5f * (cth_p + 1.0f)) * (flux_n_th - 1);
            if (n1 < 0) n1 = 0;
            if (n1 >= flux_n_th) n1 = flux_n_th - 1;
            auto ph_idx =
                idx_col_major_t<2>(index(n0, n1), extent(flux_n_E, flux_n_th));
            atomicAdd(&ph_flux[ph_idx], Nph * ptc.weight[n]);
            // printf("n0 is %d, n1 is %d, Ndot is %f, ph_flux is %f\n",
            // n0,
            //        n1, Ndot, ph_flux(n0, n1));
          }
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
    : ptc_updater_old_sph_cu<Conf>(env, grid, comm) {}

template <typename Conf>
ptc_updater_magnetar<Conf>::~ptc_updater_magnetar() {
  if (m_impl_boris != nullptr) delete m_impl_boris;
  if (m_impl_vay != nullptr) delete m_impl_vay;
  if (m_impl_higuera != nullptr) delete m_impl_higuera;
}

template <typename Conf>
void
ptc_updater_magnetar<Conf>::init() {
  ptc_updater_old_sph_cu<Conf>::init();

  // m_impl_boris =
  // std::make_unique<pusher_impl_magnetar<boris_pusher>>(this->m_env);
  // m_impl_vay =
  // std::make_unique<pusher_impl_magnetar<vay_pusher>>(this->m_env);
  // m_impl_higuera =
  // std::make_unique<pusher_impl_magnetar<higuera_pusher>>(this->m_env);
  if (this->m_pusher == Pusher::boris) {
    m_impl_boris = new pusher_impl_magnetar<boris_pusher>(this->m_env);
  } else if (this->m_pusher == Pusher::vay) {
    m_impl_vay = new pusher_impl_magnetar<vay_pusher>(this->m_env);
  } else if (this->m_pusher == Pusher::higuera) {
    m_impl_higuera = new pusher_impl_magnetar<higuera_pusher>(this->m_env);
  }

  m_ph_flux->assign_dev(0.0f);
  m_ph_flux->assign_host(0.0f);
}

template <typename Conf>
void
ptc_updater_magnetar<Conf>::register_data_components() {
  ptc_updater_old_sph_cu<Conf>::register_data_components();

  int ph_flux_n_th = 256, ph_flux_n_E = 100;
  this->m_env.params().get_value("ph_flux_n_th", ph_flux_n_th);
  this->m_env.params().get_value("ph_flux_n_E", ph_flux_n_E);
  m_ph_flux = this->m_env.template register_data<multi_array_data<float, 2>>(
      "ph_flux", extent(ph_flux_n_E, ph_flux_n_th), MemType::host_device);
  m_ph_flux->reset_after_output(true);
}

template <typename Conf>
void
ptc_updater_magnetar<Conf>::push_default(double dt) {
  m_ph_flux->init();
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
// template struct pusher_impl_magnetar<boris_pusher>;
// template struct pusher_impl_magnetar<vay_pusher>;
// template struct pusher_impl_magnetar<higuera_pusher>;

}  // namespace Aperture
