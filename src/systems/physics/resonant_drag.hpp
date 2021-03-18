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

#ifndef __RESONANT_DRAG_H_
#define __RESONANT_DRAG_H_

#include "core/cuda_control.h"
#include "core/constant_mem.h"
#include "core/math.hpp"
#include "utils/util_functions.h"
#include <cstdint>

namespace Aperture {

template <typename Scalar>
__device__ void
resonant_drag(Scalar &p1, Scalar &p2, Scalar &p3, Scalar &gamma,
              Scalar r, Scalar B1, Scalar B2, Scalar B3, Scalar q_over_m,
              Scalar dt, Scalar &Eph,
              Scalar &th_obs) {
  Scalar p = sqrt(p1 * p1 + p2 * p2 + p3 * p3);
  Scalar B = sqrt(B1 * B1 + B2 * B2 + B3 * B3);
  Scalar pdotB = (p1 * B1 + p2 * B2 + p3 * B3) / B;

  Scalar pB1 = p1 / p;
  Scalar pB2 = p2 / p;
  Scalar pB3 = p3 / p;

  Scalar mu = std::abs(B1 / B);
  Scalar p_mag_signed = sgn(pdotB) * sgn(B1) * std::abs(pdotB);
  Scalar g = sqrt(1.0f + p_mag_signed * p_mag_signed);
  // Scalar p_mag_signed = sgn(p1) * p;
  // Scalar beta = sqrt(1.0f - 1.0f / (gamma * gamma));
  Scalar beta = sqrt(1.0f - 1.0f / (g * g));
  Scalar y = std::abs((B / dev_params.BQ) /
                      (dev_params.star_kT * (g - p_mag_signed * mu)));
  if (y < 30.0f && y > 0.0f) {
    Scalar coef = dev_params.res_drag_coef * square(dev_params.star_kT) * y *
                  y / (r * r * (std::exp(y) - 1.0f));
    Scalar Nph = std::abs(coef / gamma) * dt;
    Scalar Eph =
        min(g - 1.0f,
            g * (1.0f - 1.0f / std::sqrt(1.0f + 2.0f * B / dev_params.BQ)));
    // Scalar Eres = (B / dev_params.BQ) / (g - p_mag_signed * mu);
    // if (idx == 0) {
    //   printf("Nph is %f, Eph is %f, Eres is %f\n", Nph, Eph, Eres);
    //   printf("r is %f, theta is %f, gamma is %f, p_par is %f", r, theta,
    //   gamma, p_mag_signed);
    // }
    // Do not allow the particle to lose too much energy
    // if (Eph * Nph > gamma + Eres * Nph - 1.0f)
    //   Nph = (gamma + Eres * Nph - 1.0f) / Eph;

    // if (Eph > dev_params.E_ph_min) {
    // Scalar gamma_thr_B = dev_params.gamma_thr * B / dev_params.BQ;
    // if (Eph > 2.0f && gamma_thr_B > 3.0f && gamma > gamma_thr_B) {
    if (Eph > 2.0f) {
      // Produce individual tracked photons
      if (Nph < 1.0f) {
        float u = curand_uniform(&state);
        if (u < Nph)
          flag = (flag | bit_or(ParticleFlag::emit_photon));
      } else {
        flag = (flag | bit_or(ParticleFlag::emit_photon));
      }
    } else if (dev_params.rad_cooling_on) {
      // Compute analytically the drag force on the particle
      // Scalar p_mag_signed = sgn(pdotB) * sgn(B1) * std::abs(pdotB) / B;

      Scalar drag_coef =
          coef * dev_params.star_kT * y * (g * mu - p_mag_signed);
      if (B1 < 0.0f)
        drag_coef = -drag_coef;
      p1 += B1 * dt * drag_coef / B;
      p2 += B2 * dt * drag_coef / B;
      p3 += B3 * dt * drag_coef / B;
      // Produce low energy photons that are immediately deposited to
      // an array

      // Draw emission direction in the particle rest frame, z
      // direction is the particle moving direction
      Scalar theta_p = CONST_PI * curand_uniform(&state);
      Scalar phi_p = 2.0f * CONST_PI * curand_uniform(&state);
      Scalar u = cos(theta_p);
      Scalar cphi = cos(phi_p);
      Scalar sphi = sin(phi_p);

      Eph = g * (1.0f + std::abs(beta) * u) *
            (1.0f - 1.0f / sqrt(1.0f + 2.0f * B / dev_params.BQ));

      // Lorentz transform u to the lab frame
      u = (u + beta) / (1 + beta * u);
      Scalar ph1, ph2, ph3;
      Scalar sth = sqrt(1.0f - u * u);
      ph1 = (pB1 * u - sth * ((pB3 * pB3 + pB2 * pB2) * sphi));
      ph2 = (pB2 * u + sth * (pB3 * cphi + pB1 * pB2 * sphi));
      ph3 = (pB3 * u - sth * (pB2 * cphi + pB1 * pB3 * sphi));

      // Compute the theta of the photon outgoing direction
      if (ph1 > 0.0f) {
        logsph2cart(ph1, ph2, ph3, r, theta, phi);
        theta_p = acos(ph3);
      }
    }
  }

}

}  // namespace Aperture

#endif  // __RESONANT_DRAG_H_
