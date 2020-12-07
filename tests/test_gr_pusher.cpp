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

#include "core/typedefs_and_constants.h"
#include "systems/physics/geodesic_ks.hpp"
#include "utils/logger.h"
#include <iostream>

using namespace Aperture;

int N_iterate = 3;

void
prepare_halfstep(Scalar a, vec_t<Scalar, 3> &x, vec_t<Scalar, 3> &u, Scalar dt) {
  vec_t<Scalar, 3> x0 = x, x1 = x;
  for (int i = 0; i < N_iterate; i++) {
    x1 = x0 + geodesic_ks_x_rhs(a, x, u, true) * 0.5 * dt;
    x = x1;
  }
}

void
advance_photon(Scalar a, vec_t<Scalar, 3> &x, vec_t<Scalar, 3> &u, Scalar dt,
        bool show_output = false) {
  vec_t<Scalar, 3> x0 = x, x1 = x;
  vec_t<Scalar, 3> u0 = u, u1 = u;

  for (int i = 0; i < N_iterate; i++) {
    x1 = x0 + geodesic_ks_x_rhs(a, (x0 + x) * 0.5, u, true) * dt;
    u1 = u0 + geodesic_ks_u_rhs(a, x0, (u0 + u) * 0.5, true) * dt;
    x = x1;
    u = u1;
    // if (show_output)
    //   Logger::print_info("substep, x is ({}, {}, {}), u is ({}, {}, {})", x[0],
    //                      x[1], x[2], u[0], u[1], u[2]);
  }
}

void
advance_photon_symmetric(Scalar a, vec_t<Scalar, 3> &x, vec_t<Scalar, 3> &u, Scalar dt,
                  bool show_output = false) {
  vec_t<Scalar, 3> x0 = x, x1 = x;
  vec_t<Scalar, 3> u0 = u, u1 = u;

  for (int i = 0; i < N_iterate; i++) {
    x1 = x0 + geodesic_ks_x_rhs(a, (x0 + x) * 0.5, (u0 + u) * 0.5, true) * dt;
    u1 = u0 + geodesic_ks_u_rhs(a, (x0 + x) * 0.5, (u0 + u) * 0.5, true) * dt;
    x = x1;
    u = u1;
    // if (show_output)
    //   Logger::print_info("substep, x is ({}, {}, {}), u is ({}, {}, {})", x[0],
    //                      x[1], x[2], u[0], u[1], u[2]);
  }
}

void
photon_orbit(Scalar a, Scalar r, Scalar Phi, Scalar Q, Scalar dt) {
  Logger::print_info("Photon orbit, r = {}, Phi = {}, Q = {}", r, Phi, Q);

  vec_t<Scalar, 3> x, u;
  x[0] = r;
  x[1] = M_PI * 0.5;
  x[2] = 0.0;

  // Solve for the initial u_0
  Scalar gu00 = -1.0 - 2.0 * (a * a + r * r) / r / (a * a + r * (r - 2.0));
  Scalar gu03 = -2.0 * a / r / (a * a + r * (r - 2.0));
  Scalar gu22 = 1.0 / r / r;
  Scalar gu33 = (r - 2.0) / r / (a * a + r * (r - 2.0));
  Scalar u_0 = -math::sqrt((2.0 * gu03 * Phi - gu00 - gu33 * Phi * Phi) / gu22 /
                           square(Q));
  u[2] = -u_0 * Phi;
  u[1] = Q * square(u_0);
  u[0] = -2.0 * r * u_0 / (r * r + a * a - 2.0 * r) -
         a * u[2] / (r * r + a * a - 2.0 * r);

  Logger::print_info("u_0 is {}, {}", u_0,
                     u[0] * Metric_KS::beta1(a, x[0], x[1]) -
                         square(Metric_KS::alpha(a, x[0], x[1])) *
                             Metric_KS::u0(a, x[0], x[1], u, true));

  vec_t<Scalar, 3> utmp = u;
  // advance_photon_symmetric(a, x, utmp, dt*0.5);
  // prepare_halfstep(a, x, u, dt);

  int N = 100000;
  int out_step = 1000;
  Scalar max_costh = 0.0;
  for (int n = 0; n < N; n++) {
    advance_photon_symmetric(a, x, u, dt, n % out_step == 0);

    Scalar cth = math::abs(math::cos(x[1]));
    if (cth > max_costh) max_costh = cth;
    // if (n % out_step == 0) {
    if (n == N - 1) {
      Scalar u_0_now = u[0] * Metric_KS::beta1(a, x[0], x[1]) -
                       square(Metric_KS::alpha(a, x[0], x[1])) *
                           Metric_KS::u0(a, x[0], x[1], u, true);
      Logger::print_info("r is {}, max_costh is {}", x[0], max_costh);
      Logger::print_info("delta_r is {}, delta_u is {}",
                         math::abs(x[0] - r) / r,
                         math::abs(u_0_now - u_0) / u_0);
      // Logger::print_info("u_0 is {}", u[0] * Metric_KS::beta1(a, x[0], x[1])
      // -
      //                    square(Metric_KS::alpha(a, x[0], x[1])) *
      //                    Metric_KS::u0(a, x[0], x[1], u, true));
    }
  }
  Logger::print_info("");
}

int
main(int argc, char *argv[]) {
  // Initialize parameters
  Scalar a = 1.0;
  Scalar dt = 1e-3;

  ////// Photon unstable spherical orbits

  // Case A
  photon_orbit(a, 1.8, 1.36, 12.8304, dt);
  // Case B
  photon_orbit(a, 2.0, 1.0, 16.0, dt);
  // Case C
  photon_orbit(a, 1.0 + math::sqrt(2.0), 0.0, 22.3137, dt);
  // Case D
  photon_orbit(a, 1.0 + math::sqrt(3.0), -1.0, 25.8564, dt);
  // Case E
  photon_orbit(a, 3.0, -2.0, 27.0, dt);
  // Case F
  photon_orbit(a, 1.0 + 2.0 * math::sqrt(2.0), -6.0, 9.6274, dt);

  ////// Massive particles in a magnetic field

  return 0;
}
