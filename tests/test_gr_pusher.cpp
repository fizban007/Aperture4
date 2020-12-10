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
#include "systems/physics/metric_boyer_lindquist.hpp"
#include "systems/physics/metric_kerr_schild.hpp"
#include "systems/physics/wald_solution.hpp"
#include "utils/logger.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace Aperture;

int N_iterate = 3;
Scalar eom = 1.0;

void
prepare_halfstep(Scalar a, vec_t<Scalar, 3> &x, vec_t<Scalar, 3> &u,
                 Scalar dt) {
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
    //   Logger::print_info("substep, x is ({}, {}, {}), u is ({}, {}, {})",
    //   x[0],
    //                      x[1], x[2], u[0], u[1], u[2]);
  }
}

void
advance_photon_symmetric(Scalar a, vec_t<Scalar, 3> &x, vec_t<Scalar, 3> &u,
                         Scalar dt, bool show_output = false) {
  vec_t<Scalar, 3> x0 = x, x1 = x;
  vec_t<Scalar, 3> u0 = u, u1 = u;

  for (int i = 0; i < N_iterate; i++) {
    x1 = x0 + geodesic_ks_x_rhs(a, (x0 + x) * 0.5, (u0 + u) * 0.5, true) * dt;
    u1 = u0 + geodesic_ks_u_rhs(a, (x0 + x) * 0.5, (u0 + u) * 0.5, true) * dt;
    x = x1;
    u = u1;
    // if (show_output)
    //   Logger::print_info("substep, x is ({}, {}, {}), u is ({}, {}, {})",
    //   x[0],
    //                      x[1], x[2], u[0], u[1], u[2]);
  }
}

vec_t<Scalar, 3>
lorentz_ks_u_rhs(Scalar a, const vec_t<Scalar, 3> &x, const vec_t<Scalar, 3> &u,
                 const vec_t<Scalar, 3> &B, const vec_t<Scalar, 3> &D,
                 Scalar e_over_m) {
  vec_t<Scalar, 3> result;

  Scalar sth = math::sin(x[1]);
  Scalar cth = math::cos(x[1]);
  Scalar u0 = Metric_KS::u0(a, x[0], sth, cth, u, false);
  Scalar alpha = Metric_KS::alpha(a, x[0], sth, cth);
  Scalar sqrt_g = Metric_KS::sqrt_gamma(a, x[0], sth, cth);
  Scalar g_13 = Metric_KS::g_13(a, x[0], sth, cth);

  result[0] = alpha * e_over_m *
              (Metric_KS::g_11(a, x[0], sth, cth) * D[0] + g_13 * D[2]);
  result[1] = alpha * e_over_m * Metric_KS::g_22(a, x[0], sth, cth) * D[1];
  result[2] = alpha * e_over_m *
              (Metric_KS::g_33(a, x[0], sth, cth) * D[2] + g_13 * D[0]);

  vec_t<Scalar, 3> u_upper;
  Scalar gu13 = Metric_KS::gu13(a, x[0], sth, cth);
  u_upper[0] = Metric_KS::gu11(a, x[0], sth, cth) * u[0] + gu13 * u[2];
  u_upper[1] = Metric_KS::gu22(a, x[0], sth, cth) * u[1];
  u_upper[2] = Metric_KS::gu33(a, x[0], sth, cth) * u[2] + gu13 * u[0];

  result += sqrt_g * e_over_m * cross(u_upper, B) / u0;

  return result;
}

void
advance_ptc(Scalar a, Scalar dt, Scalar Bp, vec_t<Scalar, 3> &x,
            vec_t<Scalar, 3> &u, bool show_output = false) {
  vec_t<Scalar, 3> x0 = x, x1 = x;
  vec_t<Scalar, 3> u0 = u, u1 = u;
  vec_t<Scalar, 3> D, B;

  for (int i = 0; i < N_iterate; i++) {
    auto x_tmp = (x0 + x) * 0.5;
    auto u_tmp = (u0 + u) * 0.5;
    auto x_DB = x_tmp;
    B[0] = gr_wald_solution_B(a, x_DB[0], x_DB[1], Bp, 0);
    B[1] = gr_wald_solution_B(a, x_DB[0], x_DB[1], Bp, 1);
    B[2] = gr_wald_solution_B(a, x_DB[0], x_DB[1], Bp, 2);
    D[0] = gr_wald_solution_D(a, x_DB[0], x_DB[1], Bp, 0);
    D[1] = gr_wald_solution_D(a, x_DB[0], x_DB[1], Bp, 1);
    D[2] = gr_wald_solution_D(a, x_DB[0], x_DB[1], Bp, 2);

    x1 = x0 + geodesic_ks_x_rhs(a, x_tmp, u_tmp, false) * dt;
    u1 = u0 + geodesic_ks_u_rhs(a, x_tmp, u_tmp, false) * dt +
         lorentz_ks_u_rhs(a, x_DB, u_tmp, B, D, eom) * dt;
    x = x1;
    u = u1;
    if (show_output) {
      Logger::print_info("substep, x is ({}, {}, {}), u is ({}, {}, {})", x[0],
                         x[1], x[2], u[0], u[1], u[2]);
    }
  }
}

void
photon_orbit(Scalar a, Scalar r, Scalar Phi, Scalar Q, Scalar dt, const std::string& name = "") {
  Logger::print_info("Photon orbit, r = {}, Phi = {}, Q = {}", r, Phi, Q);

  vec_t<Scalar, 3> x, u;
  x[0] = r;
  x[1] = M_PI * 0.5;
  x[2] = 0.0;

  // Solve for the initial u_0
  Scalar sth = math::sin(x[1]);
  Scalar cth = math::cos(x[1]);
  Scalar gu00 = Metric_BL::gu00(a, r, sth, cth);
  Scalar gu03 = Metric_BL::gu03(a, r, sth, cth);
  Scalar gu22 = Metric_BL::gu22(a, r, sth, cth);
  Scalar gu33 = Metric_BL::gu33(a, r, sth, cth);
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

  int N = 200000;
  int out_step = 10;
  Scalar max_costh = 0.0;
  std::ofstream outfile(std::string("photon_orbit_") + name + ".json", std::ofstream::out | std::ofstream::trunc);
  outfile << "{\"photon_orbit_" << name << "\": [";
  for (int n = 0; n <= N; n++) {
    advance_photon_symmetric(a, x, u, dt);

    Scalar cth = math::abs(math::cos(x[1]));
    if (cth > max_costh) max_costh = cth;
    if (n % out_step == 0 && name != "") {
      sth = math::sin(x[1]);
      cth = math::cos(x[1]);
      Scalar sph = math::sin(x[2]);
      Scalar cph = math::cos(x[2]);
      outfile << x[0] * sth * cph << ", " << x[0] * sth * sph << ", " << x[0] * cth;
      if (n != N) {
        outfile << "," << std::endl;
      }
    }
    if (n == N - 1) {
      Scalar u_0_now = u[0] * Metric_KS::beta1(a, x[0], x[1]) -
                       square(Metric_KS::alpha(a, x[0], x[1])) *
                           Metric_KS::u0(a, x[0], x[1], u, true);
      Logger::print_info("r is {}, max_costh is {}, u^0 is {}", x[0], max_costh,
                         Metric_KS::u0(a, x[0], x[1], u, true));
      Logger::print_info("delta_r is {}, delta_u is {}",
                         math::abs(x[0] - r) / r,
                         math::abs(u_0_now - u_0) / u_0);
      // Logger::print_info("u_0 is {}", u[0] * Metric_KS::beta1(a, x[0], x[1])
      // -
      //                    square(Metric_KS::alpha(a, x[0], x[1])) *
      //                    Metric_KS::u0(a, x[0], x[1], u, true));
    }
  }
  outfile << "]}";
  outfile.close();
  Logger::print_info("");
}

void
ptc_orbit(Scalar a, Scalar r, Scalar th, Scalar Bz, Scalar uth, Scalar uph,
          Scalar dt, const std::string& name = "") {
  Logger::print_info("Ptc orbit, r = {}, th = {}, Bz = {}, uth = {}, uph = {}",
                     r, th, Bz, uth, uph);

  vec_t<Scalar, 3> x, u;
  x[0] = r;
  x[1] = th;
  // x[2] = a * r / (r * r + a * a - 2.0 * r);
  x[2] = 0.0;

  Scalar sth = math::sin(th);
  Scalar cth = math::cos(th);

  Scalar gu00 = Metric_BL::gu00(a, r, sth, cth);
  Scalar gu03 = Metric_BL::gu03(a, r, sth, cth);
  Scalar u_0_m =
      (-gu03 * uph -
       math::sqrt(square(gu03 * uph) -
                  gu00 * (Metric_BL::gu33(a, r, sth, cth) * uph * uph +
                          Metric_BL::gu22(a, r, sth, cth) * uth * uth + 1.0))) /
      gu00;
  Scalar u_0_p =
      (-gu03 * uph +
       math::sqrt(square(gu03 * uph) -
                  gu00 * (Metric_BL::gu33(a, r, sth, cth) * uph * uph +
                          Metric_BL::gu22(a, r, sth, cth) * uth * uth + 1.0))) /
      gu00;

  Logger::print_info("u_0 is {} and {}", u_0_m, u_0_p);

  Scalar u_0 = u_0_p;  // Choose the negative root
  Scalar E = -u_0 - eom * Bz * wald_ks_A0(a, r, sth, cth);
  Scalar L = uph + eom * Bz * wald_ks_Aphi(a, r, sth, cth);
  Logger::print_info("u_0 is {}, E is {}, L is {}", u_0, E, L);

  u[0] = -2.0 * r * u_0 / (r * r + a * a - 2.0 * r) -
         a * uph / (r * r + a * a - 2.0 * r);
  u[1] = uth;
  u[2] = uph;

  Scalar u_0_alt = u[0] * Metric_KS::beta1(a, x[0], x[1]) -
                   square(Metric_KS::alpha(a, x[0], x[1])) *
                       Metric_KS::u0(a, x[0], x[1], u, false);
  Logger::print_info("u_0_alt is {}, E_alt is {}", u_0_alt,
                     -u_0_alt - eom * Bz * wald_ks_A0(a, r, sth, cth));
  int N = int(1000.0 / dt) + 1;
  int out_step = int(floor(0.2 / dt)) + 1;
  Logger::print_info("outstep is {}", out_step);
  Scalar max_costh = 0.0;
  std::ofstream outfile(std::string("ptc_orbit_") + name + ".json", std::ofstream::out | std::ofstream::trunc);
  outfile << "{\"ptc_orbit_" << name << "\": [";
  for (int n = 0; n <= N; n++) {
    advance_ptc(a, dt, Bz, x, u);

    if (n % out_step == 0) {
      sth = math::sin(x[1]);
      cth = math::cos(x[1]);
      Scalar sph = math::sin(x[2]);
      Scalar cph = math::cos(x[2]);
      // Logger::print_info("outstep {}", n);
      outfile << x[0] * sth * cph << ", " << x[0] * sth * sph << ", " << x[0] * cth;
      if (n != N) {
        outfile << "," << std::endl;
      }
    }
    if (n == N - 1) {
      Scalar u_0_now = u[0] * Metric_KS::beta1(a, x[0], x[1]) -
                       square(Metric_KS::alpha(a, x[0], x[1])) *
                           Metric_KS::u0(a, x[0], x[1], u, false);
      sth = math::sin(x[1]);
      cth = math::cos(x[1]);
      Scalar E_now = -u_0_now - eom * Bz * wald_ks_A0(a, x[0], sth, cth);
      Scalar L_now = u[2] + eom * Bz * wald_ks_Aphi(a, x[0], sth, cth);
      // Logger::print_info("r is {}, max_costh is {}", x[0], max_costh);
      Logger::print_info("delta_L is {}, delta_E is {}",
                         math::abs(L_now - L) / L, math::abs(E_now - E) / E);
      Logger::print_info("E is {}, L is {}", E_now, L_now);
      // Logger::print_info("u_0 is {}", u[0] * Metric_KS::beta1(a, x[0], x[1])
      // -
      //                    square(Metric_KS::alpha(a, x[0], x[1])) *
      //                    Metric_KS::u0(a, x[0], x[1], u, true));
    }
  }
  outfile << "]}";
  outfile.close();
  Logger::print_info("");
}

int
main(int argc, char *argv[]) {
  // Initialize parameters
  Scalar a = 1.0;
  Scalar dt = 1e-2;

  ////// Photon unstable spherical orbits

  // Case A
  photon_orbit(a, 1.8, 1.36, 12.8304, dt, "caseA");
  // Case B
  photon_orbit(a, 2.0, 1.0, 16.0, dt, "caseB");
  // Case C
  photon_orbit(a, 1.0 + math::sqrt(2.0), 0.0, 22.3137, dt, "caseC");
  // Case D
  photon_orbit(a, 1.0 + math::sqrt(3.0), -1.0, 25.8564, dt, "caseD");
  // Case E
  photon_orbit(a, 3.0, -2.0, 27.0, dt, "caseE");
  // Case F
  photon_orbit(a, 1.0 + 2.0 * math::sqrt(2.0), -6.0, 9.6274, dt, "caseF");

  dt = 0.001;
  ////// Massive particles in a magnetic field
  ptc_orbit(0.0, 4.0, M_PI * 0.5, 0.2, 0.0, 2.9, dt, "RSA1");
  ptc_orbit(0.0, 9.5, 1.6, 0.2, 0.0, -1.024, dt, "RSA3");
  ptc_orbit(0.0, 8.5, 1.06, -2.0, 0.0, 122.983, dt, "RSA5");
  ptc_orbit(0.7, 4.2, M_PI * 0.5 - 0.1, 2.0, 0.0, -1.93, dt, "RKA2");
  ptc_orbit(0.9, 4.0, M_PI * 0.5 - 0.2, 2.0, 0.0, 1.565, dt, "RKA3");
  ptc_orbit(0.9, 3.0, M_PI * 0.5, 1.0, 0.497, 0.365, dt, "RKA8");
  ptc_orbit(0.9, 3.0, M_PI * 0.5, 1000.0, 0.497, 0.365, dt, "RKA8largeB");

  return 0;
}
