/*
 * Copyright (c) 2021 Alex Chen.
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

#include "core/math.hpp"
#include "cxxopts.hpp"
#include "utils/hdf_wrapper.h"
#include "utils/logger.h"
#include "utils/vec.hpp"
#include <fstream>
#include <memory>
#include <vector>

using namespace Aperture;

using vec3 = vec_t<double, 3>;

vec3
rhs_u(const vec3& u, double Bz, double cooling_coef, double dt) {
  vec3 result;
  double u_sqr = u.dot(u);
  double gamma = math::sqrt(1.0f + u_sqr);
  double betaxb_x = u[1] * Bz / gamma;
  double betaxb_y = -u[0] * Bz / gamma;
  double betaxb_sqr = betaxb_x * betaxb_x + betaxb_y * betaxb_y;

  // result[0] =
  //     betaxb_x + cooling_coef * (betaxb_y * Bz - gamma * u[0] * betaxb_sqr);
  // result[1] =
  //     betaxb_y + cooling_coef * (-betaxb_x * Bz - gamma * u[1] * betaxb_sqr);
  result[0] = betaxb_x + cooling_coef * (-gamma * u[0] * betaxb_sqr);
  result[1] = betaxb_y + cooling_coef * (-gamma * u[1] * betaxb_sqr);
  result[2] = cooling_coef * (-gamma * u[2] * betaxb_sqr);

  return result * dt;
}

vec3
rhs_x(const vec3& u, double dt) {
  double u_sqr = u.dot(u);
  double gamma = math::sqrt(1.0f + u_sqr);

  return u * (dt / gamma);
}

vec3
rhs_u(const vec3& E, const vec3& B, const vec3& u, double e_over_m,
      double cooling_coef, double dt) {
  vec3 result;
  double u_sqr = u.dot(u);
  double gamma = math::sqrt(1.0f + u_sqr);

  vec3 Epbetaxb = E + cross(u, B) / gamma;

  result =
      e_over_m * Epbetaxb +
      cooling_coef *
          (cross(Epbetaxb, B) + E * u.dot(E) / gamma -
           u * (gamma * (Epbetaxb.dot(Epbetaxb) - square(u.dot(E) / gamma))));
          // (u * (-gamma * (Epbetaxb.dot(Epbetaxb) - square(u.dot(E) / gamma))));

  return result * dt;
}

void
iterate(vec3& x, vec3& u, const vec3& E, const vec3& B,
        double e_over_m, double cooling_coef, double dt) {
  vec3 x0 = x, x1 = x;
  vec3 u0 = u, u1 = u;

  for (int i = 0; i < 4; i++) {
    x1 = x0 + rhs_x((u0 + u) * 0.5, dt);
    u1 = u0 + rhs_u(E, B, (u0 + u) * 0.5, e_over_m, cooling_coef, dt);
    x = x1;
    u = u1;
  }
}

void
iterate(vec3& x, vec3& u, double Bz,
        double cooling_coef, double dt) {
  vec3 x0 = x, x1 = x;
  vec3 u0 = u, u1 = u;

  for (int i = 0; i < 4; i++) {
    x1 = x0 + rhs_x((u0 + u) * 0.5, dt);
    u1 = u0 + rhs_u((u0 + u) * 0.5, Bz, cooling_coef, dt);
    x = x1;
    u = u1;
  }
}

int
main(int argc, char* argv[]) {
  vec3 x(0, 0, 0);
  vec3 u(10.0, 0, 10.0);
  vec3 E(0.0, 0.0, 0.0);

  // Parse options
  cxxopts::Options options("sync cooling test", "Aperture PIC code");
  options.add_options()("h,help", "Prints this help message.")(
      "g,gamma", "initial gamma",
      cxxopts::value<double>()->default_value("10.0"))(
      "p,pitch", "initial pitch angle",
      cxxopts::value<double>()->default_value("0.7854"))(
      "B", "Bz field strength.",
      cxxopts::value<double>()->default_value("100.0"))(
      "dt", "delta t", cxxopts::value<double>()->default_value("0.001"))(
      "c,cooling", "cooling coefficient",
      cxxopts::value<double>()->default_value("1.0"))(
      "s,steps", "number of steps", cxxopts::value<int>()->default_value("10"));

  // Parse options and store the results
  auto result = options.parse(argc, argv);
  double Bz = result["B"].as<double>();
  vec3 B(0.0, 0.0, Bz);
  double dt = result["dt"].as<double>();
  double coef = result["cooling"].as<double>();
  int steps = result["steps"].as<int>();
  double g = result["gamma"].as<double>();
  double pitch = result["pitch"].as<double>();
  u[2] = sqrt(g*g - 1.0) * cos(pitch);
  u[0] = sqrt(g*g - 1.0) * sin(pitch);

  Logger::print_info("Bz is {}, dt is {}, cooling is {}", Bz, dt, coef);

  auto outfile = hdf_create("output.h5");
  std::vector<double> alphas(steps / 10);
  std::vector<double> gammas(steps / 10);
  std::vector<double> betas(steps / 10);

  for (int n = 0; n < steps; n++) {
    // iterate(x, u, Bz, coef, dt);
    iterate(x, u, E, B, 1.0, coef, dt);
    if (n % 10 == 0) {
      Logger::print_info("u is ({}, {}, {})", u[0], u[1], u[2]);
      double mu = u[2] / math::sqrt(u.dot(u));
      double gamma = math::sqrt(1.0 + u.dot(u));
      Logger::print_info("pitch angle is {}, gamma is {}, beta_para is {}",
                         std::acos(mu), gamma, u[2] / gamma);
      alphas[n / 10] = std::acos(mu);
      gammas[n / 10] = gamma;
      betas[n / 10] = u[2] / gamma;
    }
  }
  outfile.write(Bz, "Bz");
  outfile.write(dt, "dt");
  outfile.write(coef, "coef");
  outfile.write(alphas.data(), alphas.size(), "alphas");
  outfile.write(gammas.data(), gammas.size(), "gammas");
  outfile.write(betas.data(), betas.size(), "betas");
  outfile.close();

  return 0;
}
