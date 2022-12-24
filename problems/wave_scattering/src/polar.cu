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

#include "data/fields.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/compute_lorentz_factor.h"
#include "systems/data_exporter.h"
#include "systems/field_solver_polar.h"
// #include "systems/legacy/ptc_updater_sph.h"
#include "systems/gather_momentum_space.h"
#include "systems/grid_polar.hpp"
#include "systems/policies/coord_policy_polar.hpp"
#include "systems/ptc_injector_cuda.hpp"
#include "systems/ptc_updater_base.h"
#include "utils/kernel_helper.hpp"
// #include "systems/ptc_injector.h"
#include <iostream>

namespace Aperture {

template <typename Conf>
class boundary_condition : public system_t {
 protected:
  const grid_curv_t<Conf> &m_grid;
  const domain_comm<Conf> *m_comm;
  double m_E0 = 1.0;
  double m_omega_0 = 0.0;
  double m_omega_t = 0.0;
  double m_Bp = 1.0;
  int num_wavelengths = 4;

  vector_field<Conf> *E, *B, *E0, *B0;

 public:
  static std::string name() { return "boundary_condition"; }

  boundary_condition(const grid_curv_t<Conf> &grid, const domain_comm<Conf>* comm)
   : m_grid(grid), m_comm(comm) {}

  void init() override {
    sim_env().get_data("Edelta", &E);
    sim_env().get_data("E0", &E0);
    sim_env().get_data("Bdelta", &B);
    sim_env().get_data("B0", &B0);

    sim_env().params().get_value("E0", m_E0);
    sim_env().params().get_value("omega_t", m_omega_t);
    sim_env().params().get_value("Bp", m_Bp);
    sim_env().params().get_value("num_lambda", num_wavelengths);
  }

  void update(double dt, uint32_t step) override {
    auto ext = m_grid.extent();
    typedef typename Conf::idx_t idx_t;
    typedef typename Conf::value_t value_t;

    value_t time = sim_env().get_time();
    value_t Bp = m_Bp;
    value_t omega;
    // if (m_omega_t * time < 5000.0)
    value_t phase = time * m_omega_t;
    value_t length = num_wavelengths;
    value_t smooth_width = 0.2;
    if (phase < length) {
      value_t prof = 1.0;
      value_t arg = 2.0 * phase / length;
      // omega = 2.0 * M_PI * m_omega_t * m_E0 * sin(2.0 * M_PI * phase);
      // if (arg < smooth_width) {
      //   prof = square(math::sin(arg * M_PI / (smooth_width * 2.0f)));
      // } else if (arg > 1.0 - smooth_width) {
      //   prof = square(math::sin((arg - 1.0f + smooth_width) * M_PI /
      //                               (smooth_width * 2.0f) +
      //                           0.5f * M_PI));
      // }
      prof = 0.5f * (1.0f - tanh((math::abs(arg - 1.0f) - 0.6f) / 0.1f)) - 3.35e-4;
      omega = prof * m_E0 * sin(2.0 * M_PI * phase);
    } else {
      omega = 0.0;
    }
    Logger::print_debug("time is {}, Omega is {}", time, omega);

    if (m_comm == nullptr ||
        m_comm->domain_info().neighbor_left[0] == MPI_PROC_NULL) {
      kernel_launch(
          [ext, time, omega, Bp] __device__(auto e, auto b, auto e0, auto b0) {
            auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
            for (auto n1 : grid_stride_range(0, grid.dims[1])) {
              value_t theta =
                  grid_polar_t<Conf>::theta(grid.template coord<1>(n1, false));
              value_t theta_s =
                  grid_polar_t<Conf>::theta(grid.template coord<1>(n1, true));

              // For quantities that are not continuous across the surface
              for (int n0 = grid.guard[0]; n0 < grid.guard[0] + 1; n0++) {
                auto idx = idx_t(index_t<2>(n0, n1), ext);
                // value_t r = grid_polar_t<Conf>::radius(grid.template coord<0>(n0, false));
                e[0][idx] = 0.0;
                // b[1][idx] = Bp - omega * Bp;
                b[1][idx] = omega * Bp;
                b[2][idx] = 0.0;
              }
              // For quantities that are continuous across the surface
              for (int n0 = grid.guard[0]; n0 < grid.guard[0] + 1; n0++) {
                auto idx = idx_t(index_t<2>(n0, n1), ext);
                value_t r =
                    grid_polar_t<Conf>::radius(grid.template coord<0>(n0, false));
                value_t r_s =
                    grid_polar_t<Conf>::radius(grid.template coord<0>(n0, true));
                b[0][idx] = 0.0;
                e[1][idx] = 0.0;
                // if (theta_s > 0.7 && theta_s < 1.2)
                // e[2][idx] = -omega * sin(theta_s) * r_s * b0[0][idx];
                e[2][idx] = -omega * Bp;
                // else
                //   e[2][idx] = 0.0;
                // e[2][idx] = 0.0;
              }
            }
          },
          E->get_ptrs(), B->get_ptrs(), E0->get_ptrs(), B0->get_ptrs());
      CudaSafeCall(cudaDeviceSynchronize());
    }
  }
};

}  // namespace Aperture

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  using value_t = typename Conf::value_t;
  auto &env = sim_environment::instance(&argc, &argv);

  domain_comm<Conf> comm;
  grid_polar_t<Conf> grid(comm);
  grid.init();

  auto pusher = env.register_system<
      ptc_updater_new<Conf, exec_policy_cuda, coord_policy_polar>>(grid, comm);
  auto lorentz = env.register_system<compute_lorentz_factor_cu<Conf>>(grid);
  auto momentum =
      env.register_system<gather_momentum_space<Conf, exec_policy_cuda>>(grid);
  auto solver = env.register_system<field_solver_polar_cu<Conf>>(grid, &comm);
  auto bc = env.register_system<boundary_condition<Conf>>(grid, &comm);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_cuda>>(grid, &comm);

  env.init();

  // Initial conditions
  vector_field<Conf> *B0, *Bdelta, *Edelta;
  env.get_data("B0", &B0);
  double Bp = sim_env().params().get_as<double>("Bp", 100.0);
  vec_t<value_t, Conf::dim> lowers;
  sim_env().params().get_vec_t("lower", lowers);
  B0->set_values(1, [Bp, lowers](auto x, auto y, auto z) {
    auto r = grid_polar_t<Conf>::radius(x);
    auto r0 = grid_polar_t<Conf>::radius(lowers[0]);
    return Bp * lowers[0] / r;
  });

  particle_data_t *ptc;
  env.get_data("particles", &ptc);
  // ptc->append_dev({0.5f, 0.5f, 0.0f}, {0.0f, 10.0f, 0.0f}, 100, 1.0);

  auto injector =
      sim_env().register_system<ptc_injector<Conf, exec_policy_cuda>>(grid);
  injector->init();

  int n_upstream = sim_env().params().get_as<int64_t>("upstream_n", 5);
  value_t kT_upstream = sim_env().params().get_as<double>("upstream_kT", 0.01);
  value_t rho_bg = sim_env().params().get_as<double>("rho_bg", 100.0);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);
  // Background (upstream) particles
  injector->inject(
      [] __device__(auto &pos, auto &grid, auto &ext) { return true; },
      [n_upstream] __device__(auto &pos, auto &grid, auto &ext) {
        return 2 * n_upstream;
      },
      [kT_upstream] __device__(auto &pos, auto &grid, auto &ext, rng_t &rng,
                               PtcType type) {
        return rng.maxwell_juttner_3d(kT_upstream);
      },
      // [n_upstream] __device__(auto &pos, auto &grid, auto &ext) {
      [rho_bg, n_upstream, q_e, lowers] __device__(auto &x_global) {
        auto r0 = grid_polar_t<Conf>::radius(lowers[0]);
        // value_t rho = rho_bg / square(grid_polar_t<Conf>::radius(x_global[0]));
        // value_t rho = rho_bg / grid_polar_t<Conf>::radius(x_global[0]);
        value_t rho = rho_bg * r0;
        return rho / q_e / n_upstream;
      });

  env.run();
  return 0;
}
