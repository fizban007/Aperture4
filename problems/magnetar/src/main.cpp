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

#include "core/math.hpp"
#include "cxxopts.hpp"//
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/boundary_condition.hpp"//
#include "systems/compute_moments.h"
#include "systems/data_exporter.h"
#include "systems/field_solver_sph.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/grid_sph.hpp"
#include "systems/policies/coord_policy_spherical.hpp"
#include "systems/policies/coord_policy_spherical_sync_cooling.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "systems/ptc_updater_impl.hpp"
// #include "systems/radiation/IC_radiation_scheme.hpp"
#include "systems/radiative_transfer_impl.hpp"
#include "systems/resonant_scattering_scheme.hpp"
#include "utils/hdf_wrapper.h"
#include "utils/logger.h"
#include "utils/vec.hpp"
#include <fstream>
#include <memory>
#include <vector>

using namespace std;
using namespace Aperture;

template <typename Conf>
class ptc_physics_policy_gravity_r {
 public:
  using value_t = typename Conf::value_t;

  void init() { sim_env().params().get_value("gravity", m_g); }

  template <typename PtcContext, typename IntT>
  HD_INLINE void operator()(const Grid<Conf::dim, value_t>& grid,
                            PtcContext& context,
                            const vec_t<IntT, Conf::dim>& pos,
                            value_t dt) const {
    auto r = grid_sph_t<Conf>::radius(grid.coord(0, pos[0], context.x[0]));
    context.p[0] -= m_g * dt;
    context.gamma = sqrt(1.0f + context.p.dot(context.p));
  }

 private:
  value_t m_g = 0.0f;
};

template class ptc_updater<Config<2>, exec_policy_dynamic,
                          //  coord_policy_spherical_sync_cooling,
                          coord_policy_spherical>;
                           //ptc_physics_policy_gravity_r>;

template class radiative_transfer<Config<2>, exec_policy_dynamic,
                                  coord_policy_spherical, resonant_scattering_scheme>;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  using value_t = typename Conf::value_t;
  auto &env = sim_environment::instance(&argc, &argv);

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_sph_t<Conf> grid(comm);
  // auto &grid = *(env.register_system<grid_t<Conf>>(comm));
  auto solver =
      env.register_system<field_solver<Conf, exec_policy_dynamic,
                                       coord_policy_spherical>>(
          grid, &comm);
  auto pusher =
      env.register_system<ptc_updater<Conf, exec_policy_dynamic,
                                      // coord_policy_spherical_sync_cooling,
                                      coord_policy_spherical>>(
                                      //ptc_physics_policy_gravity_r>>(
          // coord_policy_spherical>>(
          grid, &comm);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  auto rad = env.register_system<radiative_transfer<
      Conf, exec_policy_dynamic, coord_policy_spherical,
      resonant_scattering_scheme>>( grid, &comm);
  auto moments =
      env.register_system<compute_moments<Conf, exec_policy_dynamic>>(grid);
  // auto rad = env.register_system<radiative_transfer<
  //     Conf, exec_policy_dynamic, coord_policy_spherical,
  //     IC_radiation_scheme>>( grid, &comm);
  auto bc = env.register_system<boundary_condition<Conf, exec_policy_dynamic>>(
      grid, &comm);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(
      grid, &comm);

  env.init();

  vector_field<Conf> *B0;
  particles_t *ptc;
  env.get_data("B0", &B0);
  env.get_data("particles", &ptc);

  // Read parameters
  float Bp = 1.0e2;
  float twist_omega = 0.01;
  float qe = 1.0;
  int ppc = 20;
  env.params().get_value("Bp", Bp);
  env.params().get_value("ppc", ppc);
  env.params().get_value("twist_omega", twist_omega);
  env.params().get_value("qe", qe);
  float kT = 1.0e-3;
  float rho0 = 1.0e4;
  env.params().get_value("kT", kT);
  env.params().get_value("rho0", rho0);

  // Set dipole initial magnetic field
  B0->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    // return Bp / (r * r);
    return Bp * 2.0 * cos(theta) / cube(r);
  });
  B0->set_values(1, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    return Bp * sin(theta) / cube(r);
  });

  // Fill the magnetosphere with pairs
  ptc_injector_dynamic<Conf> injector(grid);
  injector.inject_pairs(
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true;},
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2 * ppc; },
      [kT] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        return rng_maxwell_juttner_3d<value_t>(state, kT);
        // return vec_t<value_t, 3>(0.0, 0.0, 0.0);
      },
      [rho0, qe, ppc] LAMBDA(auto &x_global, PtcType type) {
        auto &grid = static_cast<const grid_sph_t<Conf> &>(
            exec_policy_dynamic<Conf>::grid());
        auto r = grid.radius(x_global[0]);
        auto th = grid.theta(x_global[1]);
        // This naturally gives rho ~ 1/r^3 dependence
        return rho0 * math::sin(th) / qe / ppc;
      });
      // },
      // flag_or(PtcFlag::ignore_radiation));

  env.run();
  return 0;
}
