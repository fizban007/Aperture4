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

#include "core/cuda_control.h"
#include "core/enum_types.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "injector.h"
#include "systems/compute_moments_gr_ks.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_gr_ks_mod.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/grid_ks.h"
#include "systems/policies/coord_policy_gr_ks_sph.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "systems/ptc_updater.h"
#include "systems/radiation/default_radiation_scheme_gr.hpp"
#include "systems/radiation/gr_ks_ic_radiation_scheme_fido.hpp"
#include "systems/radiative_transfer_impl.hpp"
#include "utils/hdf_wrapper.h"
#include "utils/interpolation.hpp"
#include "utils/util_functions.h"
// #define BOOST_MATH_DISABLE_FLOAT128
// #include "boost/math/special_functions/gamma.hpp"

using namespace std;

namespace Aperture {

template <typename Conf>
void initial_vacuum_wald(vector_field<Conf> &B0, vector_field<Conf> &D0,
                         const grid_ks_t<Conf> &grid);

template <typename Conf>
void initial_vacuum_wald_limited(vector_field<Conf> &B0, vector_field<Conf> &D0,
                                 const grid_ks_t<Conf> &grid, Scalar r_in,
                                 Scalar r_out);
}  // namespace Aperture

using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  using value_t = Conf::value_t;

  auto &env = sim_environment::instance(&argc, &argv, true);

  // env.params().add("log_level", (int64_t)LogLevel::debug);

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_ks_t<Conf> grid(comm);

  auto pusher = env.register_system<
      ptc_updater<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph>>(grid,
                                                                      &comm);
  auto moments =
      env.register_system<compute_moments_gr_ks<Conf, exec_policy_dynamic>>(
          grid);
  // auto injector = env.register_system<bh_injector<Conf>>(grid);
  auto floor_injector =
      env.register_system<bh_density_floor_injector<Conf>>(grid);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  auto radiation = env.register_system<
      radiative_transfer<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph,
                         default_radiation_scheme_gr>>(grid, &comm);
  //  gr_ks_ic_radiation_scheme_fido>>(grid, &comm);
  auto solver = env.register_system<
      field_solver_mod<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph>>(
      grid, &comm);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(
      grid, &comm);

  env.init();

  int ppc = 20;
  env.params().get_value("ppc", ppc);

  int damping_length = 64;
  env.params().get_value("damping_length", damping_length);

  int Nr = 1024;
  env.params().get_value("Nr", Nr);

  double size_log_r = 3.00;
  env.params().get_value("size", size_log_r);

  double log_r_min = 0.588;
  env.params().get_value("lower", log_r_min);

  double spin = 0.0000001;
  env.params().get_value("bh_spin", spin);

  double Bp = 1.0;
  env.params().get_value("Bp", Bp);

  double sigma = 0.1;
  env.params().get_value("sigma", sigma);

  // derived parameters
  double log_r_max = log_r_min + size_log_r;
  double d_log_r = size_log_r / Nr;
  double log_r_pml = log_r_max - damping_length * d_log_r;
  double r_min = math::exp(log_r_min);
  double r_max = math::exp(log_r_max);
  double r_pml = math::exp(log_r_pml);
  double r_H = 1.0 + math::sqrt(1.0 - spin * spin);
  double init_num_dens = Bp * Bp / sigma;

  // The temperature is technically constrained by the setup, but allow the user
  // to specify it if they want to.
  double kT = 2.0 / r_pml;
  env.params().get_value("kT", kT);

  // Prepare initial field
  vector_field<Conf> *B, *D, *B0, *D0;
  env.get_data("B0", &B0);
  env.get_data("E0", &D0);
  env.get_data("Bdelta", &B);
  env.get_data("Edelta", &D);

  // initial_vacuum_wald_limited(*B0, *D0, grid, r_lower*0.95, 1000.0);
  initial_vacuum_wald(*B0, *D0, grid);

  ptc_injector_dynamic<Conf> ptc_inj(grid);
  ptc_inj.inject_pairs(
      // First function is the injection criterion for each cell. pos is an
      // index_t<Dim> object marking the cell in the grid. Returns true for
      // cells that inject and false for cells that do nothing.
      [r_H, r_pml] LAMBDA(auto &pos, auto &grid, auto &ext) {
        auto r = grid_ks_t<Conf>::radius(grid.template coord<0>(pos[0], false));
        // auto th = grid_ks_t<Conf>::theta(grid.template coord<1>(pos[1],
        // false));
        return (r > r_H && r < r_pml);
      },
      // Second function returns the number of particles injected in each cell.
      // This includes all species
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) { return ppc; },
      // Third function is the momentum distribution of the injected particles.
      // Returns a vec_t<value_t, 3> object encoding the 3D momentum of this
      // particular particle
      [spin, kT] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        value_t r = grid_ks_t<Conf>::radius(x_global[0]);
        value_t theta = grid_ks_t<Conf>::theta(x_global[1]);

        vec_t<value_t, 3> u_d = rng_maxwell_juttner_3d(state, kT);
        // Now transform this momentum from the local fluid frame to the global
        // coordinate. Use the tetrads given in Benjamin Crinquand's PhD thesis:
        // https://theses.hal.science/tel-03406333v1/file/Thesis_Benjamin_Crinquand_final.pdf
        // The relevant expressions are given on pages 164-165
        // Note also that in the locally flat fluid rest frame, u^i = u_i
        value_t h_11 = Metric_KS::g_11(r, theta, spin);
        value_t h_22 = Metric_KS::g_22(r, theta, spin);
        value_t h_33 = Metric_KS::g_33(r, theta, spin);
        value_t h_13 = Metric_KS::g_13(r, theta, spin);
        value_t scriptA = math::sqrt(h_33 / (h_11 * h_33 - h_13 * h_13));

        value_t u_3 = math::sqrt(h_33) * u_d[2];
        value_t u_2 = math::sqrt(h_22) * u_d[1];
        value_t u_1 = u_d[0] / scriptA + (h_13 / h_33) * u_3;

        // The following are from an early implementation where the plasma was
        // initialized cold. value_t uu0 = math::sqrt(-1.0 / Metric_KS::g_00(r,
        // theta, spin)); value_t u_1 = Metric_KS::g_01(r, theta, spin) * uu0;
        // value_t u_2 = 0.0;
        // value_t u_3 = Metric_KS::g_03(r, theta, spin) * uu0;

        return vec_t<Scalar, 3>{u_1, u_2, u_3};
      },
      // Fourth function is the particle weight, which can depend on the global
      // coordinate.
      [ppc, spin, init_num_dens] LAMBDA(auto &x_global, PtcType type) {
        value_t r = grid_ks_t<Conf>::radius(x_global[0]);
        value_t th = grid_ks_t<Conf>::theta(x_global[1]);
        value_t sqrt_gamma = Metric_KS::sqrt_gamma(spin, r, th);
        value_t w = (init_num_dens * r * sqrt_gamma) / ppc;
        return w;
      });

  env.run();

  return 0;
}