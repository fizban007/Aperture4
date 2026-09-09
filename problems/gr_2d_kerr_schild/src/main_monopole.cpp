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
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_gr_ks_mod.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/compute_moments_gr_ks.h"
#include "systems/grid_ks.h"
#include "systems/radiative_transfer_impl.hpp"
#include "systems/radiation/default_radiation_scheme_gr.hpp"
#include "systems/policies/coord_policy_gr_ks_sph.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "systems/ptc_updater.h"
#include "utils/util_functions.h"

using namespace std;

namespace Aperture {

template <typename Conf>
void initial_vacuum_wald(vector_field<Conf> &B0, vector_field<Conf> &D0,
                         const grid_ks_t<Conf> &grid);
template <typename Conf>
void initial_nonrotating_vacuum_wald(vector_field<Conf> &B0,
                                     vector_field<Conf> &D0,
                                     const grid_ks_t<Conf> &grid);

template <typename Conf>
void initial_vacuum_monopole(vector_field<Conf> &B, vector_field<Conf> &D,
                             const grid_ks_t<Conf> &grid);

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
      ptc_updater<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph>>(grid, &comm);
  auto moments = env.register_system<compute_moments_gr_ks<Conf, exec_policy_dynamic>>(grid);
  // The density floor injector is a Bondi-accretion refill device
  // (Figueiredo+ 2026). Enable it only if this monopole run is meant to be
  // continuously refilled; note that density_floor_injector.cpp currently
  // reads "bp"/"Nr"/"size"/"lower" in a way that silently falls back to
  // defaults, so its target density and r_pml need fixing first.
  // auto floor_injector =
  //     env.register_system<bh_density_floor_injector<Conf>>(grid);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  auto radiation = env.register_system<
    radiative_transfer<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph,
                       default_radiation_scheme_gr>>(grid, &comm);
  auto solver = env.register_system<
      field_solver_mod<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph>>(grid, &comm);
  auto exporter =
      env.register_system<data_exporter<Conf, exec_policy_dynamic>>(grid, &comm);

  env.init();

  int ppc = 20;
  env.params().get_value("ppc", ppc);

  int damping_length = 64;
  env.params().get_value("damping_length", damping_length);

  // N, size and lower are per-axis arrays in the config; read the radial
  // (index 0) component. Reading them as scalars silently falls back to the
  // hardcoded default and corrupts the derived geometry below.
  int ncells[Conf::dim];
  env.params().get_array("N", ncells);
  int Nr = ncells[0];

  double size_arr[Conf::dim];
  env.params().get_array("size", size_arr);
  double size_log_r = size_arr[0];

  double lower_arr[Conf::dim];
  env.params().get_array("lower", lower_arr);
  double log_r_min = lower_arr[0];

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
  double r_pml = math::exp(log_r_pml);
  double r_H = 1.0 + math::sqrt(1.0 - spin * spin);
  double init_num_dens = Bp * Bp / sigma;

  // Prepare initial field
  vector_field<Conf> *B, *D, *B0, *D0;
  env.get_data("B0", &B0);
  env.get_data("E0", &D0);
  env.get_data("Bdelta", &B);
  env.get_data("Edelta", &D);

  initial_vacuum_monopole(*B0, *D0, grid);
  // Alternative initial fields:
  // initial_vacuum_wald(*B0, *D0, grid);
  // initial_nonrotating_vacuum_wald(*B0, *D0, grid);

  ptc_injector_dynamic<Conf> ptc_inj(grid);
  ptc_inj.inject_pairs(
      // First function is the injection criterion for each cell. pos is an
      // index_t<Dim> object marking the cell in the grid. Returns true for
      // cells that inject and false for cells that do nothing. Skip the
      // horizon interior and the outer damping layer.
      [r_H, r_pml] LAMBDA(auto &pos, auto &grid, auto &ext) {
        auto r = grid_ks_t<Conf>::radius(grid.template coord<0>(pos[0], false));
        return (r > r_H && r < r_pml);
      },
      // Second function returns the number of particles injected in each cell.
      // This includes all species
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) { return ppc; },
      // Third function is the momentum distribution of the injected particles.
      // Returns a vec_t<value_t, 3> object encoding the 3D momentum of this
      // particular particle
      [] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        return vec_t<value_t, 3>(0.0, 0.0, 0.0);
      },
      // Fourth function is the particle weight, which can depend on the global
      // coordinate. r * sqrt_gamma is the proper cell volume per unit
      // d(log r) d(theta), so dividing the target density by it (and by ppc)
      // gives the per-particle weight.
      //
      // The (r_H / r)^2 factor sets n ~ r^-2, normalized so that
      // n(r_H) = init_num_dens = Bp^2 / sigma. Because the grid is logarithmic
      // in r the cell width grows as dr ~ r, and n ~ r^-2 makes the skin depth
      // d_e ~ n^(-1/2) ~ r grow at the same rate, holding d_e / dr fixed
      // across the box. This sits between uniform density (sigma ~ r^-4) and
      // uniform magnetization (n ~ r^-4).
      [ppc, spin, init_num_dens, r_H] LAMBDA(auto &x_global, PtcType type) {
        value_t r = grid_ks_t<Conf>::radius(x_global[0]);
        value_t th = grid_ks_t<Conf>::theta(x_global[1]);
        value_t sqrt_gamma = Metric_KS::sqrt_gamma(spin, r, th);
        value_t rat = r_H / r;
        value_t w = (init_num_dens * rat * rat * r * sqrt_gamma) / ppc;
        return w;
      });

  env.run();

  return 0;
}
