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
#include "systems/field_solver_gr_ks.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/compute_moments.h"
#include "systems/grid_ks.h"
#include "systems/radiative_transfer_impl.hpp"
#include "systems/radiation/gr_ks_ic_radiation_scheme_fido.hpp"
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

// template class radiative_transfer<Config<2>, exec_policy_gpu,
//                                   coord_policy_gr_ks_sph,
//                                   default_radiation_scheme_gr>;

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
  auto moments = env.register_system<compute_moments<Conf, exec_policy_dynamic>>(grid);
  // auto injector = env.register_system<bh_injector<Conf>>(grid);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  auto radiation = env.register_system<
    radiative_transfer<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph,
                      //  default_radiation_scheme_gr>>(grid, &comm);
                       gr_ks_ic_radiation_scheme_fido>>(grid, &comm);
  auto solver = env.register_system<
      field_solver<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph>>(grid, &comm);
  auto exporter =
      env.register_system<data_exporter<Conf, exec_policy_dynamic>>(grid, &comm);

  env.init();

  // Prepare initial condition here
  vector_field<Conf> *B, *D, *B0, *D0;
  // particle_data_t *ptc;
  env.get_data("B0", &B0);
  env.get_data("E0", &D0);
  env.get_data("Bdelta", &B);
  env.get_data("Edelta", &D);
  // env.get_data("particles", &ptc);

  initial_vacuum_monopole(*B0, *D0, grid);
  // initial_nonrotating_vacuum_wald(*B, *D, grid);
  // B->copy_from(*B0);
  // D->copy_from(*D0);
  // initial_vacuum_wald(*B0, *D0, grid);
  // initial_nonrotating_vacuum_wald(*B, *D, grid);
  // B->add_by(*B0, -1.0);
  // D->add_by(*D0, -1.0);

//   pusher->fill_multiplicity(5, 5.0);
ptc_injector_dynamic<Conf> ptc_inj(grid);
ptc_inj.inject_pairs(
    // First function is the injection criterion for each cell. pos is an
    // index_t<Dim> object marking the cell in the grid. Returns true for
    // cells that inject and false for cells that do nothing.
    [] LAMBDA(auto &pos, auto &grid, auto &ext) {
      // if (pos[0] == 80 && pos[1] == 120)
      //   return true;
      // else
      //   return false;
      return true;
    },
    // Second function returns the number of particles injected in each cell.
    // This includes all species
    [] LAMBDA(auto &pos, auto &grid, auto &ext) {
      return 20;
    },
    // Third function is the momentum distribution of the injected particles.
    // Returns a vec_t<value_t, 3> object encoding the 3D momentum of this
    // particular particle
    [] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
      return vec_t<value_t, 3>(0.0, 0.0, 0.0);
    },
    // Fourth function is the particle weight, which can depend on the global
    // coordinate.
    [] LAMBDA(auto &x_global, PtcType type) {
      value_t r = grid_ks_t<Conf>::radius(x_global[0]);
      return 10.0 * math::sin(x_global[1]) * r;
    });

  env.run();

  return 0;
}
