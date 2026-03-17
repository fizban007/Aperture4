/*
 * Copyright (c) 2022 Alex Chen.
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
#include "data/particle_data.h"
#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/compute_moments_impl.hpp"
#include "systems/data_exporter_impl.hpp"
#include "systems/domain_comm_impl.hpp"
#include "systems/field_solver_cartesian_impl.hpp"
#include "systems/gather_tracked_ptc_impl.hpp"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "systems/ptc_updater_impl.hpp"
#include <iostream>

using namespace std;
// namespace Aperture {

//   //implement a  bimaxwell distribution
  



// }
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  auto &env = sim_environment::instance(&argc, &argv, true);
  typedef typename Conf::value_t value_t;
  using exec_policy = exec_policy_dynamic<Conf>;

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_t<Conf> grid(comm);
  auto pusher = env.register_system<
      ptc_updater<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid,
                                                                      &comm);
  auto solver = env.register_system<
      field_solver<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid,
                                                                       &comm);
  // auto tracker =
  //     env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  auto moments =
      env.register_system<compute_moments<Conf, exec_policy_dynamic>>(grid);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(
      grid, &comm);

  env.init();

 
  // Prepare initial conditions
  int ppc = 20;
  env.params().get_value("ppc", ppc);
  value_t betapara =10;
  env.params().get_value("betapara", betapara);
  value_t sigma = 1.0;
  env.params().get_value("sigma", sigma);
  value_t A=2.0;// Tperp/Tpara
  env.params().get_value("temp_anisotropy", A);
  value_t thetaB=0.0;// Angle of the magnetic field from vertical into the z direction, in radians
  env.params().get_value("thetaB", thetaB);
  



  // value_t Bp = 1.0;
  value_t Tpara=0.5 * betapara * sigma;
  value_t Tperp=A*Tpara;
  // value_t sqrtTpara= math::sqrt(Tpara);
  // value_t sqrtTperp= math::sqrt(Tperp);
  // printf("ppc %d Tpara %f Tperp", ppc, Tpara, Tperp);
//   env.params().get_value("rho_0", rho_0);


  ptc_injector_dynamic<Conf> injector(grid);

 //Prepare Vertical field
  vector_field<Conf> *B, *B0;
  env.get_data("B0", &B0);
  env.get_data("Bdelta", &B);


   // Set initial magnetic field
  // B0->set_values(1, [bz](Scalar x, Scalar y, Scalar z) {
  //   return math::sqrt(1.0-bz*bz);
  // });
  //By
  B0->set_values(1, [thetaB](Scalar x, Scalar y, Scalar z) {
    return math::cos(thetaB);
  });
  //Bz
  B0->set_values(2, [thetaB](Scalar x, Scalar y, Scalar z) {
    return math::sin(thetaB);
  });
    


  injector.inject_pairs(
      // First function is the injection criterion for each cell. pos is an
      // index_t<Dim> object marking the cell in the grid. Returns true for
      // cells that inject and false for cells that do nothing.
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      // Second function returns the number of particles injected in each cell.
      // This includes all species
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) { return ppc; },
      // Third function is the momentum distribution of the injected particles.
      // Returns a vec_t<value_t, 3> object encoding the 3D momentum of this
      // particular particle
      [Tperp,A,thetaB] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
      // value_t ux = rng_gaussian(state, sqrtTperp);
      // value_t uy = rng_gaussian(state, sqrtTpara);
      // value_t uz = rng_gaussian(state, sqrtTperp);
      // value_t u = math::sqrt(ux*ux + uy*uy + uz*uz);
      // value_t gamma = 1.0 / math::sqrt(1.0 - u*u);
      // return vec_t<Scalar, 3>{ux * gamma, uy* gamma, uz* gamma};
      // return vec_t<Scalar, 3>{ux,uy*math::cos(thetaB)-uz*math::sin(thetaB),uz*math::cos(thetaB)+uy*math::sin(thetaB)};
      return rng_anisotropic_maxwell_juttner_3d(state,Tperp,A,vec_t<Scalar, 3>{0,math::cos(thetaB),math::sin(thetaB)});
      },
      // Fourth function is the particle weight, which can depend on the global
      // coordinate.
      [ppc,sigma] LAMBDA(auto &x_global, PtcType type) {
        // value_t weight=1/ (sigma * ppc);
        // printf("weight %f \n", weight);
        return 1.0 / (sigma * ppc);
        // return 1/ ppc;
      });


  env.run();
  return 0;
}
