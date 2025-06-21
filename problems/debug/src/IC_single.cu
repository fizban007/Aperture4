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

#include "core/buffer.hpp"
#include "data/particle_data.h"
#include "framework/config.h"
#include "framework/environment.h"
// #include "systems/boundary_condition.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
// #include "systems/field_solver.h"
#include "systems/gather_momentum_space.h"
#include "systems/physics/ic_scattering.hpp"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/exec_policy_gpu.hpp"
#include "systems/policies/phys_policy_IC_cooling.hpp"
#include "systems/policies/ptc_physics_policy_empty.hpp"
#include "systems/ptc_updater_impl.hpp"
#include "systems/radiation/IC_radiation_scheme.hpp"
#include "systems/radiative_transfer_impl.hpp"
#include "utils/hdf_wrapper.h"
#include <iostream>

using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  typedef typename Conf::value_t value_t;
  auto &env = sim_environment::instance(&argc, &argv, false);

  value_t emin = 1.0e-5;
  sim_env().params().get_value("emin", emin);
  value_t ic_alpha = 1.25;
  sim_env().params().get_value("IC_alpha", ic_alpha);
  inverse_compton_t ic;
  Spectra::power_law_soft spec(ic_alpha, emin, 1.0);
  ic.compute_coefficients(spec, spec.emin(), spec.emax());
  auto ic_module = ic.get_ic_module();

  rng_states_t<exec_tags::device> rng_states;
  rng_states.init();
  int N = 1000000;
  value_t gamma = 1000.0;
  buffer<value_t> eph(N, MemType::device_managed);

  kernel_launch([N, ic_module, gamma] __device__ (rand_state* states, value_t* eph) {
      rng_t<exec_tags::device> rng(states);
      for (auto tid : grid_stride_range(0, N)) {
        eph[tid] = ic_module.gen_photon_e(gamma, rng.m_local_state) * gamma;
        // printf("eph is %f\n", eph[tid]);
      }
    }, rng_states.states().dev_ptr(), eph.dev_ptr());
  gpuDeviceSynchronize();

  H5File outfile = hdf_create("eph_output.h5");
  outfile.write(eph.dev_ptr(), eph.size(), "eph");
  outfile.close();

  return 0;
}
