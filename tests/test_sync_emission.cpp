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

#include "core/buffer.hpp"
#include "core/random.h"
#include "systems/sync_curv_emission.h"
#include "utils/hdf_wrapper.h"

using namespace Aperture;

int main(int argc, char *argv[]) {
  sync_curv_emission_t sc(MemType::host_only);
  auto sc_helper = sc.get_helper();

  H5File file = hdf_create("sync_test.h5");

  file.write(sc_helper.ptr_lookup, sc_helper.nx, "Fx_cumulative");

  int N = 1000000;
  rand_state state;
  rng_t<exec_tags::host> rng(&state);

  buffer<float> eph1(N), eph2(N), eph3(N);
  float Rc_over_c = 1e8/3e10;
  float omega_Q = 7.76e20;
  for (int i = 0; i < N; i++) {
    eph1[i] = sc_helper.gen_curv_photon(10, Rc_over_c, omega_Q, rng.m_local_state);
    eph2[i] = sc_helper.gen_curv_photon(1e3, Rc_over_c, omega_Q, rng.m_local_state);
    eph3[i] = sc_helper.gen_curv_photon(1e5, Rc_over_c, omega_Q, rng.m_local_state);
  }

  file.write(eph1, "gamma1e1");
  file.write(eph2, "gamma1e3");
  file.write(eph3, "gamma1e5");

  file.close();

  return 0;
}
