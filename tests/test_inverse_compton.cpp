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

#include "core/buffer.hpp"
#include "core/random.h"
#include "systems/inverse_compton.h"
#include "systems/physics/spectra.hpp"
#include "utils/hdf_wrapper.h"

using namespace Aperture;

int
main(int argc, char *argv[]) {
  inverse_compton_t ic;

  Spectra::black_body spec(1e-3);
  ic.compute_coefficients(spec, spec.emin(), spec.emax());

  H5File file = hdf_create("ic_test.h5");

  Logger::print_info("ic_rate has size {}", ic.ic_rate().size());

  file.write(ic.ic_rate().host_ptr(), ic.ic_rate().size(), "ic_rate");
  file.write(ic.gg_rate().host_ptr(), ic.gg_rate().size(), "gg_rate");

  file.write(ic.dNde(), "dNde");
  file.write(ic.dNde_thomson(), "dNde_thomson");

  file.write(ic.min_ep(), "min_ep");
  file.write(ic.dgamma(), "dgamma");
  file.write(ic.dep(), "dep");
  file.write(ic.dlep(), "dlep");

  auto ic_module = ic.get_ic_module();
  int N = 1000000;
  int n_scatter = 1000;
  // buffer<double> gamma_e(N);
  // buffer<double> eph(n_scatter * N);
  rand_state state;
  rng_t rng(&state);

  buffer<double> g1(N), g2(N), g3(N), g4(N), g5(N);
  // Prepare electron spectrum
  for (int i = 0; i < N; i++) {
    // auto u = rng.uniform<double>();
    g1[i] = ic_module.gen_photon_e(10, rng);
    g2[i] = ic_module.gen_photon_e(100, rng);
    g3[i] = ic_module.gen_photon_e(1000, rng);
    g4[i] = ic_module.gen_photon_e(10000, rng);
    g5[i] = ic_module.gen_photon_e(100000, rng);
  }

  file.write(g1, "mono1e1");
  file.write(g2, "mono1e2");
  file.write(g3, "mono1e3");
  file.write(g4, "mono1e4");
  file.write(g5, "mono1e5");

  file.close();

  return 0;
}
