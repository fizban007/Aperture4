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
  ic.compute_coefficients(spec, spec.emin(), spec.emax(), 1.0);

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
  int N = 100000;
  int n_scatter = 1000;
  buffer<double> gamma_e(N);
  buffer<double> eph(n_scatter * N);
  rand_state state;
  rng_t rng(&state);

  // Prepare electron spectrum
  for (int i = 0; i < N; i++) {

  }

  file.close();

  return 0;
}
