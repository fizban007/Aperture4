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

#include "catch2/catch_all.hpp"
#include "core/random.h"
#include "data/rng_states.h"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include "utils/timer.h"

using namespace Aperture;

TEST_CASE("Uniform random numbers", "[rng]") {
  rng_states_t<exec_tags::device> states;
  states.init();

  Logger::print_info("Init success!");

  int N = 1000000;
  int M = 20;
  buffer<float> hist(M);
  hist.assign_dev(0.0f);

  timer::stamp();
  kernel_launch([N, M] __device__ (auto states, auto hist) {
      rng_t rng(states);
      for (auto n : grid_stride_range(0, N)) {
        auto u = rng.uniform<float>();
        // if (n % 10000 == 0) printf("%d, %f\n", n, u);
        atomicAdd(&hist[clamp(int(u * M), 0, M - 1)], 1.0f / N);
      }
    }, states.states().dev_ptr(), hist.dev_ptr());
  GpuSafeCall(gpuDeviceSynchronize());
  timer::show_duration_since_stamp("Generating 1M random numbers", "ms");
  hist.copy_to_host();
  for (int m = 0; m < M; m++) {
    std::cout << hist[m] << std::endl;
    hist[m] = 0.0;
  }

}
