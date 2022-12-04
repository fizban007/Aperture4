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
#include "utils/timer.h"
#include <vector>
#include <iostream>
#include <random>

using namespace Aperture;

TEST_CASE("Uniform random numbers", "[rng]") {
  rand_state state;
  rng_t rng(&state);

  int N = 1000000;
  int M = 20;
  std::vector<double> hist(M, 0.0);

  timer::stamp();
  for (int n = 0; n < N; n++) {
    double u = rng.uniform<double>();
    hist[clamp(int(u * M), 0, M - 1)] += 1.0 / N;
  }
  timer::show_duration_since_stamp("Generating 1M random numbers", "ms");
  for (int m = 0; m < M; m++) {
    std::cout << hist[m] << std::endl;
    hist[m] = 0.0;
  }

  std::mt19937_64 eng;
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  timer::stamp();
  for (int n = 0; n < N; n++) {
    double u = dist(eng);
    hist[clamp(int(u * M), 0, M - 1)] += 1.0 / N;
  }
  timer::show_duration_since_stamp("Generating 1M random numbers using std", "ms");
  for (int m = 0; m < M; m++) {
    std::cout << hist[m] << std::endl;
    hist[m] = 0.0;
  }
}
