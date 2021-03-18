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

#include "catch.hpp"
#include "core/multi_array.hpp"
#include "core/random.h"
#include "data/rng_states.h"
#include "utils/index.hpp"
#include "utils/interpolation.hpp"
#include "utils/logger.h"
#include "utils/timer.h"
#include <random>
#include <vector>

using namespace Aperture;

TEST_CASE("Memory layout performance wrt 3D interpolation", "[.interp]") {
  uint32_t N1 = 32;
  uint32_t N2 = 32;
  uint32_t N3 = 32;
  uint32_t ppc = 30;

  uint64_t N = ppc * N1 * N2 * N3;
  std::vector<uint64_t> cells(N);
  std::vector<double> x(N);
  std::vector<double> y(N);
  std::vector<double> z(N);
  std::vector<double> x2(N);
  std::vector<double> y2(N);
  std::vector<double> z2(N);

  rand_state state;
  rng_t rng(&state);

  for (uint64_t i = 0; i < N; i++) {
    cells[i] = i / ppc;
    x[i] = rng.uniform<double>();
    y[i] = rng.uniform<double>();
    z[i] = rng.uniform<double>();
  }

  // initialize field arrays
  auto ext = extent(N1, N2, N3);
  auto v1 = make_multi_array<double, idx_col_major_t>(ext);
  auto v2 = make_multi_array<double, idx_zorder_t>(ext);
  for (auto idx_col : v1.indices()) {
    auto pos = idx_col.get_pos();
    idx_zorder_t<3> idx_z(pos, ext);
    v1[idx_col] = v2[idx_z] = pos[0] * 0.3 - pos[1] + pos[2] * 1.8;
    for (int n = 0; n < ppc; n++) {
      x2[idx_z.linear * ppc + n] = x[idx_col.linear * ppc + n];
      y2[idx_z.linear * ppc + n] = y[idx_col.linear * ppc + n];
      z2[idx_z.linear * ppc + n] = z[idx_col.linear * ppc + n];
    }
  }

  std::vector<double> result_col(N);
  std::vector<double> result_z(N);

  interp_t<1, 3> interp;

  timer::stamp();
  for (uint64_t i = 0; i < N; i++) {
    idx_col_major_t<3> idx(cells[i], ext);
    auto pos = get_pos(idx, ext);
    if (pos[0] < N1 - 1 && pos[1] < N2 - 1 && pos[2] < N3 - 1)
      result_col[i] = interp(vec_t<double, 3>(x[i], y[i], z[i]), v1, idx, ext);
    else
      result_col[i] = 0.0;
  }
  timer::show_duration_since_stamp("col major indexing", "ms");

  timer::stamp();
  for (uint64_t i = 0; i < N; i++) {
    idx_zorder_t<3> idx(cells[i], ext);
    auto pos = get_pos(idx, ext);
    idx_col_major_t<3> idx_col(pos, ext);
    if (pos[0] < N1 - 1 && pos[1] < N2 - 1 && pos[2] < N3 - 1)
      result_z[idx_col.linear * ppc + (i % ppc)] =
          interp(vec_t<double, 3>(x2[i], y2[i], z2[i]), v2, idx, ext);
    else
      result_z[idx_col.linear * ppc + (i % ppc)] = 0.0;
  }
  timer::show_duration_since_stamp("zorder indexing", "ms");

  for (uint64_t i = 0; i < N; i++) {
    REQUIRE(result_col[i] == Approx(result_z[i]));
  }
}
