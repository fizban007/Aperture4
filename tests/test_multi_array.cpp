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

#include "catch2/catch_all.hpp"
#include <catch2/catch_approx.hpp>
#include "core/multi_array.hpp"
#include "core/multi_array_exp.hpp"
#include "core/ndsubset.hpp"
#include "utils/interpolation.hpp"
#include "utils/logger.h"
#include "utils/range.hpp"
#include "utils/timer.h"
#include <algorithm>
#include <random>

using namespace Aperture;

struct Tuple {
  int32_t x, y;
};

// inline index_t<2> get_pos(const idx_col_major_t<2>& idx, const extent_t<2>& ext) {
//   return index_t<2>(idx.linear % ext[0], idx.linear / ext[0]);
// }

TEST_CASE("Morton incX and incY", "[morton]") {
  SECTION("2d case") {
    auto m = morton2(10, 5);

    auto n = m.decX(3).incY(5);
    uint64_t x, y;
    n.decode(x, y);
    REQUIRE(x == 7);
    REQUIRE(y == 10);

    m.incX(5).decY(2).decode(x, y);
    REQUIRE(x == 15);
    REQUIRE(y == 3);

    morton2(0, 0).incX(7).incY(7).decode(x, y);
    REQUIRE(x == 7);
    REQUIRE(y == 7);
  }

  SECTION("3d case") {
    auto m = morton3(10, 15, 8);

    uint64_t x, y, z;
    m.incX(4).decY(9).incZ(4).decode(x, y, z);
    REQUIRE(x == 14);
    REQUIRE(y == 6);
    REQUIRE(z == 12);
  }

  SECTION("zorder_t") {
    idx_zorder_t<2> idx(index(10, 5), extent(32, 32));
    auto n = idx.dec<0>(3).inc<1>(5).get_pos();
    REQUIRE(n[0] == 7);
    REQUIRE(n[1] == 10);

    idx_zorder_t<3> idx2(index(10, 15, 8), extent(32, 32, 32));
    auto m = idx2.dec_x(4).inc_y(9).inc_z(4).get_pos();
    REQUIRE(m[0] == 6);
    REQUIRE(m[1] == 24);
    REQUIRE(m[2] == 12);
  }
}

TEST_CASE("row and col major inc and dec", "[index]") {
  auto ext = extent(10, 10, 10);
  idx_col_major_t<3> idc(index(3, 7, 9), ext);

  // Col major is z y x index order
  auto pos = idc.inc<0>(4).dec<1>(2).dec<2>(5).get_pos();
  REQUIRE(pos[0] == 7);
  REQUIRE(pos[1] == 5);
  REQUIRE(pos[2] == 4);

  idx_row_major_t<3> idr(index(3, 7, 9), ext);

  // Row major is z y x index order
  pos = idr.inc_z(4).dec_y(2).dec_x(5).get_pos();
  REQUIRE(pos[0] == 7);
  REQUIRE(pos[1] == 5);
  REQUIRE(pos[2] == 4);
  // REQUIRE(idr.strides[0] == 10 * 10);
}

TEST_CASE("Initialize and Using multi_array", "[multi_array]") {
  Logger::init(0, LogLevel::debug);
  int X = 260, Y = 310;
  extent_t<2> ext(X, Y);

  multi_array<float, 2> a(ext);

  REQUIRE(a.extent()[0] == X);
  REQUIRE(a.extent()[1] == Y);
  REQUIRE(a.size() == 260 * 310);

  SECTION("Indexing") {
    SECTION("Column major indexing") {
      multi_array<float, 2> col_array(3, 3);
      for (uint32_t j = 0; j < 3; j++) {
        for (uint32_t i = 0; i < 3; i++) {
          col_array(i, j) = i * j;
        }
      }

      for (uint32_t j = 0; j < 3; j++) {
        for (uint32_t i = 0; i < 3; i++) {
          REQUIRE(col_array(i, j) == i * j);
        }
      }

      auto idx = col_array.get_idx(2, 1);
      REQUIRE(col_array[idx.inc<1>()] == 4.0f);
    }

    SECTION("Row Major indexing") {
      multi_array<float, 2, idx_row_major_t<2>> row_array(extent(4, 5),
                                                          MemType::host_only);

      for (uint32_t j = 0; j < 4; j++) {
        for (uint32_t i = 0; i < 5; i++) {
          row_array(j, i) = j * i;
        }
      }

      for (uint32_t j = 0; j < 4; j++) {
        for (uint32_t i = 0; i < 5; i++) {
          REQUIRE(row_array(j, i) == j * i);
        }
      }
    }

    SECTION("Z-order indexing") {
      multi_array<float, 2, idx_zorder_t<2>> zorder_array(extent(32, 32),
                                                          MemType::host_only);

      for (uint32_t j = 0; j < 32; j++) {
        for (uint32_t i = 0; i < 32; i++) {
          zorder_array(j, i) = j * i;
        }
      }

      for (uint32_t i = 0; i < zorder_array.size(); i++) {
        uint64_t x, y;
        morton2(i).decode(x, y);
        REQUIRE(zorder_array[i] == x * y);
      }

      auto idx = zorder_array.get_idx(7, 7);
      REQUIRE(idx.linear == 63);
      REQUIRE(idx.inc<0>().linear == zorder_array.get_idx(8, 7).linear);
    }

    SECTION("Z-order indexing throws when dimensions are not powers of 2") {
      REQUIRE_THROWS(multi_array<float, 2, idx_zorder_t<2>>(32, 10));
      REQUIRE_THROWS(multi_array<float, 2, idx_zorder_t<2>>(10, 16));
    }
  }

  SECTION("Moving") {
    multi_array<float, 2> a(10, 10);
    a(3, 3) = 3.0f;

    auto b = std::move(a);

    // REQUIRE(a.extent()[0] == 0);
    REQUIRE(a.size() == 0);
    REQUIRE(b.size() == 100);
    REQUIRE(b(3, 3) == 3.0f);
  }

  SECTION("Index manipulation, col major") {
    auto ext = extent(10, 20, 30, 40);
    multi_array<float, 4, idx_col_major_t<4>>::idx_t idx(index(0, 0, 0, 0), ext);

    REQUIRE(idx.linear == 0);
    // REQUIRE(idx.strides[0] == 1);
    // REQUIRE(idx.strides[1] == 10);
    // REQUIRE(idx.strides[2] == 10 * 20);
    // REQUIRE(idx.strides[3] == 10 * 20 * 30);

    auto p = idx.pos(7 + 13 * 10 + 15 * 10 * 20 + 34 * 10 * 20 * 30);
    REQUIRE(p[0] == 7);
    REQUIRE(p[1] == 13);
    REQUIRE(p[2] == 15);
    REQUIRE(p[3] == 34);

    auto idx2 = ++idx;
    REQUIRE(idx2.linear == 1);
    // REQUIRE(idx2.strides[0] == 1);
    // REQUIRE(idx2.strides[1] == 10);
    // REQUIRE(idx2.strides[2] == 10 * 20);
    // REQUIRE(idx2.strides[3] == 10 * 20 * 30);
  }

  SECTION("Index manipulation, row major") {
    auto ext = extent(20, 20, 20, 20);
    idx_row_major_t<4> idx(index(12, 5, 8, 9), ext);

    REQUIRE(idx.linear == 9 + 8 * 20 + 5 * 20 * 20 + 12 * 20 * 20 * 20);
    // REQUIRE(idx.strides[0] == 20 * 20 * 20);
    // REQUIRE(idx.strides[1] == 20 * 20);
    // REQUIRE(idx.strides[2] == 20);
    // REQUIRE(idx.strides[3] == 1);

    auto idx2 = idx++;
    REQUIRE(idx2.linear == 9 + 8 * 20 + 5 * 20 * 20 + 12 * 20 * 20 * 20);
    REQUIRE(idx.linear == 10 + 8 * 20 + 5 * 20 * 20 + 12 * 20 * 20 * 20);
    // REQUIRE(idx2.strides[0] == 20 * 20 * 20);
    // REQUIRE(idx2.strides[1] == 20 * 20);
    // REQUIRE(idx2.strides[2] == 20);
    // REQUIRE(idx2.strides[3] == 1);

    auto p = idx.pos(7 + 13 * 20 + 15 * 20 * 20 + 4 * 20 * 20 * 20);
    REQUIRE(p[0] == 4);
    REQUIRE(p[1] == 15);
    REQUIRE(p[2] == 13);
    REQUIRE(p[3] == 7);
  }

  SECTION("Index manipulation, z-order") {
    auto ext = extent(32, 32, 32);
    idx_zorder_t<3> idx(index_t<3>(3, 5, 7), ext);

    REQUIRE(idx.linear == morton3(3, 5, 7).key);

    idx += 14;

    REQUIRE(idx.linear == morton3(3, 5, 7).key + 14);
    auto p = idx.get_pos();
    uint64_t x, y, z;
    morton3(idx.linear).decode(x, y, z);
    REQUIRE(p[0] == x);
    REQUIRE(p[1] == y);
    REQUIRE(p[2] == z);
  }

  SECTION("ndptr") {
    multi_array<float, 2> a(30, 20);
    auto p = a.dev_ndptr();

    auto idx = p.idx_at(100, a.extent());
    REQUIRE(idx.linear == 100);
    // REQUIRE(idx.strides[1] == 30);

    auto pos = idx.get_pos();
    REQUIRE(pos[0] == (100 % 30));
    REQUIRE(pos[1] == (100 / 30));

    SECTION("ndptr with Z-order indexing") {
      // multi_array<float, MemType::host_only, idx_zorder_t<>>
      // array(
      //     32, 32, 1);
      auto array = make_multi_array<float, idx_zorder_t>(extent(32, 32),
                                                         MemType::host_only);
      auto p = array.dev_ndptr();

      auto idx = p.idx_at(63, a.extent());

      auto pos = idx.get_pos();
      REQUIRE(pos[0] == 7);
      REQUIRE(pos[1] == 7);
      REQUIRE(array.get_idx(pos[0], pos[1]).linear == 63);
    }
  }

  SECTION("range-based indexing") {
    // multi_array<float> a(30, 20);
    auto a = make_multi_array<float>(extent(20, 30), MemType::host_only);

    for (auto n : indices(a)) {
      a[n] = n;
    }

    for (int n = 0; n < a.size(); n++) {
      REQUIRE(a[n] == n);
    }
  }

  SECTION("make_multi_array") {
    auto arr = make_multi_array<float, idx_row_major_t>(extent(30, 30, 30),
                                                        MemType::host_only);

    REQUIRE(arr.size() == 30 * 30 * 30);
  }

  // SECTION("multi_array returning an index range") {
  //   multi_array<float, MemType::host_only, idx_col_major_t<>> a(
  //       30, 20, 1);

  //   for (auto idx : a.indices()) {
  //     a[idx] = idx.key;
  //   }

  //   for (int n = 0; n < a.size(); n++) {
  //     REQUIRE(a[n] == n);
  //   }

  //   multi_array<float, MemType::host_only, idx_row_major_t<>> b(
  //       1, 20, 30);

  //   for (auto idx : b.indices()) {
  //     b[idx] = idx.key;
  //   }

  //   for (int n = 0; n < b.size(); n++) {
  //     REQUIRE(b[n] == n);
  //   }

  //   multi_array<float, MemType::host_only, idx_zorder_t<>> c(
  //       32, 32, 32);

  //   for (auto idx : c.indices()) {
  //     c[idx] = idx.key;
  //   }

  //   for (int n = 0; n < c.size(); n++) {
  //     REQUIRE(c[n] == n);
  //   }
  // }
}

TEST_CASE("Performance of 3d interpolation on CPU",
          "[multi_array][performance][.]") {
  Logger::print_info("3d interpolation");
  uint32_t N = 128;
  uint32_t N1 = N, N2 = N, N3 = N;
  std::default_random_engine g;
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::uniform_int_distribution<uint32_t> cell_dist(0, N1 * N2 * N3);

  auto ext = extent(N1, N2, N3);

  auto v1 = make_multi_array<float, idx_col_major_t>(ext, MemType::host_only);
  auto v2 = make_multi_array<float, idx_zorder_t>(ext, MemType::host_only);

  for (auto idx : v1.indices()) {
    auto pos = idx.get_pos();
    v1[idx] = float(0.3 * pos[0] + 0.4 * pos[1] - pos[2]);
  }
  for (auto idx : v2.indices()) {
    auto pos = idx.get_pos();
    v2[idx] = float(0.3 * pos[0] + 0.4 * pos[1] - pos[2]);
  }
  for (auto idx : v1.indices()) {
    auto pos = idx.get_pos();
    REQUIRE(v1(pos[0], pos[1], pos[2]) == v2(pos[0], pos[1], pos[2]));
  }

  // Generate M random numbers
  int M = 1000000;
  buffer<float> xs(M, MemType::host_only);
  buffer<float> ys(M, MemType::host_only);
  buffer<float> zs(M, MemType::host_only);
  buffer<float> result1(M, MemType::host_only);
  buffer<float> result2(M, MemType::host_only);
  buffer<uint32_t> cells1(M, MemType::host_only);
  buffer<uint32_t> cells2(M, MemType::host_only);
  for (int n = 0; n < M; n++) {
    xs[n] = dist(g);
    ys[n] = dist(g);
    zs[n] = dist(g);
    // cells1[n]
    auto c = cell_dist(g);
    auto pos = v1.idx_at(c).get_pos();
    cells1[n] =
        v1.get_idx(clamp(pos[0], 2, int(N1 - 3)), clamp(pos[1], 2, int(N2 - 3)),
                   clamp(pos[2], 2, int(N3 - 3)))
            .linear;
    auto idx =
        v2.get_idx(clamp(pos[0], 2, int(N1 - 3)), clamp(pos[1], 2, int(N2 - 3)),
                   clamp(pos[2], 2, int(N3 - 3)));
    cells2[n] = idx.linear;
    result1[n] = 0.0f;
    result2[n] = 0.0f;
  }
  std::sort(cells1.host_ptr(), cells1.host_ptr() + cells1.size());
  std::sort(cells2.host_ptr(), cells2.host_ptr() + cells2.size());

  auto interp_kernel = [N1, N2, N3, M](const auto& f, float* result, float* xs,
                                       float* ys, float* zs, uint32_t* cells,
                                       const auto& ext) {
    auto interp = interpolator<bspline<1>, 3>{};
    for (uint32_t i : range(0, M)) {
      uint32_t cell = cells[i];
      auto idx = f.idx_at(cell);
      // auto pos = idx.get_pos();
      auto pos = get_pos(idx, ext);
      if (pos[0] < N1 - 2 && pos[1] < N2 - 2 && pos[2] < N3 - 2) {
        // result[i] = x;
        // result[i] = lerp3(f, xs[i], ys[i], zs[i], idx);
        result[i] = interp(vec_t<float, 3>(xs[i], ys[i], zs[i]), f, idx, ext);
      }
    }
  };

  timer::stamp();
  interp_kernel(v1, result1.host_ptr(), xs.host_ptr(), ys.host_ptr(),
                zs.host_ptr(), cells1.host_ptr(), ext);
  timer::show_duration_since_stamp("normal indexing", "ms");

  timer::stamp();
  interp_kernel(v2, result2.host_ptr(), xs.host_ptr(), ys.host_ptr(),
                zs.host_ptr(), cells2.host_ptr(), ext);
  timer::show_duration_since_stamp("morton indexing", "ms");

  // for (auto idx : range(0, result1.size())) {
  //   CHECK(result1[idx] == result2[idx]);
  // }
}

TEST_CASE("Performance of 2d interpolation on CPU",
          "[multi_array][performance][.]") {
  Logger::print_info("2d interpolation");
  uint32_t N = 1024;
  uint32_t N1 = N, N2 = N;
  std::default_random_engine g;
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::uniform_int_distribution<uint32_t> cell_dist(0, N1 * N2);

  auto ext = extent(N1, N2);

  auto v1 = make_multi_array<float, idx_col_major_t>(ext, MemType::host_only);
  auto v2 = make_multi_array<float, idx_zorder_t>(ext, MemType::host_only);

  for (auto idx : v1.indices()) {
    auto pos = idx.get_pos();
    v1[idx] = float(0.3 * pos[0] + 0.4 * pos[1]);
  }
  for (auto idx : v2.indices()) {
    auto pos = idx.get_pos();
    v2[idx] = float(0.3 * pos[0] + 0.4 * pos[1]);
  }
  for (auto idx : v1.indices()) {
    auto pos = idx.get_pos();
    REQUIRE(v1(pos[0], pos[1]) == v2(pos[0], pos[1]));
  }

  // Generate M random numbers
  int M = 1000000;
  buffer<float> xs(M, MemType::host_only);
  buffer<float> ys(M, MemType::host_only);
  // buffer<float> zs(M, MemType::host_only);
  buffer<float> result1(M, MemType::host_only);
  buffer<float> result2(M, MemType::host_only);
  buffer<uint32_t> cells1(M, MemType::host_only);
  buffer<uint32_t> cells2(M, MemType::host_only);
  for (int n = 0; n < M; n++) {
    xs[n] = dist(g);
    ys[n] = dist(g);
    // zs[n] = dist(g);
    // cells1[n]
    auto c = cell_dist(g);
    auto pos = v1.idx_at(c).get_pos();
    cells1[n] = v1.get_idx(clamp(pos[0], 2, int(N1 - 3)),
                          clamp(pos[1], 2, int(N2 - 3))).linear;
    auto idx = v2.get_idx(clamp(pos[0], 2, int(N1 - 3)),
                          clamp(pos[1], 2, int(N2 - 3)));
    cells2[n] = idx.linear;
    result1[n] = 0.0f;
    result2[n] = 0.0f;
  }
  // std::sort(cells1.host_ptr(), cells1.host_ptr() + cells1.size());
  // std::sort(cells2.host_ptr(), cells2.host_ptr() + cells2.size());

  auto interp_kernel = [N1, N2, M](const auto& f, float* result, float* xs,
                                       float* ys, uint32_t* cells,
                                   const auto& ext) {
    auto interp = interpolator<bspline<1>, 2>{};
    for (uint32_t i : range(0, M)) {
      uint32_t cell = cells[i];
      auto idx = f.idx_at(cell);
      auto pos = get_pos(idx, ext);
      if (pos[0] < N1 - 2 && pos[1] < N2 - 2) {
        // result[i] = x;
        // result[i] = lerp2(f, xs[i], ys[i], idx);
        result[i] = interp(vec_t<float, 3>(xs[i], ys[i], 0.0f), f, idx, ext);
      }
    }
  };

  timer::stamp();
  interp_kernel(v1, result1.host_ptr(), xs.host_ptr(), ys.host_ptr(),
                cells1.host_ptr(), ext);
  timer::show_duration_since_stamp("normal indexing", "ms");

  timer::stamp();
  interp_kernel(v2, result2.host_ptr(), xs.host_ptr(), ys.host_ptr(),
                cells2.host_ptr(), ext);
  timer::show_duration_since_stamp("morton indexing", "ms");

  for (auto idx : range(0, result1.size())) {
    CHECK(result1[idx] == result2[idx]);
  }
}

TEST_CASE("Performance of laplacian on CPU, 3d",
          "[multi_array][performance][.]") {
  Logger::print_info("3d grid based");
  uint32_t N = 256;
  uint32_t N1 = N, N2 = N, N3 = N;

  auto ext = extent(N1, N2, N3);

  auto v1 = make_multi_array<float, idx_row_major_t>(ext, MemType::host_only);
  auto v2 = make_multi_array<float, idx_zorder_t>(ext, MemType::host_only);
  auto u1 = make_multi_array<float, idx_row_major_t>(ext, MemType::host_only);
  auto u2 = make_multi_array<float, idx_zorder_t>(ext, MemType::host_only);

  for (auto idx : v1.indices()) {
    auto pos = idx.get_pos();
    v1[idx] = float(0.3 * pos[0] + 0.4 * pos[1] - pos[2]);
    u1[idx] = 0.0f;
  }
  for (auto idx : v2.indices()) {
    auto pos = idx.get_pos();
    v2[idx] = float(0.3 * pos[0] + 0.4 * pos[1] - pos[2]);
    u2[idx] = 0.0f;
  }
  for (auto idx : v1.indices()) {
    // auto pos = idx.get_pos();
    auto pos = get_pos(idx, ext);
    REQUIRE(v1(pos[0], pos[1], pos[2]) == v2(pos[0], pos[1], pos[2]));
    REQUIRE(u1(pos[0], pos[1], pos[2]) == u2(pos[0], pos[1], pos[2]));
  }

  auto diff_kernel = [N1, N2, N3, ext](const auto& f, auto& u) {
    for (auto idx : u.indices()) {
      // auto pos = idx.get_pos();
      auto pos = get_pos(idx, ext);
      if (pos[0] > 1 && pos[1] > 1 && pos[2] > 1 && pos[0] < N1 - 2 &&
          pos[1] < N2 - 2 && pos[2] < N3 - 2) {
        u[idx] =
            0.2f * (f[idx.template inc<1>(2)] - f[idx.template dec<1>(2)]) +
            0.15f * (f[idx.template inc<2>(2)] + f[idx.template dec<2>()]) +
            0.1f * (f[idx.template inc<0>()] - f[idx.template dec<0>()]) -
            0.5f * f[idx];
      }
    }
  };

  timer::stamp();
  diff_kernel(v1, u1);
  timer::show_duration_since_stamp("normal indexing", "ms");

  timer::stamp();
  diff_kernel(v2, u2);
  timer::show_duration_since_stamp("morton indexing", "ms");

  for (auto idx : u1.indices()) {
    auto pos = idx.get_pos();
    // REQUIRE(u1(pos[0], pos[1], pos[2]) == Approx(u2(pos[0], pos[1], pos[2])));
    REQUIRE_THAT(u1(pos[0], pos[1], pos[2]), Catch::Matchers::WithinULP(u2(pos[0], pos[1], pos[2]), 1));
  }
}

TEST_CASE("Performance of laplacian on CPU, 2d",
          "[multi_array][performance][.]") {
  Logger::print_info("2d grid based");
  uint32_t N = 2048;
  uint32_t N1 = N, N2 = N;

  auto ext = extent(N1, N2);

  auto v1 = make_multi_array<float, idx_col_major_t>(ext, MemType::host_only);
  auto v2 = make_multi_array<float, idx_zorder_t>(ext, MemType::host_only);
  auto u1 = make_multi_array<float, idx_col_major_t>(ext, MemType::host_only);
  auto u2 = make_multi_array<float, idx_zorder_t>(ext, MemType::host_only);

  for (auto idx : v1.indices()) {
    auto pos = idx.get_pos();
    v1[idx] = float(0.3 * pos[0] + 0.4 * pos[1]);
    u1[idx] = 0.0f;
  }
  for (auto idx : v2.indices()) {
    auto pos = idx.get_pos();
    v2[idx] = float(0.3 * pos[0] + 0.4 * pos[1]);
    u2[idx] = 0.0f;
  }
  for (auto idx : v1.indices()) {
    auto pos = idx.get_pos();
    REQUIRE(v1(pos[0], pos[1]) == v2(pos[0], pos[1]));
    REQUIRE(u1(pos[0], pos[1]) == u2(pos[0], pos[1]));
  }

  auto diff_kernel = [N1, N2](const auto& f, auto& u) {
    for (auto idx : u.indices()) {
      auto pos = idx.get_pos();
      if (pos[0] > 1 && pos[1] > 1 && pos[0] < N1 - 2 && pos[1] < N2 - 2) {
        u[idx] =
            0.2f * (f[idx.inc_x(2)] - f[idx.inc_x(1)] +
                    f[idx.dec_x(1)] - f[idx.dec_x(2)]) +
            0.1f * (f[idx.inc_y(2)] - f[idx.inc_y(1)] +
                    f[idx.dec_y(1)] - f[idx.dec_y(2)]) -
            0.5f * f[idx];
      }
    }
  };

  auto diff_kernel3 = [N1, N2, ext](const auto& f, auto& u) {
    for (auto idx : u.indices()) {
      auto pos = get_pos(idx, ext);
      if (pos[0] > 1 && pos[1] > 1 && pos[0] < N1 - 2 && pos[1] < N2 - 2) {
        u[idx] =
            0.2f * (f[idx.inc_x(2)] - f[idx.inc_x(1)] +
                    f[idx.dec_x(1)] - f[idx.dec_x(2)]) +
            0.1f * (f[idx.inc_y(2)] - f[idx.inc_y(1)] +
                    f[idx.dec_y(1)] - f[idx.dec_y(2)]) -
            0.5f * f[idx];
      }
    }
  };

  auto diff_kernel2 = [N1, N2](const auto& f, auto& u) {
    // auto idx = typename std::remove_reference_t<decltype(u)>::idx_t(0, u.extent());
    // for (auto idx : u.indices()) {
    for (auto n : range(0, u.extent().size())) {
      // auto pos = idx.get_pos();
      int i = n % N1;
      int j = n / N1;
      if (i > 1 && j > 1 && i < N1 - 2 && j < N2 - 2) {
        u[n] =
            // 0.2f * (f[idx.template inc<0>(2)] - f[idx.template inc<0>(1)] +
            //         f[idx.template dec<0>(1)] - f[idx.template dec<0>(2)]) +
            // 0.1f * (f[idx.template inc<1>(2)] - f[idx.template inc<1>(1)] +
            //         f[idx.template dec<1>(1)] - f[idx.template dec<1>(2)]) -
            // 0.5f * f[idx];
            0.2f * (f[n + 2] - f[n + 1] +
                    f[n - 1] - f[n - 2]) +
            0.1f * (f[n + 2 * N1] - f[n + N1] +
                    f[n - N1] - f[n - 2 * N1]) -
            0.5f * f[n];
      }
    }
  };

  timer::stamp();
  diff_kernel3(v1, u1);
  timer::show_duration_since_stamp("normal indexing", "ms");

  timer::stamp();
  diff_kernel(v2, u2);
  timer::show_duration_since_stamp("morton indexing", "ms");

  for (auto idx : u1.indices()) {
    auto pos = idx.get_pos();
    CHECK(u1(pos[0], pos[1]) == Catch::Approx(u2(pos[0], pos[1])));
  }
}

TEST_CASE("Assign and copy", "[multi_array]") {
  SECTION("host only") {
    auto v1 = make_multi_array<float>(extent(30, 30));
    auto v2 = make_multi_array<float>(extent(30, 30));

    v1.assign_host(3.0f);
    for (auto idx : v1.indices()) {
      REQUIRE(v1[idx] == 3.0f);
    }
  }
}

TEST_CASE("Memtype is correct", "[multi_array][managed]") {
  {
    auto m = make_multi_array<float>(extent(32, 32, 32), MemType::device_only);
    REQUIRE(m.mem_type() == MemType::device_only);
    REQUIRE(m.host_allocated() == false);
  }

  {
    auto m =
        make_multi_array<float>(extent(32, 32, 32), MemType::device_managed);
    REQUIRE(m.mem_type() == MemType::device_managed);
    REQUIRE(m.host_allocated() == false);
#ifdef GPU_ENABLED
    REQUIRE(m.host_ptr() != nullptr);
#endif
  }

  {
    auto m = make_multi_array<double>(extent(32, 32, 32), MemType::host_device);
    REQUIRE(m.mem_type() == MemType::host_device);
    REQUIRE(m.host_allocated() == true);
#ifdef GPU_ENABLED
    REQUIRE(m.dev_allocated() == true);
#endif
  }
}

TEST_CASE("Expression before subscript", "[multi_array][exp_template]") {
  auto v1 = make_multi_array<float>(extent(30, 30), MemType::host_only);
  auto v2 = make_multi_array<float>(extent(30, 30), MemType::host_only);

  v1.assign_host(1.0);
  v2.assign_host(2.0);

  REQUIRE(is_host_const_indexable<multi_array<float, 2, idx_col_major_t<2>>::cref_t>::value);
  // auto ex = -(v1 + v2) * v2 - v1;

  // for (auto idx : v1.indices()) {
    // REQUIRE(ex.at(idx) == -7.0);
  // }
}

TEST_CASE("Expression templates with constants",
          "[multi_array][exp_template]") {
  auto v = make_multi_array<float>(extent(30, 30), MemType::host_only);

  v.assign_host(5.0f);

  auto ex = (3.0f - v) * 4.0f / 2.0f;
  // auto v2 = ex.select(index(0, 0), extent(20, 10)).to_multi_array();

  for (auto idx : v.indices()) {
    REQUIRE((v + 3.0f)[idx] == 8.0f);
    REQUIRE(ex[idx] == -4.0f);
  }
}

TEST_CASE("Testing select", "[multi_array][exp_template]") {
  auto v = make_multi_array<float>(extent(30, 30), MemType::host_only);

  v.assign_host(3.0f);

  auto w = select(exec_tags::host{}, v, index(0, 0), extent(10, 10));
  w += select(exec_tags::host{}, (v * 3.0f + 4.0f), index(0, 0), extent(10, 10));

  for (auto idx : v.indices()) {
    auto pos = idx.get_pos();
    if (pos[0] < 10 && pos[1] < 10) {
      REQUIRE(v[idx] == 16.0f);
    } else {
      REQUIRE(v[idx] == 3.0f);
    }
  }

  select(exec_tags::host{}, v) = 5.0f;
  for (auto idx : v.indices()) {
    REQUIRE(v[idx] == 5.0f);
  }

  select(exec_tags::host{}, v) *= v + 4.0f;
  for (auto idx : v.indices()) {
    REQUIRE(v[idx] == 45.0f);
  }
}
