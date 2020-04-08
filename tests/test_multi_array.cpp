#include "catch.hpp"
#include "core/multi_array.hpp"
#include "utils/interpolation.hpp"
#include "utils/logger.h"
#include "utils/range.hpp"
#include "utils/timer.h"
#include <algorithm>
#include <random>

using namespace Aperture;

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
      multi_array<float, 2, MemoryModel::host_only, idx_row_major_t<2>>
          row_array(4, 5);

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
      multi_array<float, 2, MemoryModel::host_only, idx_zorder_t<2>>
          zorder_array(32, 32);

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

    SECTION(
        "Z-order indexing throws when dimensions are not powers of 2") {
      REQUIRE_THROWS(multi_array<float, 2, MemoryModel::host_only,
                                 idx_zorder_t<2>>(32, 10));
      REQUIRE_THROWS(multi_array<float, 2, MemoryModel::host_only,
                                 idx_zorder_t<2>>(10, 16));
    }
  }

  SECTION("Moving") {
    multi_array<float, 2> a(10, 10);
    a(3, 3) = 3.0f;

    auto b = std::move(a);

    REQUIRE(a.extent()[0] == 0);
    REQUIRE(a.size() == 0);
    REQUIRE(b.size() == 100);
    REQUIRE(b(3, 3) == 3.0f);
  }

  SECTION("Index manipulation, col major") {
    multi_array<float, 4>::index_type idx(
        vec<uint32_t>(0, 0, 0, 0), vec<uint32_t>(10, 20, 30, 40));

    REQUIRE(idx.linear == 0);
    REQUIRE(idx.strides[0] == 1);
    REQUIRE(idx.strides[1] == 10);
    REQUIRE(idx.strides[2] == 10 * 20);
    REQUIRE(idx.strides[3] == 10 * 20 * 30);

    auto p =
        idx.get_pos(7 + 13 * 10 + 15 * 10 * 20 + 34 * 10 * 20 * 30);
    REQUIRE(p[0] == 7);
    REQUIRE(p[1] == 13);
    REQUIRE(p[2] == 15);
    REQUIRE(p[3] == 34);

    auto idx2 = ++idx;
    REQUIRE(idx2.linear == 1);
    REQUIRE(idx2.strides[0] == 1);
    REQUIRE(idx2.strides[1] == 10);
    REQUIRE(idx2.strides[2] == 10 * 20);
    REQUIRE(idx2.strides[3] == 10 * 20 * 30);
  }

  SECTION("Index manipulation, row major") {
    idx_row_major_t<4> idx(vec<uint32_t>(12, 5, 8, 9),
                           vec<uint32_t>(20, 20, 20, 20));

    REQUIRE(idx.linear == 9 + 8 * 20 + 5 * 20 * 20 + 12 * 20 * 20 * 20);
    REQUIRE(idx.strides[0] == 20 * 20 * 20);
    REQUIRE(idx.strides[1] == 20 * 20);
    REQUIRE(idx.strides[2] == 20);
    REQUIRE(idx.strides[3] == 1);

    auto idx2 = idx++;
    REQUIRE(idx2.linear ==
            9 + 8 * 20 + 5 * 20 * 20 + 12 * 20 * 20 * 20);
    REQUIRE(idx.linear ==
            10 + 8 * 20 + 5 * 20 * 20 + 12 * 20 * 20 * 20);
    REQUIRE(idx2.strides[0] == 20 * 20 * 20);
    REQUIRE(idx2.strides[1] == 20 * 20);
    REQUIRE(idx2.strides[2] == 20);
    REQUIRE(idx2.strides[3] == 1);

    auto p = idx.get_pos(7 + 13 * 20 + 15 * 20 * 20 + 4 * 20 * 20 * 20);
    REQUIRE(p[0] == 4);
    REQUIRE(p[1] == 15);
    REQUIRE(p[2] == 13);
    REQUIRE(p[3] == 7);
  }

  SECTION("Index manipulation, z-order") {
    idx_zorder_t<3> idx(index_t<3>(3, 5, 7), extent_t<3>(32, 32, 32));

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
    auto p = a.get_ptr();

    auto idx = p.idx_at(100, a.extent());
    REQUIRE(idx.linear == 100);
    REQUIRE(idx.strides[1] == 30);

    auto pos = idx.get_pos();
    REQUIRE(pos[0] == (100 % 30));
    REQUIRE(pos[1] == (100 / 30));

    SECTION("ndptr with Z-order indexing") {
      // multi_array<float, MemoryModel::host_only, idx_zorder_t<>>
      // array(
      //     32, 32, 1);
      auto array =
          make_multi_array<float, MemoryModel::host_only, idx_zorder_t>(
              extent(32, 32));
      auto p = array.get_ptr();

      auto idx = p.idx_at(63, a.extent());

      auto pos = idx.get_pos();
      REQUIRE(pos[0] == 7);
      REQUIRE(pos[1] == 7);
      REQUIRE(array.get_idx(pos[0], pos[1]).linear == 63);
    }
  }

  SECTION("range-based indexing") {
    // multi_array<float> a(30, 20);
    auto a = make_multi_array<float, MemoryModel::host_only>(20, 30);

    for (auto n : indices(a)) {
      a[n] = n;
    }

    for (int n = 0; n < a.size(); n++) {
      REQUIRE(a[n] == n);
    }
  }

  SECTION("make_multi_array") {
    auto arr = make_multi_array<float, MemoryModel::host_only,
                                idx_row_major_t>(extent(30, 30, 30));

    REQUIRE(arr.size() == 30 * 30 * 30);
  }

  // SECTION("multi_array returning an index range") {
  //   multi_array<float, MemoryModel::host_only, idx_col_major_t<>> a(
  //       30, 20, 1);

  //   for (auto idx : a.indices()) {
  //     a[idx] = idx.key;
  //   }

  //   for (int n = 0; n < a.size(); n++) {
  //     REQUIRE(a[n] == n);
  //   }

  //   multi_array<float, MemoryModel::host_only, idx_row_major_t<>> b(
  //       1, 20, 30);

  //   for (auto idx : b.indices()) {
  //     b[idx] = idx.key;
  //   }

  //   for (int n = 0; n < b.size(); n++) {
  //     REQUIRE(b[n] == n);
  //   }

  //   multi_array<float, MemoryModel::host_only, idx_zorder_t<>> c(
  //       32, 32, 32);

  //   for (auto idx : c.indices()) {
  //     c[idx] = idx.key;
  //   }

  //   for (int n = 0; n < c.size(); n++) {
  //     REQUIRE(c[n] == n);
  //   }
  // }
}

TEST_CASE("Performance of interpolation on CPU",
          "[multi_array][interpolation]") {
  uint32_t N = 128;
  uint32_t N1 = N, N2 = N, N3 = N;
  std::default_random_engine g;
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::uniform_int_distribution<uint32_t> cell_dist(0, N1 * N2 * N3);

  auto ext = extent(N1, N2, N3);

  auto v1 =
      make_multi_array<float, MemoryModel::host_only, idx_col_major_t>(
          ext);
  auto v2 =
      make_multi_array<float, MemoryModel::host_only, idx_zorder_t>(
          ext);

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
  buffer_t<float, MemoryModel::host_only> xs(M);
  buffer_t<float, MemoryModel::host_only> ys(M);
  buffer_t<float, MemoryModel::host_only> zs(M);
  buffer_t<float, MemoryModel::host_only> result1(M);
  buffer_t<float, MemoryModel::host_only> result2(M);
  buffer_t<uint32_t, MemoryModel::host_only> cells1(M);
  buffer_t<uint32_t, MemoryModel::host_only> cells2(M);
  for (int n = 0; n < M; n++) {
    xs[n] = dist(g);
    ys[n] = dist(g);
    zs[n] = dist(g);
    cells1[n] = cell_dist(g);
    auto pos = v1.idx_at(cells1[n]).get_pos();
    auto idx = v2.get_idx(pos[0], pos[1], pos[2]);
    cells2[n] = idx.linear;
    result1[n] = 0.0f;
    result2[n] = 0.0f;
  }
  std::sort(cells1.host_ptr(), cells1.host_ptr() + cells1.size());
  std::sort(cells2.host_ptr(), cells2.host_ptr() + cells2.size());

  auto interp_kernel = [N1, N2, N3, M](const auto& f, float* result,
                                       float* xs, float* ys, float* zs,
                                       uint32_t* cells, auto ext) {
    for (uint32_t i : range(0, M)) {
      uint32_t cell = cells[i];
      auto idx = f.idx_at(cell);
      auto pos = idx.get_pos();
      if (pos[0] < N1 - 1 && pos[1] < N2 - 1 && pos[2] < N3 - 1) {
        // result[i] = x;
        result[i] = lerp3(f, xs[i], ys[i], zs[i], idx);
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
}
