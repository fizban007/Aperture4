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
#include "core/constant_mem.h"
#include "core/constant_mem_func.h"
#include "core/detail/multi_array_helpers.h"
#include "core/multi_array.hpp"
#include "core/multi_array_exp.hpp"
#include "core/ndptr.hpp"
#include "core/ndsubset.hpp"
#include "core/ndsubset_dev.hpp"
#include "utils/interpolation.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/logger.h"
#include "utils/range.hpp"
#include "utils/timer.h"
#include <algorithm>
#include <random>
#include <thrust/tuple.h>

using namespace Aperture;

#ifdef GPU_ENABLED

TEST_CASE("Invoking kernels on multi_array", "[multi_array][kernel]") {
  uint32_t N1 = 100, N2 = 300;
  auto ext = extent(N1, N2);
  auto array = make_multi_array<float>(ext, MemType::host_device);
  REQUIRE(array.host_allocated() == true);
  REQUIRE(array.dev_allocated() == true);

  kernel_launch(
      [] __device__(auto p, float value, auto ext) {
        for (auto idx : grid_stride_range(0u, ext.size())) {
          p[idx] = value;
        }
      },
      array.dev_ndptr(), 3.0f, ext);
  GpuSafeCall(gpuDeviceSynchronize());

  array.copy_to_host();

  for (auto idx : array.indices()) {
    REQUIRE(array[idx] == 3.0f);
  }
}

TEST_CASE("Different indexing on multi_array", "[multi_array][kernel]") {
  Logger::init(0, LogLevel::debug);
  uint32_t N1 = 32, N2 = 32;
  // Extent ext(1, N2, N1);
  auto ext = extent(N2, N1);
  // multi_array<float, idx_row_major_t<>> array(
  auto array =
      make_multi_array<float, idx_zorder_t>(ext, MemType::device_managed);
  // auto array = make_multi_array<float, MemType::device_managed,
  // idx_row_major_t>(ext);

  // assign_idx_array<<<128, 512>>>(array.dev_ndptr(), ext);
  kernel_launch(
      [] __device__(auto p, auto ext) {
        for (auto i : grid_stride_range(0u, ext.size())) {
          auto idx = p.idx_at(i, ext);
          // auto pos = idx.get_pos();
          auto pos = get_pos(idx, ext);
          p[i] = pos[0] * pos[1];
        }
      },
      array.dev_ndptr(), ext);
  GpuSafeCall(gpuDeviceSynchronize());

  for (auto idx : array.indices()) {
    // auto pos = idx.get_pos();
    auto pos = get_pos(idx, ext);
    REQUIRE(array[idx] == Catch::Approx((float)pos[0] * pos[1]));
  }
}

TEST_CASE("Performance of different indexing schemes",
          "[multi_array][performance][kernel][.]") {
  init_morton(morton2dLUT, morton3dLUT);
  uint32_t N = 128;
  uint32_t N1 = N, N2 = N, N3 = N;
  std::default_random_engine g;
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::uniform_int_distribution<uint32_t> cell_dist(0, N1 * N2 * N3);

  auto ext = extent(N1, N2, N3);
  // multi_array<float, idx_row_major_t<>> array(
  auto v1 = make_multi_array<float, idx_col_major_t>(ext, MemType::host_device);
  auto v2 = make_multi_array<float, idx_zorder_t>(ext, MemType::host_device);

  for (auto idx : v1.indices()) {
    auto pos = get_pos(idx, ext);
    v1[idx] = float(0.3 * pos[0] + 0.4 * pos[1] - pos[2]);
  }
  for (auto idx : v2.indices()) {
    // auto pos = idx.get_pos();
    auto pos = get_pos(idx, ext);
    v2[idx] = float(0.3 * pos[0] + 0.4 * pos[1] - pos[2]);
  }
  for (auto idx : v1.indices()) {
    // auto pos = idx.get_pos();
    auto pos = get_pos(idx, ext);
    REQUIRE(v1(pos[0], pos[1], pos[2]) == v2(pos[0], pos[1], pos[2]));
  }
  v1.copy_to_device();
  v2.copy_to_device();

  // Generate M random numbers
  int M = 1000000;
  buffer<float> xs(M, MemType::host_device);
  buffer<float> ys(M, MemType::host_device);
  buffer<float> zs(M, MemType::host_device);
  buffer<float> result1(M, MemType::host_device);
  buffer<float> result2(M, MemType::host_device);
  buffer<uint32_t> cells1(M, MemType::host_device);
  buffer<uint32_t> cells2(M, MemType::host_device);
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
  // std::sort(cells1.host_ptr(), cells1.host_ptr() + cells1.size());
  // std::sort(cells2.host_ptr(), cells2.host_ptr() + cells2.size());
  xs.copy_to_device();
  ys.copy_to_device();
  zs.copy_to_device();
  cells1.copy_to_device();
  cells2.copy_to_device();
  result1.copy_to_device();
  result2.copy_to_device();

  auto interp_kernel = [N1, N2, N3, M] __device__(
                           auto f, float* result, float* xs, float* ys,
                           float* zs, uint32_t* cells, auto ext) {
    auto interp = interpolator<bspline<1>, 3>{};
    for (uint32_t i : grid_stride_range(0, M)) {
      uint32_t cell = cells[i];
      auto idx = f.idx_at(cell, ext);
      // auto pos = idx.get_pos();
      auto pos = get_pos(idx, ext);
      if (pos[0] < N1 - 1 && pos[1] < N2 - 1 && pos[2] < N3 - 1) {
        // result[i] = x;
        // result[i] = lerp3(f, xs[i], ys[i], zs[i], idx);
        result[i] = interp(f, vec_t<float, 3>(xs[i], ys[i], zs[i]), idx);
      }
    }
  };

  GpuSafeCall(gpuDeviceSynchronize());
  Logger::print_info(
      "Measuring interpolation speed with different indexing schemes on "
      "Device");

  timer::stamp();
  kernel_launch(interp_kernel, v1.dev_ndptr_const(), result1.dev_ptr(),
                xs.dev_ptr(), ys.dev_ptr(), zs.dev_ptr(), cells1.dev_ptr(),
                ext);
  GpuSafeCall(gpuDeviceSynchronize());
  timer::show_duration_since_stamp("normal indexing", "us");

  timer::stamp();
  kernel_launch(interp_kernel, v2.dev_ndptr_const(), result2.dev_ptr(),
                xs.dev_ptr(), ys.dev_ptr(), zs.dev_ptr(), cells2.dev_ptr(),
                ext);
  GpuSafeCall(gpuDeviceSynchronize());
  timer::show_duration_since_stamp("morton indexing", "us");

  result1.copy_to_host();
  result2.copy_to_host();

  for (auto idx : range(0ul, result1.size())) {
    REQUIRE(result1[idx] == result2[idx]);
  }
}

TEST_CASE("Assign and copy on device", "[multi_array][kernel]") {
  auto v1 = make_multi_array<float>(extent(30, 30));
  auto v2 = make_multi_array<float>(extent(30, 30));

  v1.assign_dev(3.0f);
  v1.copy_to_host();
  for (auto idx : v1.indices()) {
    REQUIRE(v1[idx] == 3.0f);
  }
}

TEST_CASE("Add ndptr on device", "[multi_array][exp_template]") {
  auto ext = extent(30, 30);
  auto v1 = make_multi_array<float>(ext);
  auto v2 = make_multi_array<float>(ext);
  auto v3 = make_multi_array<float>(ext);

  v1.assign_dev(1.0f);
  v2.assign_dev(2.0f);

  kernel_launch(
      {30, 30},
      [ext] __device__(auto p1, auto p2, auto p3) {
        using idx_t = default_idx_t<2>;
        for (auto idx :
             grid_stride_range(idx_t(0, ext), idx_t(ext.size(), ext))) {
          p3[idx] = (p1 * p2)[idx];
        }
      },
      v1.dev_ndptr_const(), v2.dev_ndptr_const(), v3.dev_ndptr());
  GpuSafeCall(gpuDeviceSynchronize());
  GpuCheckError();

  v3.copy_to_host();
  for (auto idx : v3.indices()) {
    REQUIRE(v3[idx] == 2.0f);
  }
}

TEST_CASE("Testing select_dev", "[multi_array][exp_template]") {
  auto v = make_multi_array<float>(extent(30, 30), MemType::host_device);

  v.assign_dev(3.0f);

  auto w = select_dev(v, index(0, 0), extent(10, 10));
  w += select_dev((v * 3.0f + 4.0f), index(0, 0), extent(10, 10));
  v.copy_to_host();

  for (auto idx : v.indices()) {
    auto pos = idx.get_pos();
    if (pos[0] < 10 && pos[1] < 10) {
      REQUIRE(v[idx] == 16.0f);
    } else {
      REQUIRE(v[idx] == 3.0f);
    }
  }

  select_dev(v) = 2.5f;
  REQUIRE(v[0] != 2.5f);
  v.copy_to_host();
  for (auto idx : v.indices()) {
    REQUIRE(v[idx] == 2.5f);
  }

  select_dev(v) = v * 3.0f;
  v.copy_to_host();
  for (auto idx : v.indices()) {
    REQUIRE(v[idx] == 7.5f);
  }

  select(v) = 4.0f * v - 2.0f;
  for (auto idx : v.indices()) {
    REQUIRE(v[idx] == 28.0f);
  }
  v.copy_to_device();

  auto v2 = make_multi_array<float>(extent(30, 30), MemType::host_device);
  v2.assign_dev(0.0f);
  select_dev(v2, index(10, 10), extent(10, 10)) =
      select_dev(v / 7.0f + 1.0f, index(0, 0), extent(10, 10));
  v2.copy_to_host();
  for (auto idx : v.indices()) {
    auto pos = idx.get_pos();
    if (pos[0] >= 10 && pos[0] < 20 && pos[1] >= 10 && pos[1] < 20) {
      REQUIRE(v2[idx] == 5.0f);
    } else {
      REQUIRE(v2[idx] == 0.0f);
    }
  }
}

TEST_CASE("Performance of expression template",
          "[performance][exp_template][.]") {
  uint32_t N = 256;
  // using Conf = Config<3>;
  using idx_t = default_idx_t<3>;
  auto ext = extent(N, N, N);
  auto v1 = make_multi_array<float>(ext);

  v1.assign_dev(3.0f);
  Logger::print_info("Measuring performance of expression templates");

  GpuSafeCall(gpuDeviceSynchronize());

  timer::stamp();
  kernel_launch(
      [ext] __device__(auto p) {
        for (auto idx :
             grid_stride_range(idx_t(0, ext), idx_t(ext.size(), ext))) {
          p[idx] = p[idx] * 7.0f + 9.0f / p[idx];
        }
      },
      v1.dev_ndptr());
  GpuSafeCall(gpuDeviceSynchronize());
  timer::show_duration_since_stamp("Evaluation using Kernel Launch", "us");

  v1.copy_to_host();
  REQUIRE(v1[0] == 24.0f);

  v1.assign_dev(3.0f);

  timer::stamp();
  select_dev(v1) = v1 * 7.0f + 9.0f / v1;
  timer::show_duration_since_stamp("Evaluation using Exp template", "us");
  v1.copy_to_host();
  REQUIRE(v1[0] == 24.0f);

  auto v2 = make_multi_array<float>(ext);
  auto v3 = make_multi_array<float>(ext);
  v2.assign_dev(0.0f);
  v3.assign_dev(0.0f);

  timer::stamp();
  add(ExecDev{}, v2, v1, index(0, 0, 0), index(0, 0, 0), ext);
  timer::show_duration_since_stamp("Copy using copy_dev", "us");
  v2.copy_to_host();

  for (auto idx : v2.indices()) {
    REQUIRE(v2[idx] == v1[idx]);
  }

  timer::stamp();
  select_dev(v3) += v1.cref();
  GpuSafeCall(gpuDeviceSynchronize());
  timer::show_duration_since_stamp("Copy using expression template", "us");
  v3.copy_to_host();

  for (auto idx : v3.indices()) {
    REQUIRE(v3[idx] == v1[idx]);
  }
}

TEST_CASE("Testing resample", "[multi_array]") {
  uint32_t N1 = 8, N2 = 8;
  using idx_t = idx_col_major_t<2>;
  auto ext = extent(N1, N2);
  auto array = make_multi_array<float>(ext, MemType::device_managed);

  kernel_launch([ext] __device__ (auto p) {
      for (auto idx : grid_stride_range(idx_t(0, ext), idx_t(ext.size(), ext))) {
        auto pos = get_pos(idx, ext);
        p[idx] = pos[0] + pos[1];
      }
    }, array.dev_ndptr());
  GpuSafeCall(gpuDeviceSynchronize());

  Logger::print_info("array[3, 2] is {}", array[idx_t(index(3, 2), ext)]);

  int downsample = 4;
  auto ext2 = extent(N1 / downsample, N2 / downsample);
  auto arr2 = make_multi_array<float>(ext2, MemType::device_managed);

  auto offset = index(0, 0);
  resample(ExecDev{}, array, arr2, offset, offset, stagger_t(0b000), stagger_t(0b000), downsample);
  GpuSafeCall(gpuDeviceSynchronize());

  REQUIRE(arr2.dev_ptr()[0] == 48.0f / 16.0f);
  REQUIRE(arr2.dev_ptr()[1] == (88.0f + 24.0f) / 16.0f);
  REQUIRE(arr2.dev_ptr()[2] == (88.0f + 24.0f) / 16.0f);
  REQUIRE(arr2.dev_ptr()[3] == (152.0f + 24.0f) / 16.0f);
}

#endif
