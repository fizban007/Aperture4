#include "catch.hpp"
#include "core/constant_mem_func.h"
#include "core/multi_array.hpp"
#include "core/ndptr.hpp"
#include "utils/logger.h"
// #include "utils/ndptr.h"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include "utils/timer.h"
#include <algorithm>
#include <random>
#include <thrust/tuple.h>

using namespace Aperture;

#ifdef CUDA_ENABLED

template <typename Index>
HOST_DEVICE float
interp(float* f, float x, float y, float z, const Index& idx) {
  float f11 =
      (1.0 - z) * f[idx.template inc<0>().template inc<1>().linear] +
      z * f[idx.template inc<0>()
                .template inc<1>()
                .template inc<2>()
                .linear];
  float f10 = (1.0 - z) * f[idx.template inc<0>().linear] +
              z * f[idx.template inc<0>().template inc<2>().linear];
  float f01 = (1.0 - z) * f[idx.template inc<1>().linear] +
              z * f[idx.template inc<1>().template inc<2>().linear];
  float f00 =
      (1.0 - z) * f[idx.linear] + z * f[idx.template inc<2>().linear];
  float f1 = y * f11 + (1.0 - y) * f10;
  float f0 = y * f01 + (1.0 - y) * f00;
  return x * f1 + (1.0 - x) * f0;
}

template <typename Ptr, typename Index>
HOST_DEVICE
float finite_diff(const Ptr f, const Index& idx) {
  return f[idx.template inc<2>()] - f[idx.template inc<2>(-1)];
}

// HOST_DEVICE
// float interp(float* f, float x, float y, float z, const
// idx_zorder_t<3>& idx) {
//   float f11 = (1.0 - z) * f[idx.inc<0>().inc<1>().linear] +
//           z * f[idx.inc<0>().inc<1>().inc<2>().linear];
//   float f10 = (1.0 - z) * f[idx.inc<0>().linear] + z *
//   f[idx.inc<0>().inc<2>().linear]; float f01 = (1.0 - z) *
//   f[idx.inc<1>().linear] + z * f[idx.inc<1>().inc<2>().linear]; float
//   f00 = (1.0 - z) * f[idx.linear] + z * f[idx.inc<2>().linear]; float
//   f1 = y * f11 + (1.0 - y) * f10; float f0 = y * f01 + (1.0 - y) *
//   f00; return x * f1 + (1.0 - x) * f0;
// }

TEST_CASE("Invoking kernels on multi_array", "[multi_array][kernel]") {
  uint32_t N1 = 100, N2 = 300;
  auto ext = extent(N1, N2);
  auto array = make_multi_array<float, MemoryModel::host_device>(ext);
  REQUIRE(array.host_allocated() == true);
  REQUIRE(array.dev_allocated() == true);

  kernel_launch(
      [] __device__(auto p, float value, auto ext) {
        for (auto idx : grid_stride_range(0u, ext.size())) {
          p[idx] = value;
        }
      },
      array.get_ptr(), 3.0f, ext);
  CudaSafeCall(cudaDeviceSynchronize());

  array.copy_to_host();

  for (auto idx : array.indices()) {
    REQUIRE(array[idx] == 3.0f);
  }
}

TEST_CASE("Different indexing on multi_array",
          "[multi_array][kernel]") {
  Logger::init(0, LogLevel::debug);
  uint32_t N1 = 32, N2 = 32;
  // Extent ext(1, N2, N1);
  auto ext = extent(N2, N1);
  // multi_array<float, idx_row_major_t<>> array(
  auto array = make_multi_array<float, MemoryModel::device_managed,
                                idx_zorder_t>(ext);
  // auto array = make_multi_array<float, MemoryModel::device_managed,
  // idx_row_major_t>(ext);

  // assign_idx_array<<<128, 512>>>(array.get_ptr(), ext);
  kernel_launch(
      [] __device__(auto p, auto ext) {
        for (auto i : grid_stride_range(0u, ext.size())) {
          auto idx = p.idx_at(i, ext);
          auto pos = idx.get_pos();
          p[i] = pos[0] * pos[1];
        }
      },
      array.get_ptr(), ext);
  CudaSafeCall(cudaDeviceSynchronize());

  for (auto idx : array.indices()) {
    auto pos = idx.get_pos();
    REQUIRE(array[idx] == Approx((float)pos[0] * pos[1]));
  }
}

TEST_CASE("Performance of different indexing schemes",
          "[multi_array][kernel]") {
  uint32_t N1 = 256, N2 = 256, N3 = 256;
  std::default_random_engine g;
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::uniform_int_distribution<uint32_t> cell_dist(0, 256 * 256 * 256);

  auto ext = extent(N1, N2, N3);
  // multi_array<float, idx_row_major_t<>> array(
  auto v1 = make_multi_array<float, MemoryModel::device_managed,
                             idx_col_major_t>(ext);
  auto v2 = make_multi_array<float, MemoryModel::device_managed,
                             idx_col_major_t>(ext);

  for (auto idx : v1.indices()) {
    auto pos = idx.get_pos();
    v1[idx] = pos[0] + pos[1] - pos[2];
  }
  for (auto idx : v2.indices()) {
    auto pos = idx.get_pos();
    v2[idx] = pos[0] + pos[1] - pos[2];
  }

  // Generate M random numbers
  int M = 100000;
  buffer_t<float, MemoryModel::host_device> xs(M);
  buffer_t<float, MemoryModel::host_device> ys(M);
  buffer_t<float, MemoryModel::host_device> zs(M);
  buffer_t<float, MemoryModel::host_device> result(M);
  buffer_t<uint32_t, MemoryModel::host_device> cells(M);
  for (int n = 0; n < M; n++) {
    xs[n] = dist(g);
    ys[n] = dist(g);
    zs[n] = dist(g);
    cells[n] = cell_dist(g);
  }
  xs.copy_to_device();
  ys.copy_to_device();
  zs.copy_to_device();
  cells.copy_to_device();
  // std::sort(cells.host_ptr(), cells.host_ptr() + cells.size());

  auto interp_kernel = [N1, N2, N3, M] __device__(
                           auto f, float* result, float* xs, float* ys,
                           float* zs, uint32_t* cells, auto ext) {
    for (int i : grid_stride_range(0, M)) {
      uint32_t cell = cells[i];
      auto idx = f.idx_at(cell, ext);
      auto pos = idx.get_pos();
      if (pos[0] > 0 && pos[0] < N1 - 1 && pos[1] > 0 &&
          pos[1] < N2 - 1 && pos[2] > 0 && pos[2] < N3 - 1) {
        // result[i] = x;
        result[i] = interp(f.p, xs[i], ys[i], zs[i], idx);
      }
    }
  };

  auto diff_kernel = [N1, N2, N3] __device__(auto f, auto ext) {
  };

  timer::stamp();
  kernel_launch(interp_kernel, v1.get_ptr(), result.dev_ptr(),
                xs.dev_ptr(), ys.dev_ptr(), zs.dev_ptr(),
                cells.dev_ptr(), ext);
  cudaDeviceSynchronize();
  timer::show_duration_since_stamp("normal indexing", "us");

  timer::stamp();
  kernel_launch(interp_kernel, v2.get_ptr(), result.dev_ptr(),
                xs.dev_ptr(), ys.dev_ptr(), zs.dev_ptr(),
                cells.dev_ptr(), ext);
  cudaDeviceSynchronize();
  timer::show_duration_since_stamp("morton indexing", "us");
}

#endif
