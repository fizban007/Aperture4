#include "catch.hpp"
#include "core/constant_mem_func.h"
#include "core/constant_mem.h"
#include "core/multi_array.hpp"
#include "core/ndptr.hpp"
#include "utils/logger.h"
#include "utils/interpolation.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include "utils/timer.h"
#include <algorithm>
#include <random>
#include <thrust/tuple.h>

using namespace Aperture;

#ifdef CUDA_ENABLED

template <typename Ptr, typename Index>
HOST_DEVICE float
finite_diff(const Ptr f, const Index& idx) {
  return f[idx.template inc<2>()] - f[idx.template inc<2>(-1)];
}

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
  auto array = make_multi_array<float,
                                idx_zorder_t>(ext, MemType::device_managed);
  // auto array = make_multi_array<float, MemType::device_managed,
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
          "[multi_array][performance][kernel][.]") {
  init_morton(morton2dLUT, morton3dLUT);
  uint32_t N = 128;
  uint32_t N1 = N, N2 = N, N3 = N;
  std::default_random_engine g;
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::uniform_int_distribution<uint32_t> cell_dist(0, N1 * N2 * N3);

  auto ext = extent(N1, N2, N3);
  // multi_array<float, idx_row_major_t<>> array(
  auto v1 = make_multi_array<float,
                             idx_col_major_t>(ext, MemType::host_device);
  auto v2 =
      make_multi_array<float, idx_zorder_t>(
          ext, MemType::host_device);

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
    for (uint32_t i : grid_stride_range(0, M)) {
      uint32_t cell = cells[i];
      auto idx = f.idx_at(cell, ext);
      auto pos = idx.get_pos();
      if (pos[0] < N1 - 1 && pos[1] < N2 - 1 && pos[2] < N3 - 1) {
        // result[i] = x;
        result[i] = lerp3(f, xs[i], ys[i], zs[i], idx);
      }
    }
  };

  cudaDeviceSynchronize();

  timer::stamp();
  kernel_launch(interp_kernel, v1.get_const_ptr(), result1.dev_ptr(),
                xs.dev_ptr(), ys.dev_ptr(), zs.dev_ptr(),
                cells1.dev_ptr(), ext);
  cudaDeviceSynchronize();
  timer::show_duration_since_stamp("normal indexing", "us");

  timer::stamp();
  kernel_launch(interp_kernel, v2.get_const_ptr(), result2.dev_ptr(),
                xs.dev_ptr(), ys.dev_ptr(), zs.dev_ptr(),
                cells2.dev_ptr(), ext);
  cudaDeviceSynchronize();
  timer::show_duration_since_stamp("morton indexing", "us");

  result1.copy_to_host();
  result2.copy_to_host();

  for (auto idx : range(0ul, result1.size())) {
    REQUIRE(result1[idx] == result2[idx]);
  }
}

TEST_CASE("Assign and copy on device", "[multi_array][kernel]") {
  auto v1 = make_multi_array<float>(30, 30);
  auto v2 = make_multi_array<float>(30, 30);

  v1.assign(3.0f);
  v1.copy_to_host();
  for (auto idx : v1.indices()) {
    REQUIRE(v1[idx] == 3.0f);
  }
}


#endif
