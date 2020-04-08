#include "catch.hpp"
#include "core/multi_array.hpp"
#include "core/ndptr.hpp"
#include "utils/logger.h"
// #include "utils/ndptr.h"
#include "utils/range.hpp"
#include "utils/timer.h"
#include "utils/kernel_helper.hpp"
#include <random>
#include <thrust/tuple.h>

using namespace Aperture;

#ifdef CUDA_ENABLED

template <typename T, typename Index_t>
HOST_DEVICE T
interp(T* f, T x, T y, T z, const Index_t& idx) {
  T f11 = (1.0 - z) * f[idx.incX().incY().key] +
          z * f[idx.incX().incY().incZ().key];
  T f10 = (1.0 - z) * f[idx.incX().key] + z * f[idx.incX().incZ().key];
  T f01 = (1.0 - z) * f[idx.incY().key] + z * f[idx.incY().incZ().key];
  T f00 = (1.0 - z) * f[idx.key] + z * f[idx.incZ().key];
  T f1 = y * f11 + (1.0 - y) * f10;
  T f0 = y * f01 + (1.0 - y) * f00;
  return x * f1 + (1.0 - x) * f0;
}

TEST_CASE("Invoking kernels on multi_array", "[multi_array][kernel]") {
  uint32_t N1 = 100, N2 = 300;
  auto ext = extent(N1, N2);
  auto array = make_multi_array<float, MemoryModel::host_device>(ext);
  REQUIRE(array.host_allocated() == true);
  REQUIRE(array.dev_allocated() == true);

  kernel_launch(
      [] __device__ (auto p, float value, auto ext) {
        for (auto idx : grid_stride_range(0u, ext.size())) {
          p[idx] = value;
        }
      }, array.get_ptr(), 3.0f, ext);
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
  // auto array = make_multi_array<float, MemoryModel::device_managed, idx_zorder_t>(ext);
  auto array = make_multi_array<float, MemoryModel::device_managed, idx_row_major_t>(ext);

  // assign_idx_array<<<128, 512>>>(array.get_ptr(), ext);
  kernel_launch(
      [] __device__(auto p, auto ext) {
        for (auto i : grid_stride_range(0u, ext.size())) {
          auto idx = p.idx_at(i, ext);
          auto pos = idx.get_pos();
          p[i] = pos[0] * pos[1];
        }
      }, array.get_ptr(), ext);
  CudaSafeCall(cudaDeviceSynchronize());

  for (auto idx : array.indices()) {
    auto pos = idx.get_pos();
    REQUIRE(array[idx] == Approx((float)pos[0] * pos[1]));
  }
}

// TEST_CASE("Performance of different indexing schemes",
//           "[multi_array][kernel]") {
//   std::default_random_engine g;
//   std::uniform_real_distribution<double> dist(0.0, 1.0);

//   uint32_t N1 = 256, N2 = 256, N3 = 256;
//   Extent ext(N1, N2, N3);
//   // multi_array<float, idx_row_major_t<>> array(
//   multi_array<float, idx_col_major_t<>> v1(ext,
//                                            MemoryModel::device_managed);
//   multi_array<float, idx_zorder_t<>> v2(ext,
//                                         MemoryModel::device_managed);

//   uint32_t x, y, z;
//   for (auto idx : v1.indices()) {
//     idx.decode(x, y, z);
//     v1[idx] = x + y - z;
//   }
//   for (auto idx : v2.indices()) {
//     idx.decode(x, y, z);
//     v2[idx] = x + y - z;
//   }

//   // Generate M random numbers
//   int M = 100000;
//   buffer_t<float> xs(M, MemoryModel::device_managed);
//   buffer_t<float> ys(M, MemoryModel::device_managed);
//   buffer_t<float> zs(M, MemoryModel::device_managed);
//   buffer_t<float> result(M, MemoryModel::device_managed);
//   for (int n = 0; n < M; n++) {
//     xs[n] = dist(g);
//     ys[n] = dist(g);
//     zs[n] = dist(g);
//   }

//   auto interp_kernel = [N1, N2, N3, M] __device__(
//                            auto f, float* result, float* xs, float* ys,
//                            float* zs, Extent ext) {
//     float dx = 1.0f / N1;
//     float dy = 1.0f / N2;
//     float dz = 1.0f / N3;
//     for (int i : grid_stride_range(0, M)) {
//       uint32_t nx = floor(xs[i] / dx);
//       uint32_t ny = floor(ys[i] / dy);
//       uint32_t nz = floor(zs[i] / dz);
//       if (nx > 0 && nx < N1 - 1 && ny > 0 && ny < N2 - 1 && nz > 0 &&
//           nz < N3 - 1) {
//         float x = (xs[i] - nx * dx) / dx;
//         float y = (ys[i] - ny * dy) / dy;
//         float z = (zs[i] - nz * dz) / dz;
//         result[i] =
//             interp(f.p, x, y, z,
//                    typename decltype(f)::idx_type(nx, ny, nz, ext));
//       }
//     }
//   };

//   timer::stamp();
//   kernel_launch(interp_kernel, v1.get_ptr(), result.dev_ptr(),
//                 xs.dev_ptr(), ys.dev_ptr(), zs.dev_ptr(), ext);
//   timer::show_duration_since_stamp("normal indexing", "ms");

//   timer::stamp();
//   kernel_launch(interp_kernel, v2.get_ptr(), result.dev_ptr(),
//                 xs.dev_ptr(), ys.dev_ptr(), zs.dev_ptr(), ext);
//   timer::show_duration_since_stamp("morton indexing", "ms");
// }

#endif
