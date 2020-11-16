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

#include "catch.hpp"
#include "core/multi_array.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/logger.h"

#ifdef CUDA_ENABLED

#include <cusparse.h>

using namespace Aperture;

TEST_CASE("Simple test case using cusparse to solve a tri-diagonal",
          "[cusparse]") {
  cusparseHandle_t handle;

  cusparseCreate(&handle);

  uint32_t N1 = 512, N2 = 32;
  extent_t<2> ext(N1, N2);
  multi_array<float, 2> rhs(ext, MemType::host_device);
  multi_array<float, 2> compare(ext, MemType::host_device);
  using idx_t = multi_array<float, 2>::idx_t;

  buffer<float> d(N1, MemType::host_device);
  buffer<float> du(N1, MemType::host_device);
  buffer<float> dl(N1, MemType::host_device);

  kernel_launch(
      [N1] __device__(auto d, auto du, auto dl) {
        for (auto n : grid_stride_range(0, N1)) {
          d[n] = 1.0f;
          du[n] = -0.5f;
          dl[n] = -0.3f;
        }
      },
      d.dev_ptr(), du.dev_ptr(), dl.dev_ptr());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  kernel_launch(
      [N1, N2] __device__(auto rhs, auto compare) {
        extent_t<2> ext(N1, N2);
        for (auto idx :
             grid_stride_range(idx_t(0, ext), idx_t(ext.size(), ext))) {
          auto pos = get_pos(idx, ext);
          rhs[idx] = 0.4f * pos[0] + pos[1];
          // rhs[idx] = 0.4f * pos[0];
          compare[idx] = 0.4f * pos[0] + pos[1];
          // compare[idx] = 0.4f * pos[0];
        }
      },
      rhs.dev_ndptr(), compare.dev_ndptr());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
  compare.copy_to_host();

  // Actually do the tri-diagonal solve here:
  size_t buffer_size;
  cusparseSgtsv2_bufferSizeExt(handle, N1, N2, dl.dev_ptr(), d.dev_ptr(),
                               du.dev_ptr(), rhs.dev_ptr(), N1, &buffer_size);
  Logger::print_info("buffer size is {}", buffer_size);

  buffer<char> tmp(buffer_size, MemType::device_only);

  auto status = cusparseSgtsv2(handle, N1, N2, dl.dev_ptr(), d.dev_ptr(), du.dev_ptr(), rhs.dev_ptr(), N1, tmp.dev_ptr());
  CudaSafeCall(cudaDeviceSynchronize());
  REQUIRE(status == CUSPARSE_STATUS_SUCCESS);
  // Logger::print_info("Solver status is {}", status);

  rhs.copy_to_host();
  for (auto idx : rhs.indices()) {
    auto pos = get_pos(idx, ext);
    if (pos[0] > 1 && pos[0] < N1 - 1) {
      CHECK(1.0f * rhs[idx] - 0.3f * rhs[idx.dec_x()] - 0.5f * rhs[idx.inc_x()] == Approx(compare[idx]));
    }
  }

  cusparseDestroy(handle);
}

#endif
