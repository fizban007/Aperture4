/*
 * Copyright (c) 2022 Alex Chen.
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

#pragma once

#include "gpu_translation_layer.h"

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
#define GPU_ERROR_CHECK  //!< Defines whether to check error
#define GpuSafeCall(err) \
  __gpuSafeCall(         \
      err, __FILE__,      \
      __LINE__)  //!< Wrapper to allow display of file and line number
#define CudaSafeCall(err) GpuSafeCall(err)
#define GpuCheckError() \
  __gpuCheckError(      \
      __FILE__,          \
      __LINE__)  //!< Wrapper to allow display of file and line number
#define CudaCheckError() GpuCheckError()

#include <stdio.h>

///  Checks last kernel launch error.
inline void
__gpuCheckError(const char *file, const int line) {
#ifdef GPU_ERROR_CHECK
  gpuError_t err = gpuGetLastError();
  if (gpuSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
            gpuGetErrorString(err));
    std::exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
#ifndef NDEBUG
  err = gpuDeviceSynchronize();
  if (gpuSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file,
            line, gpuGetErrorString(err));
    std::exit(-1);
  }
#endif

#endif
}

///  Checks memory allocation error
inline void
__gpuSafeCall(gpuError_t err, const char *file, const int line) {
#ifdef GPU_ERROR_CHECK
  if (gpuSuccess != err) {
    fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
            gpuGetErrorString(err));
    // gpuGetLastError();
    // exit(-1);
    throw(gpuGetErrorString(err));
  }
#endif
}
#else
#define GpuSafeCall(err) err
#define GpuCheckError()
#endif
