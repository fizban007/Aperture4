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

#ifndef __CUDA_CONTROL_H_
#define __CUDA_CONTROL_H_

#if defined(__CUDACC__) && defined(CUDA_ENABLED)
#define HOST_DEVICE __host__ __device__
#define HD_INLINE __host__ __device__ __forceinline__
#else
#define HOST_DEVICE
#define HD_INLINE inline
#endif

#if defined(CUDA_ENABLED)
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK  //!< Defines whether to check error
#define CudaSafeCall(err) \
  __cudaSafeCall(         \
      err, __FILE__,      \
      __LINE__)  //!< Wrapper to allow display of file and line number
#define CudaCheckError() \
  __cudaCheckError(      \
      __FILE__,          \
      __LINE__)  //!< Wrapper to allow display of file and line number

#include <stdio.h>

///  Checks last kernel launch error.
inline void
__cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
            cudaGetErrorString(err));
    std::exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
#ifndef NDEBUG
  err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file,
            line, cudaGetErrorString(err));
    std::exit(-1);
  }
#endif

#endif
}

///  Checks memory allocation error
inline void
__cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
            cudaGetErrorString(err));
    cudaGetLastError();
    // exit(-1);
    throw(cudaGetErrorString(err));
  }
#endif
}
#else
#define CudaSafeCall(err) err
#define CudaCheckError()
#endif

#endif  // __CUDA_CONTROL_H_
