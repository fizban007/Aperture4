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

#ifndef _GPU_TRANSLATION_LAYER_H_
#define _GPU_TRANSLATION_LAYER_H_

#if (defined(HIP_ENABLED) && defined(__HIPCC__)) || \
    (defined(CUDA_ENABLED) && defined(__CUDACC__))
#define HOST_DEVICE __host__ __device__
#define HD_INLINE __host__ __device__ __forceinline__
#define LAMBDA __device__
#else
#define HOST_DEVICE
#define HD_INLINE inline
#define LAMBDA
#endif

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
#include <cstdlib>
#if defined(HIP_ENABLED)
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
using gpuError_t = hipError_t;
using gpuStream_t = hipStream_t;
using gpuDeviceProp_t = hipDeviceProp_t;
using gpuFuncAttributes = hipFuncAttributes;

#define gpuSuccess hipSuccess
#define gpuDeviceSynchronize() hipDeviceSynchronize()
#define gpuGetLastError() hipGetLastError()
#define gpuGetErrorString(err) hipGetErrorString(err)
#define gpuGetDevice(devId) hipGetDevice(devId)
#define gpuGetDeviceProperties(prop, id) hipGetDeviceProperties(prop, id)
#define gpuFuncGetAttributes(attrib, f) hipFuncGetAttributes(attrib, f)

#elif defined(CUDA_ENABLED)
#include <cuda_runtime.h>
using gpuError_t = cudaError;
using gpuStream_t = cudaStream_t;
using gpuDeviceProp_t = cudaDeviceProp;
using gpuFuncAttributes = cudaFuncAttributes;

#define gpuSuccess cudaSuccess
#define gpuDeviceSynchronize() cudaDeviceSynchronize()
#define gpuGetLastError() cudaGetLastError()
#define gpuGetErrorString(err) cudaGetErrorString(err)
#define gpuGetDevice(devId) cudaGetDevice(devId)
#define gpuGetDeviceProperties(prop, id) cudaGetDeviceProperties(prop, id)
#define gpuFuncGetAttributes(attrib, f) cudaFuncGetAttributes(attrib, f)

#endif
#endif

#endif  // _GPU_TRANSLATION_LAYER_H_
