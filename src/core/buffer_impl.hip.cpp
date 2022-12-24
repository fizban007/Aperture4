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

#include "buffer.hpp"
#include "core/gpu_translation_layer.h"
#include "core/gpu_error_check.h"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

namespace Aperture {

template <typename T>
void
ptr_assign(T* array, size_t start, size_t end, const T& value, ExecDev) {
  auto ptr = thrust::device_pointer_cast(array);
  thrust::fill(ptr + start, ptr + end, value);
  GpuSafeCall(gpuDeviceSynchronize());
}

template <typename T>
void
ptr_copy(T* src, T* dst, size_t num, size_t src_pos, size_t dst_pos, ExecDev) {
  auto src_ptr = thrust::device_pointer_cast(src);
  auto dst_ptr = thrust::device_pointer_cast(dst);
  thrust::copy(src_ptr + src_pos, src_ptr + src_pos + num,
               dst_ptr + dst_pos);
  GpuSafeCall(gpuDeviceSynchronize());
}


template void ptr_assign(int*, size_t, size_t, const int&, ExecDev);
template void ptr_assign(long*, size_t, size_t, const long&, ExecDev);
template void ptr_assign(uint32_t*, size_t, size_t, const uint32_t&, ExecDev);
template void ptr_assign(uint64_t*, size_t, size_t, const uint64_t&, ExecDev);
template void ptr_assign(float*, size_t, size_t, const float&, ExecDev);
template void ptr_assign(double*, size_t, size_t, const double&, ExecDev);

template void ptr_copy(int*, int*, size_t, size_t, size_t, ExecDev);
template void ptr_copy(long*, long*, size_t, size_t, size_t, ExecDev);
template void ptr_copy(uint32_t*, uint32_t*, size_t, size_t, size_t, ExecDev);
template void ptr_copy(uint64_t*, uint64_t*, size_t, size_t, size_t, ExecDev);
template void ptr_copy(float*, float*, size_t, size_t, size_t, ExecDev);
template void ptr_copy(double*, double*, size_t, size_t, size_t, ExecDev);

}
