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

#include "buffer_impl.hpp"
#include "core/gpu_translation_layer.h"
#include "core/gpu_error_check.h"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

namespace Aperture {

template <typename T>
void
ptr_assign_dev(T* array, size_t start, size_t end, const T& value) {
  auto ptr = thrust::device_pointer_cast(array);
  thrust::fill(ptr + start, ptr + end, value);
  GpuSafeCall(gpuDeviceSynchronize());
}

template <typename T>
void
ptr_copy_dev(T* src, T* dst, size_t num, size_t src_pos, size_t dst_pos) {
  auto src_ptr = thrust::device_pointer_cast(src);
  auto dst_ptr = thrust::device_pointer_cast(dst);
  thrust::copy(src_ptr + src_pos, src_ptr + src_pos + num,
               dst_ptr + dst_pos);
  GpuSafeCall(gpuDeviceSynchronize());
}


template void ptr_assign_dev(int*, size_t, size_t, const int&);
template void ptr_assign_dev(long*, size_t, size_t, const long&);
template void ptr_assign_dev(uint32_t*, size_t, size_t, const uint32_t&);
template void ptr_assign_dev(uint64_t*, size_t, size_t, const uint64_t&);
template void ptr_assign_dev(float*, size_t, size_t, const float&);
template void ptr_assign_dev(double*, size_t, size_t, const double&);

template void ptr_copy_dev(int*, int*, size_t, size_t, size_t);
template void ptr_copy_dev(long*, long*, size_t, size_t, size_t);
template void ptr_copy_dev(uint32_t*, uint32_t*, size_t, size_t, size_t);
template void ptr_copy_dev(uint64_t*, uint64_t*, size_t, size_t, size_t);
template void ptr_copy_dev(float*, float*, size_t, size_t, size_t);
template void ptr_copy_dev(double*, double*, size_t, size_t, size_t);

}
