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
#include <algorithm>
#include <cstdint>

namespace Aperture {

template <typename T>
void
ptr_assign(T* array, size_t start, size_t end, const T& value) {
  std::fill(array + start, array + end, value);
}

template <typename T>
void
ptr_copy(T* src, T* dst, size_t num, size_t src_pos, size_t dst_pos) {
  std::copy(src + src_pos, src + src_pos + num, dst + dst_pos);
}

template void ptr_assign(int*, size_t, size_t, const int&);
template void ptr_assign(long*, size_t, size_t, const long&);
template void ptr_assign(uint32_t*, size_t, size_t, const uint32_t&);
template void ptr_assign(uint64_t*, size_t, size_t, const uint64_t&);
template void ptr_assign(float*, size_t, size_t, const float&);
template void ptr_assign(double*, size_t, size_t, const double&);

template void ptr_copy(int*, int*, size_t, size_t, size_t);
template void ptr_copy(long*, long*, size_t, size_t, size_t);
template void ptr_copy(uint32_t*, uint32_t*, size_t, size_t, size_t);
template void ptr_copy(uint64_t*, uint64_t*, size_t, size_t, size_t);
template void ptr_copy(float*, float*, size_t, size_t, size_t);
template void ptr_copy(double*, double*, size_t, size_t, size_t);

#if !defined(CUDA_ENABLED) && !defined(HIP_ENABLED)
template <typename T>
void
ptr_assign_dev(T* array, size_t start, size_t end, const T& value) {}

template <typename T>
void
ptr_copy_dev(T* src, T* dst, size_t num, size_t src_pos, size_t dst_pos) {}

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
#endif

}  // namespace Aperture
