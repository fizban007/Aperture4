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
#include <algorithm>
#include <cstdint>

namespace Aperture {

template <typename T>
void
ptr_assign(exec_tags::host, T* array, size_t start, size_t end, const T& value) {
  std::fill(array + start, array + end, value);
}

template <typename T>
void
ptr_copy(exec_tags::host, T* src, T* dst, size_t num, size_t src_pos, size_t dst_pos) {
  std::copy(src + src_pos, src + src_pos + num, dst + dst_pos);
}

template void ptr_assign(exec_tags::host, int*, size_t, size_t, const int&);
template void ptr_assign(exec_tags::host, long*, size_t, size_t, const long&);
template void ptr_assign(exec_tags::host, uint32_t*, size_t, size_t, const uint32_t&);
template void ptr_assign(exec_tags::host, uint64_t*, size_t, size_t, const uint64_t&);
template void ptr_assign(exec_tags::host, float*, size_t, size_t, const float&);
template void ptr_assign(exec_tags::host, double*, size_t, size_t, const double&);

template void ptr_copy(exec_tags::host, int*, int*, size_t, size_t, size_t);
template void ptr_copy(exec_tags::host, long*, long*, size_t, size_t, size_t);
template void ptr_copy(exec_tags::host, uint32_t*, uint32_t*, size_t, size_t, size_t);
template void ptr_copy(exec_tags::host, uint64_t*, uint64_t*, size_t, size_t, size_t);
template void ptr_copy(exec_tags::host, float*, float*, size_t, size_t, size_t);
template void ptr_copy(exec_tags::host, double*, double*, size_t, size_t, size_t);

}  // namespace Aperture
