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

#ifndef __BUFFER_IMPL_H_
#define __BUFFER_IMPL_H_

#include <cstdlib>

namespace Aperture {

template <typename T>
void ptr_assign(T* array, size_t start, size_t end, const T& value);

template <typename T>
void ptr_assign_dev(T* array, size_t start, size_t end, const T& value);

template <typename T>
void ptr_copy(T* src, T* dst, size_t num, size_t src_pos, size_t dst_pos);

template <typename T>
void ptr_copy_dev(T* src, T* dst, size_t num, size_t src_pos, size_t dst_pos);

}

#endif // __BUFFER_IMPL_H_
