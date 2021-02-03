/*
 * Copyright (c) 2021 Alex Chen.
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

#ifndef __DATA_ADAPTER_H_
#define __DATA_ADAPTER_H_

namespace Aperture {

template <typename T>
struct cuda_adapter;

template <typename T>
inline typename cuda_adapter<T>::type
adapt_cuda(T& t) {
  return cuda_adapter<T>::apply(t);
}

template <typename T>
inline typename cuda_adapter<T>::const_type
adapt_cuda(const T& t) {
  return cuda_adapter<T>::apply(t);
}

}  // namespace Aperture

#endif  // __DATA_ADAPTER_H_
