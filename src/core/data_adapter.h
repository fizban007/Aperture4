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

#include "utils/nonown_ptr.hpp"

namespace Aperture {

template <typename T>
struct cuda_adapter {
  typedef T type;
  typedef T const_type;

  // static type apply(T& t) {
  //   return t;
  // }
  static const_type apply(const T& t) { return t; }
};

template <typename T>
struct host_adapter {
  typedef T type;
  typedef T const_type;

  // static type apply(T& t) {
  //   return t;
  // }
  static const_type apply(const T& t) { return t; }
};

template <typename T>
inline typename cuda_adapter<T>::type
adapt_cuda(nonown_ptr<T>& t) {
  return cuda_adapter<T>::apply(*t);
}

template <typename T>
inline typename cuda_adapter<T>::type
adapt_cuda(T& t) {
  return cuda_adapter<T>::apply(t);
}

template <typename T>
inline typename cuda_adapter<T>::const_type
adapt_cuda(const nonown_ptr<T>& t) {
  return cuda_adapter<T>::apply(*t);
}

template <typename T>
inline typename cuda_adapter<T>::const_type
adapt_cuda(const T& t) {
  return cuda_adapter<T>::apply(t);
}

template <typename T>
inline typename host_adapter<T>::type
adapt_host(nonown_ptr<T>& t) {
  return host_adapter<T>::apply(*t);
}

template <typename T>
inline typename host_adapter<T>::type
adapt_host(T& t) {
  return host_adapter<T>::apply(t);
}

template <typename T>
inline typename host_adapter<T>::const_type
adapt_host(const nonown_ptr<T>& t) {
  return host_adapter<T>::apply(*t);
}

template <typename T>
inline typename host_adapter<T>::const_type
adapt_host(const T& t) {
  return host_adapter<T>::apply(t);
}

}  // namespace Aperture

#endif  // __DATA_ADAPTER_H_
