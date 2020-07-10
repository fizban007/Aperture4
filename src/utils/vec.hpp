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

#ifndef __VEC_H_
#define __VEC_H_

#include "core/cuda_control.h"
#include "utils/type_traits.hpp"
#include "utils/util_functions.h"
#include <cmath>
#include <iostream>

namespace Aperture {

////////////////////////////////////////////////////////////////////////////////
///  A static vector type with compile-time size. This is useful for passing
///  small arrays with compile-time known dimensions. Since it is difficult to
///  send array as a function parameter between host and device, this wrapper
///  can be used to achieve the same thing without additional code.
////////////////////////////////////////////////////////////////////////////////
template <typename T, int Rank>
class vec_t {
 private:
  T memory[Rank] = {};

 public:
  typedef vec_t<T, Rank> self_type;

  HOST_DEVICE vec_t() {}
  HOST_DEVICE vec_t(const T (&v)[Rank]) {
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      memory[i] = v[i];
    }
  }
  template <typename... Args, typename = all_convertible_to<T, Args...>>
  HOST_DEVICE vec_t(Args... args) : memory{T(args)...} {}
  HOST_DEVICE ~vec_t() {}

  HD_INLINE T& operator[](std::size_t n) { return memory[n]; }
  HD_INLINE const T& operator[](std::size_t n) const { return memory[n]; }
  HD_INLINE T& at(std::size_t n) { return memory[n]; }
  HD_INLINE const T& at(std::size_t n) const { return memory[n]; }

  HD_INLINE self_type& operator=(const self_type& other) = default;

  template <typename U, typename = is_convertible_to<U, T>>
  HD_INLINE bool operator<(const vec_t<U, Rank>& other) const {
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      if (memory[i] >= other[i]) {
        return false;
      }
    }
    return true;
  }

  template <typename U, typename = is_convertible_to<U, T>>
  HD_INLINE bool operator<=(const vec_t<U, Rank>& other) const {
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      if (memory[i] > other[i]) {
        return false;
      }
    }
    return true;
  }

  HD_INLINE bool operator==(const self_type& other) const {
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      if (memory[i] != other.memory[i]) {
        return false;
      }
    }
    return true;
  }

  HD_INLINE bool operator!=(const self_type& other) const {
    return !operator==(other);
  }

  template <typename U, typename = is_convertible_to<U, T>>
  HD_INLINE self_type& operator+=(const vec_t<U, Rank>& other) {
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      memory[i] += T(other[i]);
    }
    return *this;
  }

  template <typename U, typename = is_convertible_to<U, T>>
  HD_INLINE self_type operator+(const vec_t<U, Rank>& other) const {
    self_type result = *this;
    result += other;
    return result;
  }

  template <typename U, typename = is_convertible_to<U, T>>
  HD_INLINE self_type& operator-=(const vec_t<U, Rank>& other) {
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      memory[i] -= other[i];
    }
    return *this;
  }

  template <typename U, typename = is_convertible_to<U, T>>
  HD_INLINE self_type operator-(const vec_t<U, Rank>& other) const {
    self_type result = *this;
    result -= other;
    return result;
  }

  template <typename U, typename = is_convertible_to<U, T>>
  HD_INLINE self_type& operator*=(const vec_t<U, Rank>& other) {
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      memory[i] *= other[i];
    }
    return *this;
  }

  template <typename U, typename = is_convertible_to<U, T>>
  HD_INLINE self_type operator*(const vec_t<U, Rank>& other) const {
    self_type result = *this;
    result *= other;
    return result;
  }

  HD_INLINE self_type& operator*=(T v) {
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      memory[i] *= v;
    }
    return *this;
  }

  HD_INLINE self_type operator*(T v) const {
    self_type result = *this;
    result *= v;
    return result;
  }

  template <typename U, typename = is_convertible_to<U, T>>
  HD_INLINE self_type& operator/=(const vec_t<U, Rank>& other) {
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      memory[i] /= other[i];
    }
    return *this;
  }

  template <typename U, typename = is_convertible_to<U, T>>
  HD_INLINE self_type operator/(const vec_t<U, Rank>& other) const {
    self_type result = *this;
    result /= other;
    return result;
  }

  HD_INLINE self_type operator/=(T v) {
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      memory[i] /= v;
    }
    return *this;
  }

  HD_INLINE self_type operator/(T v) const {
    self_type result = *this;
    result /= v;
    return result;
  }

  HD_INLINE void set(const T& value) {
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      memory[i] = value;
    }
  }

  HD_INLINE T dot(const self_type& other) const {
    T result = 0;
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      result += memory[i] * other.memory[i];
    }
    return result;
  }

  constexpr int rank() const { return Rank; }

  HD_INLINE T product() const {
    T result = memory[0];
#pragma unroll
    for (int i = 1; i < Rank; i++) {
      result *= memory[i];
    }
    return result;
  }
};

template <typename T, typename... Args,
          typename = all_convertible_to<T, Args...>>
HD_INLINE auto
vec(Args... args) {
  return vec_t<T, sizeof...(Args)>(T(args)...);
}

template <int Rank>
class extent_t : public vec_t<uint32_t, Rank> {
 public:
  using base_class = vec_t<uint32_t, Rank>;
  using base_class::base_class;

  HOST_DEVICE extent_t(const base_class& other) : base_class(other) {}
  HOST_DEVICE extent_t<Rank>& operator=(const extent_t<Rank>& other) = default;

  HOST_DEVICE uint32_t size() const { return this->product(); }
};

template <typename... Args, typename = all_convertible_to<uint32_t, Args...>>
HD_INLINE auto
extent(Args... args) {
  return extent_t<sizeof...(Args)>(uint32_t(args)...);
}

template <int Rank>
using index_t = vec_t<int32_t, Rank>;

template <typename... Args, typename = all_convertible_to<uint32_t, Args...>>
HD_INLINE auto
index(Args... args) {
  return index_t<sizeof...(Args)>(int32_t(args)...);
}

template <int Rank>
bool
not_power_of_two(const extent_t<Rank>& ext) {
  for (int i = 0; i < Rank; i++) {
    if (not_power_of_two(ext[i])) {
      return true;
    }
  }
  return false;
}

}  // namespace Aperture

#endif  // __VEC_H_
