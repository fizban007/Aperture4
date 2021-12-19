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
 protected:
  T memory[Rank] = {};

 public:
  typedef vec_t<T, Rank> self_type;

  HD_INLINE vec_t() {}
  HD_INLINE vec_t(const T (&v)[Rank]) {
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      memory[i] = v[i];
    }
  }
  HD_INLINE vec_t(const T& v, const vec_t<T, Rank - 1>& vec) {
    memory[0] = v;
#pragma unroll
    for (int i = 1; i < Rank; i++) {
      memory[i] = vec[i - 1];
    }
  }
  HD_INLINE vec_t(const T& v1, const T& v2,
                  const vec_t<T, Rank - 2>& vec) {
    memory[0] = v1;
    memory[1] = v2;
#pragma unroll
    for (int i = 2; i < Rank; i++) {
      memory[i] = vec[i - 2];
    }
  }
  HD_INLINE vec_t(const T& v1, const T& v2, const T& v3,
                  const vec_t<T, Rank - 3>& vec) {
    memory[0] = v1;
    memory[1] = v2;
    memory[2] = v3;
#pragma unroll
    for (int i = 3; i < Rank; i++) {
      memory[i] = vec[i - 3];
    }
  }
  HD_INLINE vec_t(const vec_t<T, Rank - 1>& vec, const T& v) {
    memory[Rank - 1] = v;
#pragma unroll
    for (int i = 0; i < Rank - 1; i++) {
      memory[i] = vec[i];
    }
  }
  template <typename... Args, typename = all_convertible_to<T, Args...>>
  HD_INLINE vec_t(Args... args) : memory{T(args)...} {}
  HD_INLINE vec_t(const self_type& vec) = default;
  HD_INLINE vec_t(self_type&& vec) = default;
  HD_INLINE ~vec_t() {}

  HD_INLINE T& operator[](std::size_t n) { return memory[n]; }
  HD_INLINE const T& operator[](std::size_t n) const { return memory[n]; }
  HD_INLINE T& at(std::size_t n) { return memory[n]; }
  HD_INLINE const T& at(std::size_t n) const { return memory[n]; }

  HD_INLINE self_type& operator=(const self_type& other) = default;
  HD_INLINE self_type& operator=(const T& v) {
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      memory[i] = v;
    }
    return *this;
  }
  HD_INLINE self_type& operator=(const T (&v)[Rank]) {
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      memory[i] = v[i];
    }
    return *this;
  }

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

  template <int A, int B, typename = std::enable_if_t<B - A <= Rank>>
  HD_INLINE vec_t<T, B - A> subset() const {
    vec_t<T, B - A> result;
#pragma unroll
    for (int i = A; i < B; i++) {
      result[i - A] = memory[i];
    }
    return result;
  }

  template <typename U>
  HD_INLINE T dot(const vec_t<U, Rank>& other) const {
    T result = 0;
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      result += memory[i] * other[i];
    }
    return result;
  }

  constexpr int rank() const { return Rank; }

  const T* data() const { return memory; }

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

template <typename T, int Rank>
HD_INLINE vec_t<T, Rank>
operator*(const T& q, const vec_t<T, Rank>& v) {
  return v * q;
}

template <typename T>
HD_INLINE vec_t<T, 3>
cross(const vec_t<T, 3>& u, const vec_t<T, 3>& v) {
  return vec_t<T, 3>(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                     u[0] * v[1] - u[1] * v[0]);
}

template <int Rank>
class extent_t : public vec_t<uint32_t, Rank> {
 private:
  mutable vec_t<int64_t, Rank> m_strides;
  mutable bool has_strides = false;

 public:
  using base_class = vec_t<uint32_t, Rank>;
  // using base_class::base_class;

  // vec_t<int64_t, Rank> strides;

  HD_INLINE void get_strides() const {
    m_strides[0] = 1;
    for (int i = 1; i < Rank; i++) {
      m_strides[i] = m_strides[i - 1] * this->memory[i - 1];
    }
    has_strides = true;
  }

  HOST_DEVICE extent_t(uint32_t v, const extent_t<Rank - 1>& vec)
      : base_class(v, vec) {
    get_strides();
  }

  HOST_DEVICE extent_t(uint32_t v1, uint32_t v2,
                       const extent_t<Rank - 2>& vec)
      : base_class(v1, v2, vec) {
    get_strides();
  }

  HOST_DEVICE extent_t(uint32_t v1, uint32_t v2, uint32_t v3,
                       const extent_t<Rank - 3>& vec)
      : base_class(v1, v2, v3, vec) {
    get_strides();
  }

  HOST_DEVICE extent_t(const extent_t<Rank - 1>& vec, uint32_t v)
      : base_class(vec, v) {
    get_strides();
  }

  template <typename... Args, typename = all_convertible_to<uint32_t, Args...>>
  HOST_DEVICE extent_t(Args... args) : base_class(args...) {
    get_strides();
  }

  HOST_DEVICE extent_t(const base_class& other) : base_class(other) {
    get_strides();
  }

  HOST_DEVICE extent_t<Rank>& operator=(const extent_t<Rank>& other) = default;

  HOST_DEVICE uint32_t size() const { return this->product(); }

  HOST_DEVICE const vec_t<int64_t, Rank>& strides() const {
    if (!has_strides) get_strides();
    return m_strides;
  }
};

template <typename... Args, typename = all_convertible_to<uint32_t, Args...>>
HD_INLINE auto
extent(Args... args) {
  return extent_t<sizeof...(Args)>(uint32_t(args)...);
}

template <int Rank>
// using index_t = vec_t<int32_t, Rank>;
class index_t : public vec_t<int32_t, Rank> {
 public:
  HD_INLINE index_t() : vec_t<int32_t, Rank>(0) {}

  HD_INLINE index_t(const vec_t<int32_t, Rank>& v) : vec_t<int32_t, Rank>(v) {}

  HD_INLINE index_t(int32_t v, const vec_t<int32_t, Rank - 1>& vec)
      : vec_t<int32_t, Rank>(v, vec) {}

  HD_INLINE index_t(int32_t v1, int32_t v2, const vec_t<int32_t, Rank - 2>& vec)
      : vec_t<int32_t, Rank>(v1, v2, vec) {}

  HD_INLINE index_t(int32_t v1, int32_t v2, int32_t v3,
                    const vec_t<int32_t, Rank - 3>& vec)
      : vec_t<int32_t, Rank>(v1, v2, v3, vec) {}

  HD_INLINE index_t(const vec_t<int32_t, Rank - 1>& vec, const int32_t& v)
      : vec_t<int32_t, Rank>(vec, v) {}

  template <typename... Args, typename = all_convertible_to<int32_t, Args...>>
  HD_INLINE explicit index_t(Args... args) : vec_t<int32_t, Rank>(args...) {}

  HD_INLINE index_t<Rank> inc_x(int n) {
    index_t<Rank> result = *this;
    result.memory[0] += n;
    return result;
  }

  HD_INLINE index_t<Rank> inc_y(int n) {
    index_t<Rank> result = *this;
    result.memory[1] += n;
    return result;
  }

  HD_INLINE index_t<Rank> inc_z(int n) {
    index_t<Rank> result = *this;
    result.memory[2] += n;
    return result;
  }

  HD_INLINE index_t<Rank> dec_x(int n) {
    index_t<Rank> result = *this;
    result.memory[0] -= n;
    return result;
  }

  HD_INLINE index_t<Rank> dec_y(int n) {
    index_t<Rank> result = *this;
    result.memory[1] -= n;
    return result;
  }

  HD_INLINE index_t<Rank> dec_z(int n) {
    index_t<Rank> result = *this;
    result.memory[2] -= n;
    return result;
  }
};

template <typename... Args, typename = all_convertible_to<int32_t, Args...>>
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
