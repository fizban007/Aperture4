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

#ifndef _UTIL_FUNCTIONS_H_
#define _UTIL_FUNCTIONS_H_

#include "core/cuda_control.h"
#include "core/enum_types.h"
#include "core/typedefs_and_constants.h"
#include "utils/type_traits.hpp"
#include <cmath>
#include <string>

namespace Aperture {

template <typename T>
HD_INLINE T
atomic_add(T* addr, type_identity_t<T> value) {
// atomic_add(T* addr, T value) {
#ifdef __CUDACC__
  return atomicAdd(addr, value);
#else
  T tmp;
#pragma omp atomic capture
  {
    tmp = *addr;
    *addr += value;
  }
  return tmp;
#endif
}

template <typename T>
HD_INLINE T
square(const T& val) {
  return val * val;
}

template <typename T>
HD_INLINE T
cube(const T& val) {
  return val * val * val;
}

template <typename T>
HD_INLINE void
swap_values(T& a, T& b) {
  T tmp = a;
  a = b;
  b = tmp;
}

template <typename T>
HD_INLINE int
sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T>
HD_INLINE T
clamp(T val, type_identity_t<T> a, type_identity_t<T> b) {
  return std::max(T(a), std::min(T(b), val));
}

// Flag manipulation functions
template <typename Flag>
HD_INLINE bool
check_flag(uint32_t flag, Flag bit) {
  return (flag & (1 << static_cast<int>(bit))) != 0;
}

template <typename Flag>
HD_INLINE uint32_t
flag_or(Flag bit) {
  return (1 << static_cast<int>(bit));
}

template <typename Flag, typename... P>
HD_INLINE uint32_t
flag_or(Flag bit, P... bits) {
  return ((1 << static_cast<int>(bit)) | flag_or(bits...));
}

template <typename... Flag>
HD_INLINE void
set_flag(uint32_t& flag, Flag... bits) {
  flag |= flag_or(bits...);
}

template <typename... Flag>
HD_INLINE void
clear_flag(uint32_t& flag, Flag... bits) {
  flag &= ~static_cast<int>(flag_or(bits...));
}

template <typename... Flag>
HD_INLINE void
toggle_flag(uint32_t& flag, Flag... bits) {
  flag ^= static_cast<int>(flag_or(bits...));
}

// Get an integer representing particle type from a given flag
// HD_INLINE uint32_t
// get_ptc_type(uint32_t flag) {
//   return (flag >> (32 - max_ptc_type_bits));
// }
template <typename Uint>
HD_INLINE Uint
get_ptc_type(Uint flag) {
  return (flag >> (32 - max_ptc_type_bits));
}

// Generate a particle flag from a give particle type
HD_INLINE uint32_t
gen_ptc_type_flag(PtcType type) {
  return ((uint32_t)type << (32 - max_ptc_type_bits));
}

// Set a given flag such that it now represents given particle type
HD_INLINE uint32_t
set_ptc_type_flag(uint32_t flag, PtcType type) {
  return (flag & ((uint32_t)-1 >> max_ptc_type_bits)) | gen_ptc_type_flag(type);
}

template <typename T, typename = is_integral_t<T>>
HD_INLINE bool
not_power_of_two(T num) {
  return (num != 1) && (num & (num - 1));
}

template <typename T, typename = is_integral_t<T>>
HD_INLINE bool
is_power_of_two(T num) {
  return !not_power_of_two(num);
}

// compute the next highest power of 2 of 32-bit result
template <typename T, typename = is_integral_t<T>>
HD_INLINE T
next_power_of_two(T num) {
  T p = 1;
  while (p < num) p <<= 1;
  return p;
}

HD_INLINE float
to_float(int32_t n) {
  return (float)n;
}

HD_INLINE double
to_double(int32_t n) {
  return (double)n;
}

HD_INLINE int32_t
roundi(Scalar n) {
  return std::round(n);
}

}  // namespace Aperture

#endif  // _UTIL_FUNCTIONS_H_
