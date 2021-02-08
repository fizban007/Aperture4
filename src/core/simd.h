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

#ifndef _SIMD_H_
#define _SIMD_H_

// #include <immintrin.h>
#include "core/typedefs_and_constants.h"
#define MAX_VECTOR_SIZE 512
#include "utils/index.hpp"
#include "vectorclass.h"

namespace Aperture {

namespace simd {

#if !defined(USE_DOUBLE) && (defined(__AVX512F__) || defined(__AVX512__))
#define USE_SIMD
#pragma message "using AVX512 with float"
typedef Vec16ui Vec_ui_t;
typedef Vec16i Vec_i_t;
typedef Vec16ib Vec_ib_t;
typedef Vec16f Vec_f_t;
const Vec_f_t vec_inc =
    Vec16f(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
           11.0f, 12.0f, 13.0f, 14.0f, 15.0f);
constexpr int vec_width = 16;
#elif defined(USE_DOUBLE) && (defined(__AVX512F__) || defined(__AVX512__))
#define USE_SIMD
#pragma message "using AVX512 with double"
typedef Vec8uq Vec_ui_t;
typedef Vec8q Vec_i_t;
typedef Vec8qb Vec_ib_t;
typedef Vec8d Vec_f_t;
const Vec_f_t vec_inc = Vec8d(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
constexpr int vec_width = 8;
#elif !defined(USE_DOUBLE) && defined(__AVX2__)
#define USE_SIMD
#pragma message "using AVX2 with float"
typedef Vec8ui Vec_idx_t;
typedef Vec8ui Vec_ui_t;
typedef Vec8i Vec_i_t;
typedef Vec8ib Vec_ib_t;
typedef Vec8f Vec_f_t;
const Vec_f_t vec_inc = Vec8f(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
constexpr int vec_width = 8;
#elif defined(USE_DOUBLE) && defined(__AVX2__)
#define USE_SIMD
#pragma message "using AVX2 with double"
typedef Vec8ui Vec_idx_t;
typedef Vec4uq Vec_ui_t;
typedef Vec4q Vec_i_t;
typedef Vec4qb Vec_ib_t;
typedef Vec4d Vec_f_t;
const Vec_f_t vec_inc = Vec4d(0.0, 1.0, 2.0, 3.0);
constexpr int vec_width = 4;
#else
#undef USE_SIMD
typedef uint32_t Vec_idx_t;
typedef uint32_t Vec_ui_t;
typedef int Vec_i_t;
typedef bool Vec_ib_t;
typedef Scalar Vec_f_t;
constexpr Vec_f_t vec_inc = 0.0;
constexpr int vec_width = 1;
#endif

inline Vec_i_t
mod(const Vec_i_t& a, uint32_t b) {
  auto n = a / b;
  return a - n * b;
}

template <int Rank>
inline vec_t<Vec_i_t, Rank> get_pos(const Vec_i_t& linear,
                                    const extent_t<Rank>& ext);

template <>
inline vec_t<Vec_i_t, 1>
get_pos(const Vec_i_t& linear, const extent_t<1>& ext) {
  return vec_t<Vec_i_t, 1>(linear);
}

template <>
inline vec_t<Vec_i_t, 2>
get_pos(const Vec_i_t& linear, const extent_t<2>& ext) {
  return vec_t<Vec_i_t, 2>(linear / ext[0], mod(linear, ext[0]));
}

template <>
inline vec_t<Vec_i_t, 3>
get_pos(const Vec_i_t& linear, const extent_t<3>& ext) {
  return vec_t<Vec_i_t, 3>(linear / (ext[0] * ext[1]),
                           mod((linear / ext[0]), ext[1]), mod(linear, ext[0]));
}

}  // namespace simd

namespace math {

inline simd::Vec_f_t
sqrt(simd::Vec_f_t x) {
  return ::sqrt(x);
}
inline simd::Vec_f_t
floor(simd::Vec_f_t x) {
  return ::floor(x);
}

}  // namespace math

}  // namespace Aperture

#endif  // _SIMD_H_
