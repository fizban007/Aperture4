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

#pragma once

#include "core/cuda_control.h"
#include "core/ndptr.hpp"
#include "core/simd.h"
#include "utils/stagger.h"
#include "utils/vec.hpp"

namespace Aperture {

namespace simd {

template <int N>
struct bspline;

template <>
struct bspline<0> {
  enum { radius = 1, support = 1 };

  template <typename FloatT>
  HD_INLINE FloatT operator()(FloatT dx) const {
    return select(abs(dx) <= 0.5f, 1.0f, 0.0f);
  }
};

template <>
struct bspline<1> {
  enum { radius = 1, support = 2 };

  template <typename FloatT>
  HD_INLINE FloatT operator()(FloatT dx) const {
    FloatT abs_dx = abs(dx);
    return max(FloatT(1.0f) - abs_dx, 0.0f);
  }
};

template <>
struct bspline<2> {
  enum { radius = 2, support = 3 };

  template <typename FloatT>
  HD_INLINE FloatT operator()(FloatT dx) const {
    FloatT abs_dx = abs(dx);
    return select(
        abs_dx < 0.5f, 0.75f - dx * dx,
        select(abs_dx < 1.5f, square((FloatT)1.5f - abs_dx) * 0.5f, 0.0f));
    // if (abs_dx < 0.5f) {
    //   return 0.75f - dx * dx;
    // } else if (abs_dx < 1.5f) {
    //   FloatT tmp = 1.5f - abs_dx;
    //   return 0.5f * tmp * tmp;
    // } else {
    //   return 0.0f;
    // }
  }
};

template <>
struct bspline<3> {
  enum { radius = 2, support = 4 };

  template <typename FloatT>
  HD_INLINE FloatT operator()(FloatT dx) const {
    FloatT abs_dx = abs(dx);
    return select(abs_dx < 1.0f,
                  2.0f / 3.0f - square(abs_dx) + 0.5f * cube(abs_dx),
                  select(abs_dx < 2.0f, cube(2.0f - abs_dx) / 6.0f), 0.0f);
    // if (abs_dx < 1.0f) {
    //   FloatT tmp = abs_dx * abs_dx;
    //   return 2.0f / 3.0f - tmp + 0.5f * tmp * abs_dx;
    // } else if (abs_dx < 2.0f) {
    //   FloatT tmp = 2.0f - abs_dx;
    //   return 1.0f / 6.0f * tmp * tmp * tmp;
    // } else {
    //   return 0.0f;
    // }
  }
};

template <typename Interp, typename FloatT>
FloatT HD_INLINE
interp_cell(const Interp& interp, FloatT rel_pos, int rel_c) {
  // The actual distance between particle and t
  FloatT x = (FloatT)rel_c - rel_pos;
  return interp(x);
}

template <typename Interp, typename FloatT>
FloatT HD_INLINE
interp_cell(const Interp& interp, FloatT rel_pos, int rel_c, int stagger) {
  // The actual distance between particle and t
  FloatT x = (FloatT)rel_c + 0.5f * (1 - stagger) - rel_pos;
  return interp(x);
}

template <typename Interp, int Dim>
struct interpolator;

template <typename Interp>
struct interpolator<Interp, 1> {
  Interp interp;

  template <typename value_t, typename IntT, typename FloatT>
  HOST_DEVICE FloatT operator()(const value_t* f, const vec_t<FloatT, 3>& x,
                                const IntT& idx, const extent_t<1>& ext,
                                stagger_t stagger = stagger_t(0b111)) const {
    FloatT result = 0.0f;
#pragma unroll
    for (int i = stagger[0] - Interp::radius;
         i <= Interp::support - Interp::radius; i++) {
      // int ii = i + pos[0] - Interp::radius;
      FloatT val = lookup<vec_width>(idx + i * ext.strides()[0], f);
      // val.load(&f[idx + i * ext.strides()[0]]);
      result += val * interp_cell(interp, x[0], i, stagger[0]);
    }
    return result;
  }
};

template <typename Interp>
struct interpolator<Interp, 2> {
  Interp interp;

  template <typename value_t, typename IntT, typename FloatT>
  HOST_DEVICE FloatT operator()(const value_t* f, const vec_t<FloatT, 3>& x,
                                const IntT& idx, const extent_t<2>& ext,
                                stagger_t stagger = stagger_t(0b111)) const {
    FloatT result = 0.0f;
#pragma unroll
    for (int j = stagger[1] - Interp::radius;
         j <= Interp::support - Interp::radius; j++) {
      auto idx_j = idx + j * ext.strides()[1];
#pragma unroll
      for (int i = stagger[0] - Interp::radius;
           i <= Interp::support - Interp::radius; i++) {
        FloatT val = lookup<vec_width>(idx_j + i * ext.strides()[0], f);
        // val.load(&f[idx_j + i * ext.strides()[0]]);
        result += val * interp_cell(interp, x[0], i, stagger[0]) *
                  interp_cell(interp, x[1], j, stagger[1]);
      }
    }
    return result;
  }
};

template <typename Interp>
struct interpolator<Interp, 3> {
  Interp interp;

  template <typename value_t, typename IntT, typename FloatT>
  HOST_DEVICE FloatT operator()(const value_t* f, const vec_t<FloatT, 3>& x,
                                const IntT& idx, const extent_t<3>& ext,
                                stagger_t stagger = stagger_t(0b111)) const {
    FloatT result = 0.0f;
#pragma unroll
    for (int k = stagger[2] - Interp::radius;
         k <= Interp::support - Interp::radius; k++) {
      auto idx_k = idx + k * ext.strides()[2];
#pragma unroll
      for (int j = stagger[1] - Interp::radius;
           j <= Interp::support - Interp::radius; j++) {
        auto idx_j = idx_k + j * ext.strides()[1];
#pragma unroll
        for (int i = stagger[0] - Interp::radius;
             i <= Interp::support - Interp::radius; i++) {
          FloatT val = lookup<vec_width>(idx_j + i * ext.strides()[0], f);
          // val.load(&f[idx_j + i * ext.strides()[0]]);
          result += val * interp_cell(interp, x[0], i, stagger[0]) *
                    interp_cell(interp, x[1], j, stagger[1]) *
                    interp_cell(interp, x[2], k, stagger[2]);
        }
      }
    }
    return result;
  }
};

}  // namespace simd

}  // namespace Aperture
