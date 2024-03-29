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
#include "utils/stagger.h"
#include "utils/vec.hpp"

namespace Aperture {

template <int N>
struct bspline;

template <>
struct bspline<0> {
  enum { radius = 1, support = 1 };

  template <typename FloatT>
  HD_INLINE FloatT operator()(FloatT dx) const {
    return (std::abs(dx) <= 0.5f ? 1.0f : 0.0f);
  }
};

template <>
struct bspline<1> {
  enum { radius = 1, support = 2 };

  template <typename FloatT>
  HD_INLINE FloatT operator()(FloatT dx) const {
    FloatT abs_dx = std::abs(dx);
    return std::max(1.0f - abs_dx, (FloatT)0.0);
  }
};

template <>
struct bspline<2> {
  enum { radius = 2, support = 3 };

  template <typename FloatT>
  HD_INLINE FloatT operator()(FloatT dx) const {
    FloatT abs_dx = std::abs(dx);
    if (abs_dx < 0.5f) {
      return 0.75f - dx * dx;
    } else if (abs_dx < 1.5f) {
      FloatT tmp = 1.5f - abs_dx;
      return 0.5f * tmp * tmp;
    } else {
      return 0.0f;
    }
  }
};

template <>
struct bspline<3> {
  enum { radius = 2, support = 4 };

  template <typename FloatT>
  HD_INLINE FloatT operator()(FloatT dx) const {
    FloatT abs_dx = std::abs(dx);
    if (abs_dx < 1.0f) {
      FloatT tmp = abs_dx * abs_dx;
      return 2.0f / 3.0f - tmp + 0.5f * tmp * abs_dx;
    } else if (abs_dx < 2.0f) {
      FloatT tmp = 2.0f - abs_dx;
      return 1.0f / 6.0f * tmp * tmp * tmp;
    } else {
      return 0.0f;
    }
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

  //   template <typename Ptr, typename Index_t, typename FloatT>
  //   HOST_DEVICE auto operator()(const Ptr& f, const vec_t<FloatT, 3>& x,
  //                               const Index_t& idx) const ->
  //       typename Ptr::value_t {
  //     typename Ptr::value_t result = 0.0f;
  // #pragma unroll
  //     for (int i = 1 - Interp::radius; i <= Interp::support - Interp::radius;
  //     i++) {
  //       // int ii = i + pos[0] - Interp::radius;
  //       result += f[idx.inc_x(i)] * interp_cell(interp, x[0], i);
  //     }
  //     return result;
  //   }

  template <typename Ptr, typename Index_t, typename FloatT>
  // HOST_DEVICE auto operator()(const Ptr& f, const vec_t<FloatT, 3>& x,
  //                             const Index_t& idx,
  //                             stagger_t stagger = stagger_t(0b111)) const ->
  HOST_DEVICE auto operator()(const vec_t<FloatT, 3>& x, const Ptr& f,
                              const Index_t& idx, const extent_t<1>& ext,
                              stagger_t stagger = stagger_t(0b111)) const ->
      typename Ptr::value_t {
    typename Ptr::value_t result = 0.0f;
    int i_lower = 1 - Interp::radius - (stagger[0] == 0 && x[0] < 0.5f);
    int i_upper =
        Interp::support - Interp::radius - (stagger[0] == 0 && x[0] < 0.5f);
#pragma unroll
    for (int i = i_lower; i <= i_upper; i++) {
      result += f[idx.inc_x(i)] * interp_cell(interp, x[0], i, stagger[0]);
    }
    return result;
  }
};

template <typename Interp>
struct interpolator<Interp, 2> {
  Interp interp;

  //   template <typename Ptr, typename Index_t, typename FloatT>
  //   HOST_DEVICE auto operator()(const Ptr& f, const vec_t<FloatT, 3>& x,
  //                               const Index_t& idx) const ->
  //       typename Ptr::value_t {
  //     typename Ptr::value_t result = 0.0f;
  // #pragma unroll
  //     for (int j = 1 - Interp::radius; j <= Interp::support - Interp::radius;
  //          j++) {
  //       auto idx_j = idx.inc_y(j);
  // #pragma unroll
  //       for (int i = 1 - Interp::radius; i <= Interp::support -
  //       Interp::radius;
  //            i++) {
  //         result += f[idx_j.inc_x(i)] *
  //         // result += f[idx.inc_x(i).inc_y(j)] *
  //                   interp_cell(interp, x[0], i) *
  //                   interp_cell(interp, x[1], j);
  //       }
  //     }
  //     return result;
  //   }

  template <typename Ptr, typename Index_t, typename FloatT>
  // HOST_DEVICE auto operator()(const Ptr& f, const vec_t<FloatT, 3>& x,
  //                             const Index_t& idx,
  //                             stagger_t stagger = stagger_t(0b111)) const ->
  HOST_DEVICE auto operator()(const vec_t<FloatT, 3>& x, const Ptr& f,
                              const Index_t& idx, const extent_t<2>& ext,
                              stagger_t stagger = stagger_t(0b111)) const ->
      typename Ptr::value_t {
    typename Ptr::value_t result = 0.0f;
    int j_lower = 1 - Interp::radius - (stagger[1] == 0 && x[1] < 0.5f);
    int j_upper =
        Interp::support - Interp::radius - (stagger[1] == 0 && x[1] < 0.5f);
    int i_lower = 1 - Interp::radius - (stagger[0] == 0 && x[0] < 0.5f);
    int i_upper =
        Interp::support - Interp::radius - (stagger[0] == 0 && x[0] < 0.5f);
#pragma unroll
    for (int j = j_lower; j <= j_upper; j++) {
      auto idx_j = idx.inc_y(j);
#pragma unroll
      for (int i = i_lower; i <= i_upper; i++) {
        result += f[idx_j.inc_x(i)] * interp_cell(interp, x[0], i, stagger[0]) *
                  interp_cell(interp, x[1], j, stagger[1]);
      }
    }
    return result;
  }
};

template <typename Interp>
struct interpolator<Interp, 3> {
  Interp interp;

  //   template <typename Ptr, typename Index_t, typename FloatT>
  //   HOST_DEVICE auto operator()(const Ptr& f, const vec_t<FloatT, 3>& x,
  //                               const Index_t& idx) const ->
  //       typename Ptr::value_t {
  //     typename Ptr::value_t result = 0.0f;
  // #pragma unroll
  //     for (int k = 1 - Interp::radius; k <= Interp::support - Interp::radius;
  //          k++) {
  //       auto idx_k = idx.inc_z(k);
  // #pragma unroll
  //       for (int j = 1 - Interp::radius; j <= Interp::support -
  //       Interp::radius;
  //            j++) {
  //         auto idx_j = idx_k.inc_y(j);
  // #pragma unroll
  //         for (int i = 1 - Interp::radius; i <= Interp::support -
  //         Interp::radius;
  //              i++) {
  //           result += f[idx_j.inc_x(i)] *
  //                     interp_cell(interp, x[0], i) *
  //                     interp_cell(interp, x[1], j) *
  //                     interp_cell(interp, x[2], k);
  //         }
  //       }
  //     }
  //     return result;
  //   }

  template <typename Ptr, typename Index_t, typename FloatT>
  HOST_DEVICE auto operator()(const vec_t<FloatT, 3>& x, const Ptr& f,
                              const Index_t& idx, const extent_t<3>& ext,
                              stagger_t stagger = stagger_t(0b111)) const ->
      typename Ptr::value_t {
    typename Ptr::value_t result = 0.0;
    int k_lower = 1 - Interp::radius - (stagger[2] == 0 && x[2] < 0.5f);
    int k_upper =
        Interp::support - Interp::radius - (stagger[2] == 0 && x[2] < 0.5f);
    int j_lower = 1 - Interp::radius - (stagger[1] == 0 && x[1] < 0.5f);
    int j_upper =
        Interp::support - Interp::radius - (stagger[1] == 0 && x[1] < 0.5f);
    int i_lower = 1 - Interp::radius - (stagger[0] == 0 && x[0] < 0.5f);
    int i_upper =
        Interp::support - Interp::radius - (stagger[0] == 0 && x[0] < 0.5f);
#pragma unroll
    for (int k = k_lower; k <= k_upper; k++) {
      auto idx_k = idx.inc_z(k);
#pragma unroll
      for (int j = j_lower; j <= j_upper; j++) {
        auto idx_j = idx_k.inc_y(j);
#pragma unroll
        for (int i = i_lower; i <= i_upper; i++) {
          result += f[idx_j.inc_x(i)] *
                    interp_cell(interp, x[0], i, stagger[0]) *
                    interp_cell(interp, x[1], j, stagger[1]) *
                    interp_cell(interp, x[2], k, stagger[2]);
        }
      }
    }
    return result;
  }
};

template <int Order, int Dim>
struct interp_t {
  // template <typename value_t, typename array_t, typename Idx_t>
  // value_t operator()(const value_t& x, const array_t& f, const Idx_t& idx,
  // int stagger);
};

template <>
struct interp_t<1, 1> {
  template <typename value_t, typename array_t, typename Idx_t>
  HD_INLINE value_t operator()(const vec_t<value_t, 3>& x, const array_t& f,
                               const Idx_t& idx, const vec_t<uint32_t, 1>& ext,
                               stagger_t stagger = stagger_t(0b111)) const {
    if (stagger[0] == 1) {
      return (1.0f - x[0]) * f[idx] + x[0] * f[inc_x(idx, ext)];
    } else if (stagger[0] == 0 && x[0] < 0.5f) {
      return (x[0] + 0.5f) * f[dec_x(idx, ext)] + (0.5f - x[0]) * f[idx];
    } else {
      return (x[0] - 0.5f) * f[idx] + (1.5f - x[0]) * f[inc_x(idx, ext)];
    }
  }
};

template <>
struct interp_t<1, 2> {
  template <typename value_t, typename array_t, typename Idx_t>
  HD_INLINE value_t operator()(const vec_t<value_t, 3>& x, const array_t& f,
                               const Idx_t& idx, const vec_t<uint32_t, 2>& ext,
                               stagger_t stagger = stagger_t(0b111)) const {
    interp_t<1, 1> interp;
    if (stagger[1] == 1) {
      return (1.0f - x[1]) * interp(x, f, idx, ext.subset<0, 1>(), stagger) +
             x[1] * interp(x, f, inc_y(idx, ext), ext.subset<0, 1>(), stagger);
    } else if (stagger[1] == 0 && x[1] < 0.5f) {
      return (x[1] + 0.5f) *
                 interp(x, f, dec_y(idx, ext), ext.subset<0, 1>(), stagger) +
             (0.5f - x[1]) * interp(x, f, idx, ext.subset<0, 1>(), stagger);
    } else {
      return (x[1] - 0.5f) * interp(x, f, idx, ext.subset<0, 1>(), stagger) +
             (1.5f - x[1]) *
                 interp(x, f, inc_y(idx, ext), ext.subset<0, 1>(), stagger);
    }
  }
};

template <>
struct interp_t<1, 3> {
  template <typename value_t, typename array_t, typename Idx_t>
  HD_INLINE value_t operator()(const vec_t<value_t, 3>& x, const array_t& f,
                               const Idx_t& idx, const vec_t<uint32_t, 3>& ext,
                               stagger_t stagger = stagger_t(0b111)) const {
    interp_t<1, 2> interp;
    if (stagger[2] == 1) {
      return (1.0f - x[2]) * interp(x, f, idx, ext.subset<0, 2>(), stagger) +
             x[2] * interp(x, f, inc_z(idx, ext), ext.subset<0, 2>(), stagger);
    } else if (stagger[2] == 0 && x[2] < 0.5f) {
      return (x[2] + 0.5f) *
                 interp(x, f, dec_z(idx, ext), ext.subset<0, 2>(), stagger) +
             (0.5f - x[2]) * interp(x, f, idx, ext.subset<0, 2>(), stagger);
    } else {
      return (x[2] - 0.5f) * interp(x, f, idx, ext.subset<0, 2>(), stagger) +
             (1.5f - x[2]) *
                 interp(x, f, inc_z(idx, ext), ext.subset<0, 2>(), stagger);
    }
  }
};

template <int Dim>
struct lerp;

// 1D interpolation
template <>
struct lerp<1> {
  template <class Value_t, typename Index_t, typename FloatT>
  HOST_DEVICE Value_t operator()(const ndptr<Value_t, 1, Index_t>& f,
                                 const vec_t<FloatT, 3>& x,
                                 const Index_t& idx) {
    return x[0] * f[idx.inc_x()] + (1.0f - x[0]) * f[idx];
  }

  template <class Ptr>
  HOST_DEVICE typename Ptr::value_t operator()(const Ptr& f,
                                               const typename Ptr::idx_t& idx,
                                               stagger_t in, stagger_t out) {
    int dx_m = (in[0] == out[0] ? 0 : -out[0]);
    int dx_p = (in[0] == out[0] ? 0 : 1 - out[0]);

    return 0.5f * (f[idx.inc_x(dx_m)] + f[idx.inc_x(dx_p)]);
  }
};

// 2D interpolation
template <>
struct lerp<2> {
  template <class Value_t, typename Index_t, typename FloatT>
  HOST_DEVICE Value_t operator()(const ndptr<Value_t, 2, Index_t>& f,
                                 const vec_t<FloatT, 3>& x,
                                 const Index_t& idx) {
    FloatT f1 = x[1] * f[idx.inc_x().inc_y()] + (1.0f - x[1]) * f[idx.inc_x()];
    FloatT f0 = x[1] * f[idx.inc_y()] + (1.0f - x[1]) * f[idx];
    return x[0] * f1 + (1.0f - x[0]) * f0;
  }

  template <class Ptr>
  HOST_DEVICE typename Ptr::value_t operator()(const Ptr& f,
                                               const typename Ptr::idx_t& idx,
                                               stagger_t in, stagger_t out) {
    int dx_m = (in[0] == out[0] ? 0 : -out[0]);
    int dx_p = (in[0] == out[0] ? 0 : 1 - out[0]);
    int dy_m = (in[1] == out[1] ? 0 : -out[1]);
    int dy_p = (in[1] == out[1] ? 0 : 1 - out[1]);

    typename Ptr::value_t f1 = 0.5f * (f[idx.inc_x(dx_p).inc_y(dy_p)] +
                                       f[idx.inc_x(dx_p).inc_y(dy_m)]);
    typename Ptr::value_t f0 = 0.5f * (f[idx.inc_x(dx_m).inc_y(dy_p)] +
                                       f[idx.inc_x(dx_m).inc_y(dy_m)]);
    return 0.5f * (f1 + f0);
  }
};

// 3D interpolation
template <>
struct lerp<3> {
  template <class Value_t, typename Index_t, typename FloatT>
  HOST_DEVICE Value_t operator()(const ndptr<Value_t, 3, Index_t>& f,
                                 const vec_t<FloatT, 3>& x,
                                 const Index_t& idx) {
    FloatT f11 = (1.0f - x[2]) * f[idx.inc_x().inc_y()] +
                 x[2] * f[idx.inc_x().inc_y().inc_z()];
    FloatT f10 = (1.0f - x[2]) * f[idx.inc_x()] + x[2] * f[idx.inc_x().inc_z()];
    FloatT f01 = (1.0f - x[2]) * f[idx.inc_y()] + x[2] * f[idx.inc_y().inc_z()];
    FloatT f00 = (1.0f - x[2]) * f[idx] + x[2] * f[idx.inc_z()];
    FloatT f1 = x[1] * f11 + (1.0f - x[1]) * f10;
    FloatT f0 = x[1] * f01 + (1.0f - x[1]) * f00;
    return x[0] * f1 + (1.0f - x[0]) * f0;
  }

  template <class Ptr>
  HOST_DEVICE typename Ptr::value_t operator()(const Ptr& f,
                                               const typename Ptr::idx_t& idx,
                                               stagger_t in, stagger_t out) {
    int dx_m = (in[0] == out[0] ? 0 : -out[0]);
    int dx_p = (in[0] == out[0] ? 0 : 1 - out[0]);
    int dy_m = (in[1] == out[1] ? 0 : -out[1]);
    int dy_p = (in[1] == out[1] ? 0 : 1 - out[1]);
    int dz_m = (in[2] == out[2] ? 0 : -out[2]);
    int dz_p = (in[2] == out[2] ? 0 : 1 - out[2]);

    typename Ptr::value_t f11 =
        0.5f * (f[idx.inc_x(dx_p).inc_y(dy_p).inc_z(dz_m)] +
                f[idx.inc_x(dx_p).inc_y(dy_p).inc_z(dz_p)]);
    typename Ptr::value_t f10 =
        0.5f * (f[idx.inc_x(dx_p).inc_y(dy_m).inc_z(dz_m)] +
                f[idx.inc_x(dx_p).inc_y(dy_m).inc_z(dz_p)]);
    typename Ptr::value_t f01 =
        0.5f * (f[idx.inc_x(dx_m).inc_y(dy_p).inc_z(dz_m)] +
                f[idx.inc_x(dx_m).inc_y(dy_p).inc_z(dz_p)]);
    typename Ptr::value_t f00 =
        0.5f * (f[idx.inc_x(dx_m).inc_y(dy_m).inc_z(dz_m)] +
                f[idx.inc_x(dx_m).inc_y(dy_m).inc_z(dz_p)]);
    typename Ptr::value_t f1 = 0.5f * (f11 + f10);
    typename Ptr::value_t f0 = 0.5f * (f01 + f00);
    return 0.5f * (f1 + f0);
  }
};

//// The following two functions do not know the stagger, but are just written
//// for quick tests. Avoid using these in production code
template <typename Ptr, typename Index>
HOST_DEVICE float
lerp2(const Ptr& f, float x, float y, const Index& idx) {
  float f1 = y * f[idx.inc_x().inc_y()] + (1.0f - y) * f[idx.inc_x()];
  float f0 = y * f[idx.inc_y()] + (1.0f - y) * f[idx];
  return x * f1 + (1.0f - x) * f0;
}

template <typename Ptr, typename Index>
HOST_DEVICE float
lerp3(const Ptr& f, float x, float y, float z, const Index& idx) {
  float f11 =
      (1.0f - z) * f[idx.inc_x().inc_y()] + z * f[idx.inc_x().inc_y().inc_z()];
  float f10 = (1.0f - z) * f[idx.inc_x()] + z * f[idx.inc_x().inc_z()];
  float f01 = (1.0f - z) * f[idx.inc_y()] + z * f[idx.inc_y().inc_z()];
  float f00 = (1.0f - z) * f[idx] + z * f[idx.inc_z()];
  float f1 = y * f11 + (1.0f - y) * f10;
  float f0 = y * f01 + (1.0f - y) * f00;
  return x * f1 + (1.0f - x) * f0;
}

}  // namespace Aperture
