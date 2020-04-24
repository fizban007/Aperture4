#ifndef __INTERPOLATION_H_
#define __INTERPOLATION_H_

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
    return (std::abs(dx) <= 0.5 ? 1.0 : 0.0);
  }
};

template <>
struct bspline<1> {
  enum { radius = 1, support = 2 };

  template <typename FloatT>
  HD_INLINE FloatT operator()(FloatT dx) const {
    double abs_dx = std::abs(dx);
    return std::max(1.0 - abs_dx, 0.0);
  }
};

template <>
struct bspline<2> {
  enum { radius = 2, support = 3 };

  template <typename FloatT>
  HD_INLINE FloatT operator()(FloatT dx) const {
    FloatT abs_dx = std::abs(dx);
    if (abs_dx < 0.5) {
      return 0.75 - dx * dx;
    } else if (abs_dx < 1.5) {
      FloatT tmp = 1.5 - abs_dx;
      return 0.5 * tmp * tmp;
    } else {
      return 0.0;
    }
  }
};

template <>
struct bspline<3> {
  enum { radius = 2, support = 4 };

  template <typename FloatT>
  HD_INLINE FloatT operator()(FloatT dx) const {
    FloatT abs_dx = std::abs(dx);
    if (abs_dx < 1.0) {
      FloatT tmp = abs_dx * abs_dx;
      return 2.0 / 3.0 - tmp + 0.5 * tmp * abs_dx;
    } else if (abs_dx < 2.0) {
      FloatT tmp = 2.0 - abs_dx;
      return 1.0 / 6.0 * tmp * tmp * tmp;
    } else {
      return 0.0;
    }
  }
};

template <typename Interp, typename FloatT>
FloatT HD_INLINE
interp_cell(const Interp& interp, FloatT rel_pos, int c, int t) {
  // The actual distance between particle and t
  FloatT x = (FloatT)t - (rel_pos + (FloatT)c);
  return interp(x);
}

template <typename Interp, typename FloatT>
FloatT HD_INLINE
interp_cell(const Interp& interp, FloatT rel_pos, int c, int t, int stagger) {
  // The actual distance between particle and t
  FloatT x = ((FloatT)t + (stagger == 1 ? 0.0 : 0.5)) - (rel_pos + (FloatT)c);
  return interp(x);
}

template <typename Interp, int Dim>
struct interpolator;

template <typename Interp>
struct interpolator<Interp, 1> {
  Interp interp;

  template <typename Ptr, typename Index_t, typename FloatT>
  HOST_DEVICE auto operator()(const Ptr& f, const vec_t<FloatT, 3>& x,
                              const Index_t& idx) ->
      typename Ptr::value_t {
    typename Ptr::value_t result = 0.0;
#pragma unroll
    for (int i = 1 - Interp::radius; i <= Interp::support - Interp::radius; i++) {
      // int ii = i + pos[0] - Interp::radius;
      result += f[idx.inc_x(i)] * interp_cell(interp, x[0], 0, i);
    }
    return result;
  }

  template <typename Ptr, typename Index_t, typename FloatT>
  HOST_DEVICE auto operator()(const Ptr& f, const vec_t<FloatT, 3>& x,
                              const Index_t& idx, stagger_t stagger) ->
      typename Ptr::value_t {
    typename Ptr::value_t result = 0.0;
#pragma unroll
    for (int i = stagger[0] - Interp::radius; i <= Interp::support - Interp::radius; i++) {
      // int ii = i + pos[0] - Interp::radius;
      result += f[idx.inc_x(i)] * interp_cell(interp, x[0], 0, i, stagger[0]);
    }
    return result;
  }
};

template <typename Interp>
struct interpolator<Interp, 2> {
  Interp interp;

  template <typename Ptr, typename Index_t, typename FloatT>
  HOST_DEVICE auto operator()(const Ptr& f, const vec_t<FloatT, 3>& x,
                              const Index_t& idx) ->
      typename Ptr::value_t {
    typename Ptr::value_t result = 0.0;
#pragma unroll
    for (int j = 1 - Interp::radius; j <= Interp::support - Interp::radius;
         j++) {
#pragma unroll
      for (int i = 1 - Interp::radius; i <= Interp::support - Interp::radius;
           i++) {
        result += f[idx.inc_x(i).inc_y(j)] *
                  interp_cell(interp, x[0], 0, i) *
                  interp_cell(interp, x[1], 0, j);
      }
    }
    return result;
  }

  template <typename Ptr, typename Index_t, typename FloatT>
  HOST_DEVICE auto operator()(const Ptr& f, const vec_t<FloatT, 3>& x,
                              const Index_t& idx, stagger_t stagger) ->
      typename Ptr::value_t {
    typename Ptr::value_t result = 0.0;
#pragma unroll
    for (int j = stagger[1] - Interp::radius; j <= Interp::support - Interp::radius;
         j++) {
#pragma unroll
      for (int i = stagger[0] - Interp::radius; i <= Interp::support - Interp::radius;
           i++) {
        result += f[idx.inc_x(i).inc_y(j)] *
                  interp_cell(interp, x[0], 0, i, stagger[0]) *
                  interp_cell(interp, x[1], 0, j, stagger[1]);
      }
    }
    return result;
  }

};

template <typename Interp>
struct interpolator<Interp, 3> {
  Interp interp;

  template <typename Ptr, typename Index_t, typename FloatT>
  HOST_DEVICE auto operator()(const Ptr& f, const vec_t<FloatT, 3>& x,
                              const Index_t& idx) ->
      typename Ptr::value_t {
    typename Ptr::value_t result = 0.0;
#pragma unroll
    for (int k = 1 - Interp::radius; k <= Interp::support - Interp::radius;
         k++) {
#pragma unroll
      for (int j = 1 - Interp::radius; j <= Interp::support - Interp::radius;
           j++) {
#pragma unroll
        for (int i = 1 - Interp::radius; i <= Interp::support - Interp::radius;
             i++) {
          result += f[idx.inc_x(i).inc_y(j).inc_z(k)] *
                    interp_cell(interp, x[0], 0, i) *
                    interp_cell(interp, x[1], 0, j) *
                    interp_cell(interp, x[2], 0, k);
        }
      }
    }
    return result;
  }

  template <typename Ptr, typename Index_t, typename FloatT>
  HOST_DEVICE auto operator()(const Ptr& f, const vec_t<FloatT, 3>& x,
                              const Index_t& idx, stagger_t stagger) ->
      typename Ptr::value_t {
    typename Ptr::value_t result = 0.0;
#pragma unroll
    for (int k = stagger[2] - Interp::radius; k <= Interp::support - Interp::radius;
         k++) {
#pragma unroll
      for (int j = stagger[1] - Interp::radius; j <= Interp::support - Interp::radius;
           j++) {
#pragma unroll
        for (int i = stagger[0] - Interp::radius; i <= Interp::support - Interp::radius;
             i++) {
          result += f[idx.inc_x(i).inc_y(j).inc_z(k)] *
                    interp_cell(interp, x[0], 0, i, stagger[0]) *
                    interp_cell(interp, x[1], 0, j, stagger[1]) *
                    interp_cell(interp, x[2], 0, k, stagger[2]);
        }
      }
    }
    return result;
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

    return 0.5 * (f[idx.inc_x(dx_m)] + f[idx.inc_x(dx_p)]);
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

#endif  // __INTERPOLATION_H_
