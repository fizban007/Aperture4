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

  HD_INLINE double operator()(double dx) const {
    return (std::abs(dx) <= 0.5 ? 1.0 : 0.0);
  }
};

template <>
struct bspline<1> {
  enum { radius = 1, support = 2 };

  HD_INLINE double operator()(double dx) const {
    double abs_dx = std::abs(dx);
    return std::max(1.0 - abs_dx, 0.0);
  }
};

template <>
struct bspline<2> {
  enum { radius = 2, support = 3 };

  HD_INLINE double operator()(double dx) const {
    double abs_dx = std::abs(dx);
    if (abs_dx < 0.5) {
      return 0.75 - dx * dx;
    } else if (abs_dx < 1.5) {
      double tmp = 1.5 - abs_dx;
      return 0.5 * tmp * tmp;
    } else {
      return 0.0;
    }
  }
};

template <>
struct bspline<3> {
  enum { radius = 2, support = 4 };

  HD_INLINE double operator()(double dx) const {
    double abs_dx = std::abs(dx);
    if (abs_dx < 1.0) {
      double tmp = abs_dx * abs_dx;
      return 2.0 / 3.0 - tmp + 0.5 * tmp * abs_dx;
    } else if (abs_dx < 2.0) {
      double tmp = 2.0 - abs_dx;
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
                              const Index_t& idx, const index_t<1>& pos) ->
      typename Ptr::value_t {
    typename Ptr::value_t result = 0.0;
#pragma unroll
    for (int i = 1; i <= Interp::support; i++) {
      int ii = i + pos[0] - Interp::radius;
      result += f[ii] * interp_cell(interp, x[0], pos[0], ii);
    }
    return result;
  }
};

template <typename Interp>
struct interpolator<Interp, 2> {
  Interp interp;

  template <typename Ptr, typename Index_t, typename FloatT>
  HOST_DEVICE auto operator()(const Ptr& f, const vec_t<FloatT, 3>& x,
                              const Index_t& idx, const index_t<2>& pos) ->
      typename Ptr::value_t {
    typename Ptr::value_t result = 0.0;
#pragma unroll
    for (int j = 1 - Interp::radius; j <= Interp::support - Interp::radius;
         j++) {
      int jj = j + pos[1];
#pragma unroll
      for (int i = 1 - Interp::radius; i <= Interp::support - Interp::radius;
           i++) {
        int ii = i + pos[0];
        // printf("idx is %lu\n", idx.inc_x(i).inc_y(j).linear);
        // printf("x is %f, %f, pos is %d, %d\n", x[0], x[1], pos[0], pos[1]);
        // printf("f is %f, interp is %f, %f\n", f[idx.inc_x(i).inc_y(j)],
        //        interp_cell(interp, x[0], pos[0], ii),
        //        interp_cell(interp, x[1], pos[1], jj));
        result += f[idx.inc_x(i).inc_y(j)] *
                  interp_cell(interp, x[0], pos[0], ii) *
                  interp_cell(interp, x[1], pos[1], jj);
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
                              const Index_t& idx, const index_t<3>& pos) ->
      typename Ptr::value_t {
    typename Ptr::value_t result = 0.0;
#pragma unroll
    for (int k = 1 - Interp::radius; k <= Interp::support - Interp::radius;
         k++) {
      int kk = k + pos[2];
#pragma unroll
      for (int j = 1 - Interp::radius; j <= Interp::support - Interp::radius;
           j++) {
        int jj = j + pos[1];
#pragma unroll
        for (int i = 1 - Interp::radius; i <= Interp::support - Interp::radius;
             i++) {
          int ii = i + pos[0];
          result += f[idx.inc_x(i).inc_y(j).inc_z(k)] *
                    interp_cell(interp, x[0], pos[0], ii) *
                    interp_cell(interp, x[1], pos[1], jj) *
                    interp_cell(interp, x[2], pos[2], kk);
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
                                 const vec_t<FloatT, 3>& x, const Index_t& idx,
                                 const stagger_t& st_in,
                                 const stagger_t& st_out) {
    return x[0] * f[idx.inc_x()] + (1.0f - x[0]) * f[idx];
  }
};

// 2D interpolation
template <>
struct lerp<2> {
  template <class Value_t, typename Index_t, typename FloatT>
  HOST_DEVICE Value_t operator()(const ndptr<Value_t, 2, Index_t>& f,
                                 const vec_t<FloatT, 3>& x, const Index_t& idx,
                                 const stagger_t& st_in,
                                 const stagger_t& st_out) {
    FloatT f1 = x[1] * f[idx.inc_x().inc_y()] + (1.0f - x[1]) * f[idx.inc_x()];
    FloatT f0 = x[1] * f[idx.inc_y()] + (1.0f - x[1]) * f[idx];
    return x[0] * f1 + (1.0f - x[0]) * f0;
  }
};

// 3D interpolation
template <>
struct lerp<3> {
  template <class Value_t, typename Index_t, typename FloatT>
  HOST_DEVICE Value_t operator()(const ndptr<Value_t, 3, Index_t>& f,
                                 const vec_t<FloatT, 3>& x, const Index_t& idx,
                                 const stagger_t& st_in,
                                 const stagger_t& st_out) {
    FloatT f11 = (1.0f - x[2]) * f[idx.inc_x().inc_y()] +
                 x[2] * f[idx.inc_x().inc_y().inc_z()];
    FloatT f10 = (1.0f - x[2]) * f[idx.inc_x()] + x[2] * f[idx.inc_x().inc_z()];
    FloatT f01 = (1.0f - x[2]) * f[idx.inc_y()] + x[2] * f[idx.inc_y().inc_z()];
    FloatT f00 = (1.0f - x[2]) * f[idx] + x[2] * f[idx.inc_z()];
    FloatT f1 = x[1] * f11 + (1.0f - x[1]) * f10;
    FloatT f0 = x[1] * f01 + (1.0f - x[1]) * f00;
    return x[0] * f1 + (1.0f - x[0]) * f0;
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
