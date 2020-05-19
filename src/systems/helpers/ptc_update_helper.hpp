#ifndef __PTC_UPDATE_HELPER_H_
#define __PTC_UPDATE_HELPER_H_

#include "core/cuda_control.h"

namespace Aperture {

template <typename FloatT>
struct EB_t {
  FloatT E1, E2, E3, B1, B2, B3;
};

template <typename FloatT>
HD_INLINE FloatT
center2d(FloatT sx0, FloatT sx1, FloatT sy0, FloatT sy1) {
  return (2.0f * sx1 * sy1 + sx0 * sy1 + sx1 * sy0 + 2.0f * sx0 * sy0) *
         0.166666666667f;
}

template <typename FloatT>
HD_INLINE FloatT
movement3d(FloatT sx0, FloatT sx1, FloatT sy0, FloatT sy1, FloatT sz0,
           FloatT sz1) {
  return (sz1 - sz0) * center2d(sx0, sx1, sy0, sy1);
}

template <typename FloatT>
HD_INLINE FloatT
movement2d(FloatT sx0, FloatT sx1, FloatT sy0, FloatT sy1) {
  return (sy1 - sy0) * 0.5f * (sx0 + sx1);
}



}

#endif // __PTC_UPDATE_HELPER_H_
