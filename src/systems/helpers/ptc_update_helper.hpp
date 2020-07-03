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

#ifndef __PTC_UPDATE_HELPER_H_
#define __PTC_UPDATE_HELPER_H_

#include "core/cuda_control.h"
#include "core/particle_structs.h"
#include <cstdint>

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

template <typename Pusher>
struct pusher_impl_t {
  Pusher pusher;

  template <typename Scalar>
  HD_INLINE void operator()(ptc_ptrs& ptc, uint32_t n, EB_t<Scalar>& EB,
                            Scalar qdt_over_2m, Scalar dt) {
    pusher(ptc.p1[n], ptc.p2[n], ptc.p3[n], ptc.E[n], EB.E1, EB.E2,
           EB.E3, EB.B1, EB.B2, EB.B3, qdt_over_2m, dt);
  }
};


}

#endif // __PTC_UPDATE_HELPER_H_
