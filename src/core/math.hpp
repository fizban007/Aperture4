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

#ifndef __MATH_H_
#define __MATH_H_

#include "core/cuda_control.h"
#include <cmath>

namespace Aperture {

namespace math {

HD_INLINE double abs(double x) { return std::abs(x); }
HD_INLINE float abs(float x) { return fabsf(x); }

HD_INLINE double sin(double x) { return ::sin(x); }
HD_INLINE float sin(float x) { return sinf(x); }

HD_INLINE double cos(double x) { return ::cos(x); }
HD_INLINE float cos(float x) { return cosf(x); }

HD_INLINE double acos(double x) { return ::acos(x); }
HD_INLINE float acos(float x) { return acosf(x); }

HD_INLINE double atan2(double y, double x) { return ::atan2(y, x); }
HD_INLINE float atan2(float y, float x) { return atan2f(y, x); }

HD_INLINE double exp(double x) { return ::exp(x); }
HD_INLINE float exp(float x) { return expf(x); }

HD_INLINE double log(double x) { return ::log(x); }
HD_INLINE float log(float x) { return logf(x); }

HD_INLINE double sqrt(double x) { return ::sqrt(x); }
HD_INLINE float sqrt(float x) { return sqrtf(x); }

HD_INLINE double floor(double x) { return ::floor(x); }
HD_INLINE float floor(float x) { return floorf(x); }

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

}


}

#endif // __MATH_H_
