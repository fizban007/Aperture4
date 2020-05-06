#ifndef __MATH_H_
#define __MATH_H_

#include "core/cuda_control.h"
#include <cmath>

namespace Aperture {

namespace math {

// HD_INLINE double abs(double x) { return fabs(x); }
HD_INLINE float abs(float x) { return fabsf(x); }

// HD_INLINE double sin(double x) { return sin(x); }
HD_INLINE float sin(float x) { return sinf(x); }

// HD_INLINE double cos(double x) { return cos(x); }
HD_INLINE float cos(float x) { return cosf(x); }

// HD_INLINE double acos(double x) { return acos(x); }
HD_INLINE float acos(float x) { return acosf(x); }

// HD_INLINE double atan2(double y, double x) { return atan2(y, x); }
HD_INLINE float atan2(float y, float x) { return atan2f(y, x); }

// HD_INLINE double exp(double x) { return exp(x); }
HD_INLINE float exp(float x) { return expf(x); }

// HD_INLINE double log(double x) { return log(x); }
HD_INLINE float log(float x) { return logf(x); }

// HD_INLINE double sqrt(double x) { return sqrt(x); }
HD_INLINE float sqrt(float x) { return sqrtf(x); }

// HD_INLINE double floor(double x) { return floor(x); }
HD_INLINE float floor(float x) { return floorf(x); }

}


}

#endif // __MATH_H_
