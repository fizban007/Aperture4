#ifndef __INTERPOLATION_H_
#define __INTERPOLATION_H_

#include "core/cuda_control.h"

namespace Aperture {

template <typename Ptr, typename Index>
HOST_DEVICE float
lerp3(const Ptr& f, float x, float y, float z, const Index& idx) {
  float f11 = (1.0f - z) * f[idx.inc_x().inc_y()] +
              z * f[idx.inc_x().inc_y().inc_z()];
  float f10 = (1.0f - z) * f[idx.inc_x()] + z * f[idx.inc_x().inc_z()];
  float f01 = (1.0f - z) * f[idx.inc_y()] + z * f[idx.inc_y().inc_z()];
  float f00 = (1.0f - z) * f[idx] + z * f[idx.inc_z()];
  float f1 = y * f11 + (1.0f - y) * f10;
  float f0 = y * f01 + (1.0f - y) * f00;
  return x * f1 + (1.0f - x) * f0;
}

}  // namespace Aperture

#endif  // __INTERPOLATION_H_
