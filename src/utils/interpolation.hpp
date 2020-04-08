#ifndef __INTERPOLATION_H_
#define __INTERPOLATION_H_

#include "core/cuda_control.h"

namespace Aperture {

template <typename Ptr, typename Index>
HOST_DEVICE float
lerp3(const Ptr& f, float x, float y, float z, const Index& idx) {
  float f11 =
      (1.0f - z) * f[idx.template inc<0>().template inc<1>()] +
      z * f[idx.template inc<0>()
                .template inc<1>()
                .template inc<2>()
                ];
  float f10 = (1.0f - z) * f[idx.template inc<0>()] +
              z * f[idx.template inc<0>().template inc<2>()];
  float f01 = (1.0f - z) * f[idx.template inc<1>()] +
              z * f[idx.template inc<1>().template inc<2>()];
  float f00 =
      (1.0f - z) * f[idx] + z * f[idx.template inc<2>()];
  float f1 = y * f11 + (1.0f - y) * f10;
  float f0 = y * f01 + (1.0f - y) * f00;
  return x * f1 + (1.0f - x) * f0;
}


}  // namespace Aperture

#endif  // __INTERPOLATION_H_
