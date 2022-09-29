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

#include "core/cuda_control.h"

namespace Aperture {

#if (defined(CUDA_ENABLED) && defined(__CUDACC__)) || \
    (defined(HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))

extern __constant__ double dev_gauss_xs[5];
extern __constant__ double dev_gauss_ws[5];

template <typename Func>
__device__ double
gauss_quad_dev(const Func& f, double a, double b) {
  double xm = 0.5 * (b + a);
  double xr = 0.5 * (b - a);
  double result = 0.0;
  for (int i = 0; i < 5; i++) {
    double dx = xr * dev_gauss_xs[i];
    result += dev_gauss_ws[i] * (f(xm + dx) + f(xm - dx));
  }

  return result * xr;
}

#endif

template <typename Func>
double
gauss_quad(const Func& f, double a, double b) {
  static double gauss_xs[5] = {0.1488743389816312, 0.4333953941292472,
                               0.6794095682990244, 0.8650633666889845,
                               0.9739065285171717};

  static double gauss_ws[5] = {0.2955242247147529, 0.2692667193099963,
                               0.2190863625159821, 0.1494513491505806,
                               0.0666713443086881};
  double xm = 0.5 * (b + a);
  double xr = 0.5 * (b - a);
  double result = 0.0;
  for (int i = 0; i < 5; i++) {
    double dx = xr * gauss_xs[i];
    result += gauss_ws[i] * (f(xm + dx) + f(xm - dx));
  }

  return result * xr;
}

}  // namespace Aperture
