/*
 * Copyright (c) 2021 Alex Chen.
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

#ifndef __IC_SCATTERING_H_
#define __IC_SCATTERING_H_

#include "core/cuda_control.h"
#include "core/math.hpp"
#include "core/multi_array.hpp"
#include "core/typedefs_and_constants.h"
#include "utils/binary_search.h"
#include "utils/util_functions.h"

namespace Aperture {

struct ic_scatter_t {
  using value_t = Scalar;
  using spec_array_t =
      typename multi_array<value_t, 2, idx_col_major_t<2>>::cref_t;

  HD_INLINE int array_upper_bound(value_t u, int n, const spec_array_t& array,
                                  value_t& l, value_t& h) const {
    const auto& spec_ext = array.ext();
    auto idx = idx_col_major_t<2>(index(0, n), spec_ext);
#if (defined(CUDA_ENABLED) & defined(__CUDACC__)) || \
    (defined(HIP_ENABLED) & defined(__HIPCC__))
    auto result =
        upper_bound(u, array.dev_ptr().p + n * spec_ext[0], spec_ext[0]);
#else
    auto result =
        upper_bound(u, array.ptr().p + n * spec_ext[0], spec_ext[0]);
#endif
    if (result <= spec_ext[0] - 1) {
      l = array[idx + result - 1];
      h = array[idx + result];
      return result;
    } else {
      return result;
    }
  }

  HD_INLINE int find_n_gamma(value_t gamma) const {
    return clamp((int)floor(math::log(gamma) / dgamma), 0, dNde.ext()[1] - 1);
  }

  // u is a generated random number between 0 and 1
  template <typename Rng>
  HOST_DEVICE value_t gen_photon_e(value_t gamma, Rng& rng) const {
    int n_gamma = find_n_gamma(gamma);
    value_t l, h;
    value_t u = rng.template uniform<value_t>();
    int b = array_upper_bound(u, n_gamma, dNde, l, h);
    // TODO: This is an arbitrary division
    if (b < 1 || gamma < 2.0) {
      u = rng.template uniform<value_t>();
      b = array_upper_bound(u, n_gamma, dNde_thomson, l, h);
      value_t bb = (u - l) / (h - l) + b - 1;

      return clamp(math::exp(dlep * bb) * min_ep, 0.0, 1.0);
    } else if (b >= dNde.ext()[0]) {
      // return clamp(dep * dNde.ext()[0], 0.0, gamma - 1.01);
      return clamp(dep * dNde.ext()[0], 0.0, 1.0);
    } else {
      value_t bb = (u - l) / (h - l) + b - 1;
      // return clamp(dep * bb, 0.0, gamma - 1.01);
      return clamp(dep * bb, 0.0, 1.0);
    }
  }

  HOST_DEVICE value_t ic_scatter_rate(value_t gamma) const {
    int n_gamma = find_n_gamma(gamma);
    if (n_gamma >= dNde.ext()[1] - 1) return ic_rate[n_gamma];
    value_t x = (math::log(gamma) - dgamma * n_gamma) / dgamma;
    return ic_rate[n_gamma] * (1.0f - x) + ic_rate[n_gamma + 1] * x;
  }

  HOST_DEVICE value_t gg_scatter_rate(value_t eph) const {
    if (eph < 2.0f) return 0.0f;
    int n_eph = find_n_gamma(eph);
    if (n_eph >= dNde.ext()[1] - 1) return gg_rate[n_eph];
    value_t x = (math::log(eph) - dgamma * n_eph) / dgamma;
    return gg_rate[n_eph] * (1.0f - x) + gg_rate[n_eph + 1] * x;
  }

  static HD_INLINE value_t gamma(int n, value_t dgamma) {
    return math::exp(n * dgamma);
  }

  static HD_INLINE value_t ep(int n, value_t dep) {
    return n * dep;
  }

  static HD_INLINE value_t e_log(int n, value_t dloge, value_t min_e) {
    return min_e * math::exp(n * dloge);
  }

  spec_array_t dNde;
  spec_array_t dNde_thomson;
  value_t* ic_rate;
  value_t* gg_rate;
  value_t min_ep, dgamma, dep, dlep;
  value_t compactness, e_mean;
};

}  // namespace Aperture

#endif  // __IC_SCATTERING_H_
