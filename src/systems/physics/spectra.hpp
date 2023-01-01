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

#pragma once

#include "core/cuda_control.h"
#include "core/math.hpp"
#include "core/typedefs_and_constants.h"
#include "utils/util_functions.h"

namespace Aperture {

namespace Spectra {

struct power_law_hard {
  HOST_DEVICE power_law_hard(Scalar delta, Scalar emin, Scalar emax)
      : delta_(delta), emin_(emin), emax_(emax) {}

  HD_INLINE Scalar operator()(Scalar e) const {
    if (e < emax_ && e > emin_)
      return delta_ * pow(e / emax_, delta_) / e;
    else
      return 0.0;
  }

  HD_INLINE Scalar emean() const { return delta_ * emax_ / (1.0f + delta_); }
  HD_INLINE Scalar emin() const { return emin_; }
  HD_INLINE Scalar emax() const { return emax_; }

  Scalar delta_, emin_, emax_;
};

struct power_law_soft {
  HOST_DEVICE power_law_soft(Scalar alpha, Scalar emin, Scalar emax)
      : alpha_(alpha), emin_(emin), emax_(emax) {}

  HD_INLINE Scalar operator()(Scalar e) const {
    if (e < emax_ && e > emin_)
      return alpha_ * pow(e / emin_, -alpha_) / e;
    else
      return 0.0;
  }

  HD_INLINE Scalar emean() const {
    if (alpha_ > 1.0f) {
      return emin_ * alpha_ / (alpha_ - 1.0f);
    } else {
      return (emax_ * pow(emax_ / emin_, -alpha_) - emin_) / (1.0f - alpha_);
    }
  }
  HD_INLINE Scalar emin() const { return emin_; }
  HD_INLINE Scalar emax() const { return emax_; }

  Scalar alpha_, emin_, emax_;
};

struct broken_power_law {
  HOST_DEVICE broken_power_law(Scalar alpha, Scalar delta, Scalar ep,
                               Scalar emin, Scalar emax)
      : alpha_(alpha), delta_(delta), epeak_(ep), emin_(emin), emax_(emax) {}

  HD_INLINE Scalar operator()(Scalar e) const {
    if (e < epeak_ && e > emin_)
      return pow(e / epeak_, delta_) / e;
    else if (e > epeak_ && e < emax_)
      return pow(e / epeak_, -alpha_) / e;
    else
      return 0.0;
  }

  HD_INLINE Scalar emean() const {
    Scalar e1 =
        (emax_ - epeak_ * pow(epeak_ / emax_, delta_)) / (1.0f + delta_);
    Scalar e2 =
        (epeak_ * pow(epeak_ / emin_, -alpha_) - emin_) / (alpha_ - 1.0f);
    return e1 + e2;
  }

  HD_INLINE Scalar emin() const { return emin_; }
  HD_INLINE Scalar emax() const { return emax_; }

  Scalar alpha_, delta_, epeak_, emin_, emax_;
};

struct black_body {
  HOST_DEVICE black_body(Scalar kT) : kT_(kT) {}

  HD_INLINE double operator()(double e) const {
    // The normalization factor comes as 8 \pi/(h^3 c^3) (me c^2)^3
    // return 1.75464e30 * e * e / (exp(e / kT_) - 1.0);

    // The integral of e^2/(exp(e/kT) - 1) is 2*(kT)^3*Zeta(3)
    return e * e / (exp(e / kT_) - 1.0) / (2.40411 * cube(kT_));
  }

  HD_INLINE Scalar emean() const { return 2.7f * kT_; }
  HD_INLINE Scalar emin() const { return 1e-10 * kT_; }
  HD_INLINE Scalar emax() const { return 1e3 * kT_; }

  Scalar kT_;
};

struct mono_energetic {
  HOST_DEVICE mono_energetic(Scalar e0, Scalar de) : e0_(e0), de_(de) {}

  HD_INLINE Scalar operator()(Scalar e) const {
    if (e < e0_ + de_ && e > e0_ - de_)
      return 1.0 / (2.0 * de_);
    else
      return 0.0;
  }

  HD_INLINE Scalar emean() const { return e0_; }
  HD_INLINE Scalar emin() const { return e0_ * 1e-2; }
  HD_INLINE Scalar emax() const { return e0_ * 1e2; }

  Scalar e0_, de_;
};

}  // namespace Spectra

}  // namespace Aperture
