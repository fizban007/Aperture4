/*
 * Copyright (c) 2026 Alex Chen.
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
#include "utils/util_functions.h"

namespace Aperture {

// ===========================================================================
// Landau-Lifshitz radiation reaction (synchrotron drag).
//
// The reduced LL force, with the particle's own acceleration eliminated in
// favour of the zeroth-order Lorentz force:
//
//   F_rad = c_r [ (E + v x B) x B + E (u.E)/gamma
//                 - u gamma ( (E + v x B)^2 - ((u.E)/gamma)^2 ) ]
//
// with v = u/gamma and u the spatial 4-velocity.  This returns the drag
// ALONE -- no Lorentz force -- so it composes with any pusher (Boris,
// Vay, ...) as an operator split.
//
// Two properties matter for how it is used:
//
//  - At large gamma the last term dominates and is ANTI-PARALLEL to u.
//    The drag is then pure energy loss with the momentum direction left
//    alone, i.e. THE PITCH ANGLE IS PRESERVED.  This is the physically
//    correct high-gamma synchrotron behaviour, and the reason this force
//    is used in preference to schemes that damp the perpendicular
//    momentum directly (those drive the pitch angle to zero at every
//    gamma, which is wrong).
//
//  - It is regular everywhere.  There is no drift frame and no 1/B, so
//    E >= B and B -> 0 need no guarding: the drag falls smoothly to zero
//    with the fields.  Regions where the field vanishes (a reconnection
//    layer's interior) stay uncooled on their own, without a special case.
//
// c_r ("cooling_coef") is 2 r_e t_0 / 3c in code units; see
// pusher_synchrotron::init() for the compactness / gamma_rad
// parameterizations that set it.
// ===========================================================================
template <typename value_t>
HD_INLINE void
sync_drag_force(value_t u1, value_t u2, value_t u3, value_t gamma, value_t E1,
                value_t E2, value_t E3, value_t B1, value_t B2, value_t B3,
                value_t cooling_coef, value_t& f1, value_t& f2, value_t& f3) {
  // E' = E + v x B, the field in the (instantaneous) particle frame
  value_t ep1 = E1 + (u2 * B3 - u3 * B2) / gamma;
  value_t ep2 = E2 + (u3 * B1 - u1 * B3) / gamma;
  value_t ep3 = E3 + (u1 * B2 - u2 * B1) / gamma;

  value_t uE = u1 * E1 + u2 * E2 + u3 * E3;
  value_t ep_sq = ep1 * ep1 + ep2 * ep2 + ep3 * ep3;
  // gamma ( E'^2 - (u.E/gamma)^2 ) -- the scalar multiplying -u
  value_t drag = gamma * (ep_sq - square(uE / gamma));

  f1 = cooling_coef * ((ep2 * B3 - ep3 * B2) + E1 * uE / gamma - u1 * drag);
  f2 = cooling_coef * ((ep3 * B1 - ep1 * B3) + E2 * uE / gamma - u2 * drag);
  f3 = cooling_coef * ((ep1 * B2 - ep2 * B1) + E3 * uE / gamma - u3 * drag);
}

// ===========================================================================
// One radiation-reaction substep, applied AFTER a pusher has already taken
// the Lorentz force (operator split).
//
// The force above splits exactly into a decay rate and a residual,
//
//   du/dt = -nu u + g,
//   nu = c_r gamma ( E'^2 - (u.E/gamma)^2 ),
//   g  = c_r ( E' x B + E (u.E)/gamma ),
//
// which is integrated with an EXPONENTIAL midpoint step: nu and g are
// frozen at the midpoint (recovered by fixed-point iteration) and the
// resulting linear ODE is solved exactly,
//
//   u^{n+1} = u^n exp(-nu dt) + g (1 - exp(-nu dt)) / nu.
//
// This matters because the locked limit these runs target sits at a
// cooling time of order the STEP.  A plain implicit-midpoint fixed point
// (what pusher_synchrotron::iterate does) stops contracting once
// nu dt >~ 1 and diverges to NaN; the exponential form is bounded for
// every nu dt -- as nu dt -> inf it relaxes onto u -> g/nu, which is the
// exact fixed point of the ODE, so the strongly-cooled endpoint is right
// rather than merely finite.  Fixed points are shared with the ODE at any
// stiffness, so an over-cooled particle lands on the drift solution
// instead of on zero.
//
// gamma is recomputed from the updated momentum on exit.
// ===========================================================================
template <typename ptc_float_t, typename value_t>
HD_INLINE void
sync_drag_substep(ptc_float_t& u1, ptc_float_t& u2, ptc_float_t& u3,
                  ptc_float_t& gamma, value_t E1, value_t E2, value_t E3,
                  value_t B1, value_t B2, value_t B3, value_t cooling_coef,
                  value_t dt, int n_iter = 4) {
  if (cooling_coef == value_t(0)) return;
  const value_t u01 = u1, u02 = u2, u03 = u3;
  value_t n1 = u1, n2 = u2, n3 = u3;

  for (int i = 0; i < n_iter; i++) {
    value_t m1 = value_t(0.5) * (u01 + n1);
    value_t m2 = value_t(0.5) * (u02 + n2);
    value_t m3 = value_t(0.5) * (u03 + n3);
    value_t gm = math::sqrt(value_t(1) + m1 * m1 + m2 * m2 + m3 * m3);

    value_t ep1 = E1 + (m2 * B3 - m3 * B2) / gm;
    value_t ep2 = E2 + (m3 * B1 - m1 * B3) / gm;
    value_t ep3 = E3 + (m1 * B2 - m2 * B1) / gm;
    value_t uE = m1 * E1 + m2 * E2 + m3 * E3;
    value_t ep_sq = ep1 * ep1 + ep2 * ep2 + ep3 * ep3;

    value_t nu = cooling_coef * gm * (ep_sq - square(uE / gm));
    value_t g1 = cooling_coef * ((ep2 * B3 - ep3 * B2) + E1 * uE / gm);
    value_t g2 = cooling_coef * ((ep3 * B1 - ep1 * B3) + E2 * uE / gm);
    value_t g3 = cooling_coef * ((ep1 * B2 - ep2 * B1) + E3 * uE / gm);

    // nu < 0 is possible where E_par dominates ((u.E/gamma)^2 > E'^2); the
    // reduced LL form then transfers energy TO the particle.  Bound the
    // growth per step so a pathological cell cannot run away.
    value_t x = nu * dt;
    if (x < value_t(-1)) x = value_t(-1);
    value_t ex = math::exp(-x);
    // dt (1 - e^-x)/x, the phi_1 function; series near x = 0.
    value_t phi = (math::abs(x) < value_t(1e-4))
                      ? dt * (value_t(1) - value_t(0.5) * x)
                      : dt * (value_t(1) - ex) / x;

    n1 = u01 * ex + g1 * phi;
    n2 = u02 * ex + g2 * phi;
    n3 = u03 * ex + g3 * phi;
  }

  u1 = n1;
  u2 = n2;
  u3 = n3;
  gamma = math::sqrt(value_t(1) + n1 * n1 + n2 * n2 + n3 * n3);
}

}  // namespace Aperture
