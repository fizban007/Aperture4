#pragma once

#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_gidx.h"
#include "systems/prismatic/prismatic_mesh_ptrs.h"
#include <cmath>

namespace Aperture {

// =========================================================================
// Spherical-geometry helpers for face/edge quadratures, plus the analytic
// dipole / Deutsch field evaluators.  Shared between the global
// dec_field_solver and the distributed solver core (dec_solver_dist);
// framework-free so tests can use them directly.
//
// Mesh vertices are stored in (r, θ, φ) and faces live on the sphere
// surface, not on chord triangles.  All position-evaluations in the
// quadratures below go through these helpers so that the Gauss samples
// actually fall on the curved primal element.
// =========================================================================

// Fetch (r, unit-direction) for global vertex index vi.  The angular part
// is read from the Cartesian sphere_v{x,y,z} buffer (which we retain for
// particle operations) to avoid recomputing cos/sin per Gauss point.
HD_INLINE void vertex_unit(const prismatic_mesh_ptrs& mp, gidx_t vi,
                           Scalar& r, Scalar& ux, Scalar& uy, Scalar& uz) {
  int k = int(vi / mp.N_vert_s);
  int s = int(vi % mp.N_vert_s);
  r = mp.radii[k];
  ux = mp.sphere_vx[s];
  uy = mp.sphere_vy[s];
  uz = mp.sphere_vz[s];
}

// =========================================================================
// Phase 7D — analytic vertex-id decode for the BC/IC quadratures.  The
// global tri_face_v* / rect_face_v* / edge_v0/v1 helper tables no longer
// exist under a sphere-only mesh; the ids they held are pure arithmetic
// over the persisted sphere tables (IDENTICAL integers — the tables were
// built from exactly these expressions).
// =========================================================================
HD_INLINE void tri_face_vertex_ids(const prismatic_mesh_ptrs& mp, gidx_t g,
                                   gidx_t& vi0, gidx_t& vi1, gidx_t& vi2) {
  const int k = int(g / mp.N_tri), t = int(g - gidx_t(k) * mp.N_tri);
  vi0 = gidx_t(k) * mp.N_vert_s + mp.tri_verts[t * 3 + 0];
  vi1 = gidx_t(k) * mp.N_vert_s + mp.tri_verts[t * 3 + 1];
  vi2 = gidx_t(k) * mp.N_vert_s + mp.tri_verts[t * 3 + 2];
}

// Corners v0 = (k, a), v1 = (k, b), v3 = (k+1, a) of rect face g — the
// three the quadratures use (v2 = (k+1, b) is implied).
HD_INLINE void rect_face_vertex_ids(const prismatic_mesh_ptrs& mp, gidx_t g,
                                    gidx_t& vi0, gidx_t& vi1, gidx_t& vi3) {
  const int k = int(g / mp.N_edge_s), e = int(g - gidx_t(k) * mp.N_edge_s);
  vi0 = gidx_t(k) * mp.N_vert_s + mp.sphere_edge_v0[e];
  vi1 = gidx_t(k) * mp.N_vert_s + mp.sphere_edge_v1[e];
  vi3 = gidx_t(k + 1) * mp.N_vert_s + mp.sphere_edge_v0[e];
}

// Endpoints of h-edge g (global h-edge index in [0, (N_r+1)·N_edge_s)).
HD_INLINE void h_edge_vertex_ids(const prismatic_mesh_ptrs& mp, gidx_t g,
                                 gidx_t& v0, gidx_t& v1) {
  const int k = int(g / mp.N_edge_s), e = int(g - gidx_t(k) * mp.N_edge_s);
  v0 = gidx_t(k) * mp.N_vert_s + mp.sphere_edge_v0[e];
  v1 = gidx_t(k) * mp.N_vert_s + mp.sphere_edge_v1[e];
}

// Endpoints of v-edge g (global v-edge index in [0, N_r·N_vert_s)).
HD_INLINE void v_edge_vertex_ids(const prismatic_mesh_ptrs& mp, gidx_t g,
                                 gidx_t& v0, gidx_t& v1) {
  const int k = int(g / mp.N_vert_s), s = int(g - gidx_t(k) * mp.N_vert_s);
  v0 = gidx_t(k) * mp.N_vert_s + s;
  v1 = gidx_t(k + 1) * mp.N_vert_s + s;
}

// Slerp two unit vectors, plus its u-derivative.  At u=0 returns û_a, at
// u=1 returns û_b.  For very small α falls back to the linear tangent —
// the quadrature inner integrand handles the α→0 limit gracefully.
HD_INLINE void slerp_uv(Scalar ax, Scalar ay, Scalar az,
                        Scalar bx, Scalar by, Scalar bz, Scalar u,
                        Scalar& ux, Scalar& uy, Scalar& uz,
                        Scalar& dux, Scalar& duy, Scalar& duz) {
  Scalar dot = ax*bx + ay*by + az*bz;
  if (dot >  Scalar(1)) dot =  Scalar(1);
  if (dot < -Scalar(1)) dot = -Scalar(1);
  Scalar alpha = std::acos(dot);
  Scalar sa = std::sin(alpha);
  if (sa < Scalar(1e-12)) {
    ux = ax; uy = ay; uz = az;
    dux = bx - ax; duy = by - ay; duz = bz - az;
    return;
  }
  Scalar w0 = std::sin((Scalar(1) - u) * alpha) / sa;
  Scalar w1 = std::sin(u * alpha) / sa;
  ux = w0*ax + w1*bx;
  uy = w0*ay + w1*by;
  uz = w0*az + w1*bz;
  Scalar dw0 = -alpha * std::cos((Scalar(1) - u) * alpha) / sa;
  Scalar dw1 =  alpha * std::cos(u * alpha) / sa;
  dux = dw0*ax + dw1*bx;
  duy = dw0*ay + dw1*by;
  duz = dw0*az + dw1*bz;
}

// Spherical-triangle parametrization using radially-projected barycentric
// interpolation.  Domain: (u, t) ∈ [0, 1]²  with v = (1-u)·t  so that
//     λ_a = (1-u)(1-t),  λ_b = u,  λ_c = (1-u)·t
// Writes (x, y, z) on the sphere of radius r and (nx, ny, nz) = ∂P/∂u × ∂P/∂t,
// i.e. the vector surface element per du·dt (so the integrand in this
// parametrization is B · (nx,ny,nz) — the (1-u) Jacobian is already folded
// in through ∂λ/∂t factors).
HD_INLINE void tri_sphere_sample(Scalar r,
                                 Scalar ax, Scalar ay, Scalar az,
                                 Scalar bx, Scalar by, Scalar bz,
                                 Scalar cx, Scalar cy, Scalar cz,
                                 Scalar u, Scalar t,
                                 Scalar& x, Scalar& y, Scalar& z,
                                 Scalar& nx, Scalar& ny, Scalar& nz) {
  Scalar la = (Scalar(1) - u) * (Scalar(1) - t);
  Scalar lb = u;
  Scalar lc = (Scalar(1) - u) * t;
  Scalar qx = la*ax + lb*bx + lc*cx;
  Scalar qy = la*ay + lb*by + lc*cy;
  Scalar qz = la*az + lb*bz + lc*cz;
  Scalar qn = std::sqrt(qx*qx + qy*qy + qz*qz);
  Scalar ihx = qx / qn, ihy = qy / qn, ihz = qz / qn;
  x = r * ihx;  y = r * ihy;  z = r * ihz;

  // ∂λ/∂u = (-(1-t), 1, -t),  ∂λ/∂t = (-(1-u), 0, (1-u)).
  Scalar dqdu_x = -(Scalar(1) - t) * ax + bx - t * cx;
  Scalar dqdu_y = -(Scalar(1) - t) * ay + by - t * cy;
  Scalar dqdu_z = -(Scalar(1) - t) * az + bz - t * cz;
  Scalar dqdt_x = (Scalar(1) - u) * (cx - ax);
  Scalar dqdt_y = (Scalar(1) - u) * (cy - ay);
  Scalar dqdt_z = (Scalar(1) - u) * (cz - az);

  // ∂û/∂ξ = (I − û⊗û) · ∂Q/∂ξ / |Q|
  Scalar qinv = Scalar(1) / qn;
  Scalar pdu = ihx*dqdu_x + ihy*dqdu_y + ihz*dqdu_z;  // û·∂Q/∂u
  Scalar pdt = ihx*dqdt_x + ihy*dqdt_y + ihz*dqdt_z;
  Scalar duhx = qinv * (dqdu_x - pdu*ihx);
  Scalar duhy = qinv * (dqdu_y - pdu*ihy);
  Scalar duhz = qinv * (dqdu_z - pdu*ihz);
  Scalar dthx = qinv * (dqdt_x - pdt*ihx);
  Scalar dthy = qinv * (dqdt_y - pdt*ihy);
  Scalar dthz = qinv * (dqdt_z - pdt*ihz);

  // n = r² · (∂û/∂u × ∂û/∂t)
  Scalar r2 = r * r;
  nx = r2 * (duhy*dthz - duhz*dthy);
  ny = r2 * (duhz*dthx - duhx*dthz);
  nz = r2 * (duhx*dthy - duhy*dthx);
}

// Rectangular face (ruled surface between shells r0 and r1 along the
// great-circle arc û_a→û_b).  Parametrization:
//     r(v) = (1-v)·r0 + v·r1,   û(u) = slerp(û_a, û_b, u),
//     P    = r(v) · û(u).
// Writes (x, y, z) and (nx, ny, nz) = ∂P/∂u × ∂P/∂v.
HD_INLINE void rect_sphere_sample(Scalar r0, Scalar r1,
                                  Scalar ax, Scalar ay, Scalar az,
                                  Scalar bx, Scalar by, Scalar bz,
                                  Scalar u, Scalar v,
                                  Scalar& x, Scalar& y, Scalar& z,
                                  Scalar& nx, Scalar& ny, Scalar& nz) {
  Scalar ux, uy, uz, dux, duy, duz;
  slerp_uv(ax, ay, az, bx, by, bz, u, ux, uy, uz, dux, duy, duz);
  Scalar rv = (Scalar(1) - v) * r0 + v * r1;
  Scalar drdv = r1 - r0;
  x = rv * ux;  y = rv * uy;  z = rv * uz;
  // ∂P/∂u = rv · dû/du,  ∂P/∂v = drdv · û
  Scalar pux = rv * dux, puy = rv * duy, puz = rv * duz;
  Scalar pvx = drdv * ux, pvy = drdv * uy, pvz = drdv * uz;
  nx = puy*pvz - puz*pvy;
  ny = puz*pvx - pux*pvz;
  nz = pux*pvy - puy*pvx;
}

// Horizontal (arc) edge sample at parameter t ∈ [0,1] on the sphere of
// radius r.  Writes position (x, y, z) and line element dl = ∂P/∂t.
HD_INLINE void h_edge_sphere_sample(Scalar r,
                                    Scalar ax, Scalar ay, Scalar az,
                                    Scalar bx, Scalar by, Scalar bz,
                                    Scalar t,
                                    Scalar& x, Scalar& y, Scalar& z,
                                    Scalar& dlx, Scalar& dly, Scalar& dlz) {
  Scalar ux, uy, uz, dux, duy, duz;
  slerp_uv(ax, ay, az, bx, by, bz, t, ux, uy, uz, dux, duy, duz);
  x = r * ux;  y = r * uy;  z = r * uz;
  dlx = r * dux;  dly = r * duy;  dlz = r * duz;
}

// Lense-Thirring angular velocity profile for the fake-GR frame-drag
// terms: omega_lt(r) = w0 * (r_star / r)^p with integer p (3 physically;
// 0 gives a uniform drag, used by the unit tests where curl(v x B) is
// analytic).  About the SPIN axis z by the code's convention (obliquity
// tilts the magnetic axis, never Omega).
HD_INLINE Scalar frame_drag_omega(Scalar r, Scalar w0, Scalar r_star,
                                  int p) {
  Scalar q = r_star / r, f = Scalar(1);
  for (int i = 0; i < p; i++) f *= q;
  return w0 * f;
}

// v_LT = omega_lt(r) ẑ × x — the PROGRADE dragging velocity of the local
// inertial frame.  This is the physical drag, not the metric shift; see
// frame_drag_shift below for the sign relation, which matters.
HD_INLINE void frame_drag_velocity(Scalar x, Scalar y, Scalar z, Scalar w0,
                                   Scalar r_star, int p, Scalar& vx,
                                   Scalar& vy, Scalar& vz) {
  Scalar r = std::sqrt(x * x + y * y + z * z);
  Scalar w = frame_drag_omega(r, w0, r_star, p);
  vx = -w * y;
  vy = w * x;
  vz = Scalar(0);
}

// =========================================================================
// The 3+1 SHIFT VECTOR,  beta = -v_LT.
//
// THIS IS THE ONE PLACE THE FRAME-DRAG SIGN IS DECIDED.  Everything else
// (Faraday's effective circulation, the particle position update) is
// written in terms of beta, so the convention cannot drift apart between
// the field solver and the pusher.
//
// In the standard 3+1 slow-rotation form (Philippov, Cerutti, Tchekhovskoy
// & Spitkovsky 2015, arXiv:1510.01734, Eqs. 8-11) the metric drags
// PROGRADE while the shift enters with the opposite sign, beta^phi =
// -omega_LT.  The two consumers are then
//
//     Faraday :  curl(alpha E + beta x B) = -dB/dt   =>  E_eff = alpha E - v_LT x B
//     particle:  dx/dt = alpha v - beta              =>  dx/dt = alpha v + v_LT
//
// WHY THE SIGN IS NOT COSMETIC.  Stationarity of Faraday (Ferraro
// isorotation) forces the E_eff drift rate to be constant on flux
// surfaces.  With beta = -v_LT the plasma's FIDO rate is
// Omega_F - omega_lt(r) (the Muslimov-Tsygan reduction) and the shift adds
// omega_lt back, so the COORDINATE rotation is rigid at Omega_F and the
// state is stationary.  With the opposite sign the coordinate rate becomes
// Omega_F + omega_lt(r), which varies ALONG a field line -- differential
// rotation of a frozen-in plasma, i.e. NO stationary state exists at all.
// Measured before the fix: a saturated ~50x tangential-E shell at
// r = 1.02-1.2 in run ns_rotator_L6_a00_cool_fp32_gca10_mu0_gr.
// Regression: tests/test_dec_frame_drag.cpp, "the MT state is stationary
// under the solver".
// =========================================================================
HD_INLINE void frame_drag_shift(Scalar x, Scalar y, Scalar z, Scalar w0,
                                Scalar r_star, int p, Scalar& bx,
                                Scalar& by, Scalar& bz) {
  frame_drag_velocity(x, y, z, w0, r_star, p, bx, by, bz);
  bx = -bx;
  by = -by;
  bz = -bz;
}

// Lapse alpha(r) = sqrt(1 - r_s/r) with r_s = compactness * r_star
// (Schwarzschild; the slow-rotation correction to alpha is O(a^2) and is
// dropped alongside the gravitomagnetic tensor term, matching PCTS15's
// "the last two terms may be justifiably neglected").
//
// compactness <= 0 returns exactly 1, so the whole GR path collapses to a
// bitwise no-op in flat space -- the invariant asserted by
// "Lapse + shift: gr_compactness = 0 is an exact no-op".
//
// Floored at r_s: r_min sits far outside the horizon for any neutron-star
// compactness (C = 0.5 => alpha(R*) = 0.707), so the floor is a guard
// against a mis-set config, never a physical regime.
HD_INLINE Scalar gr_lapse(Scalar r, Scalar compactness, Scalar r_star) {
  if (compactness <= Scalar(0)) return Scalar(1);
  Scalar a2 = Scalar(1) - compactness * r_star / r;
  if (a2 < Scalar(1e-4)) a2 = Scalar(1e-4);
  return std::sqrt(a2);
}

// =========================================================================
// 3+1 metric terms as seen by the PARTICLE pusher.
//
// PCTS15 Eq. 7:  dp/dt = alpha q (E + v x B) + alpha m gamma g + alpha H.p
//                dx/dt = alpha v - beta   =   alpha v + v_LT
// We keep the leading term of each and drop the gravitational acceleration
// and gravitomagnetic tensor -- the paper's own stated approximation for
// strong pulsar fields.
//
// The shift is a PURE TRANSPORT term: it advects every particle identically
// regardless of momentum, never enters dp/dt, and so does not perturb the
// gyration, mu, or the adiabaticity that the GCA dispatch tests.  That is
// why it drops into the guiding-centre velocity without disturbing the
// rest of the push.  alpha and v_LT are functions of POSITION ONLY (static
// metric): no new particle state, no field interpolation, no checkpoint
// schema change.
// =========================================================================
struct gr_metric_params {
  bool enabled = false;
  Scalar omega_lt0 = 0;    // omega_LT at r_star
  Scalar r_star = 1;
  int lt_p = 3;            // omega_LT ~ (r_star/r)^lt_p
  Scalar compactness = 0;  // r_s / r_star; <= 0 => alpha == 1
};

// Is this parameter set the identity metric?  Zero-strength GR takes the
// SAME code path as disabled GR rather than producing alpha = 1, v_LT = 0
// and trusting the arithmetic to collapse.  It would not collapse in
// general: x + 0.0 != x when x is -0.0 (it returns +0.0), and -0.0 is
// precisely what `-w*y` yields at zero drag -- and a sign-of-zero reaching
// the triangle walk is a discrete flip, not a rounding difference.
// Callers branch on this, never on `enabled` alone.
HD_INLINE bool gr_is_identity(const gr_metric_params& g) {
  return !g.enabled ||
         (g.omega_lt0 == Scalar(0) && g.compactness <= Scalar(0));
}

// alpha and v_LT = -beta at a Cartesian point.  Identity params return
// exactly (1, 0).
HD_INLINE void gr_metric_at(const gr_metric_params& g, Scalar x, Scalar y,
                            Scalar z, Scalar& alpha, Scalar& vx, Scalar& vy,
                            Scalar& vz) {
  if (gr_is_identity(g)) {
    alpha = Scalar(1);
    vx = vy = vz = Scalar(0);
    return;
  }
  Scalar r = std::sqrt(x * x + y * y + z * z);
  alpha = gr_lapse(r, g.compactness, g.r_star);
  frame_drag_velocity(x, y, z, g.omega_lt0, g.r_star, g.lt_p, vx, vy, vz);
}

// =========================================================================
// Device-callable analytic field evaluators
// =========================================================================

HD_INLINE void dipole_B_impl(Scalar x, Scalar y, Scalar z,
                              Scalar mx, Scalar my, Scalar mz,
                              Scalar& Bx, Scalar& By, Scalar& Bz) {
  Scalar r2 = x*x + y*y + z*z;
  Scalar r = std::sqrt(r2);
  Scalar r5 = r2*r2*r;
  Scalar mdotr = mx*x + my*y + mz*z;
  Scalar factor = Scalar(3.0) * mdotr / r5;
  Scalar r3 = r2*r;
  Bx = factor*x - mx/r3;
  By = factor*y - my/r3;
  Bz = factor*z - mz/r3;
}

// Point magnetic quadrupole from the scalar potential
//     Phi = (1/2) x·Q·x / r^5,   B = -grad Phi
// with Q a SYMMETRIC TRACELESS tensor (Qzz = -Qxx-Qyy).  Consistent with
// dipole_B_impl (Phi_dip = m·x / r^3); far field ~ Q/r^4.  For a zonal
// (axisymmetric) quadrupole the polar surface field is
// B_z(0,0,R) = (3/2) Qzz / R^4.
HD_INLINE void quadrupole_B_impl(Scalar x, Scalar y, Scalar z,
                                 Scalar Qxx, Scalar Qxy, Scalar Qxz,
                                 Scalar Qyy, Scalar Qyz, Scalar Qzz,
                                 Scalar& Bx, Scalar& By, Scalar& Bz) {
  Scalar r2 = x*x + y*y + z*z;
  Scalar r = std::sqrt(r2);
  Scalar r5 = r2*r2*r;
  Scalar r7 = r5*r2;
  // Q·x and x·Q·x
  Scalar Qx = Qxx*x + Qxy*y + Qxz*z;
  Scalar Qy = Qxy*x + Qyy*y + Qyz*z;
  Scalar Qz = Qxz*x + Qyz*y + Qzz*z;
  Scalar xQx = x*Qx + y*Qy + z*Qz;
  Scalar factor = Scalar(2.5) * xQx / r7;
  Bx = factor*x - Qx/r5;
  By = factor*y - Qy/r5;
  Bz = factor*z - Qz/r5;
}

// =========================================================================
// Generalized stellar multipole: a (possibly shifted) point dipole plus a
// (possibly shifted) point quadrupole, all rigidly corotating.
//
// stellar_extras holds the BODY-FRAME extensions beyond the centered
// dipole; ALL-ZERO (the default) means "centered dipole" and every
// consumer then takes an arithmetic path BITWISE IDENTICAL to the legacy
// dipole-only code:
//   - zero offsets stay exactly +0.0 (never rotated, so no -0.0 can
//     appear) and x - (+0.0) is an exact identity for every x including
//     -0.0;
//   - the quadrupole term sits behind a has_quad branch, so no
//     "+ 0.0" ever touches the dipole field values.
//
// Frame conventions (matching the legacy obliquity handling): the body
// frame coincides with the lab frame at phase 0 and rotates about the
// spin axis z by phase = Omega*t.  The obliquity applies to the DIPOLE
// MOMENT ONLY, m_body = Bp (sin chi, 0, cos chi) — exactly the legacy
// moment.  The quadrupole tensor and both offsets are specified raw in
// the body frame (any orientation can be encoded directly in Q), and
// Bp >= 0 is assumed as everywhere else in the code.
// =========================================================================
struct stellar_extras {
  Scalar dip_off[3] = {0, 0, 0};   // dipole offset (body frame)
  Scalar quad_Q[5] = {0, 0, 0, 0, 0};  // Qxx, Qxy, Qxz, Qyy, Qyz (traceless)
  Scalar quad_off[3] = {0, 0, 0};  // quadrupole offset (body frame)
};

HD_INLINE bool stellar_has_dip_off(const stellar_extras& e) {
  return e.dip_off[0] != Scalar(0) || e.dip_off[1] != Scalar(0) ||
         e.dip_off[2] != Scalar(0);
}

HD_INLINE bool stellar_has_quad(const stellar_extras& e) {
  return e.quad_Q[0] != Scalar(0) || e.quad_Q[1] != Scalar(0) ||
         e.quad_Q[2] != Scalar(0) || e.quad_Q[3] != Scalar(0) ||
         e.quad_Q[4] != Scalar(0);
}

HD_INLINE bool stellar_extras_present(const stellar_extras& e) {
  return stellar_has_dip_off(e) || stellar_has_quad(e) ||
         e.quad_off[0] != Scalar(0) || e.quad_off[1] != Scalar(0) ||
         e.quad_off[2] != Scalar(0);
}

// Lab-frame snapshot of the rotated multipole at one instant — the value
// type the field kernels capture.  Defaults describe a zero centered
// dipole (offsets +0.0, has_quad false).
struct stellar_moments {
  Scalar mx = 0, my = 0, mz = 0;   // dipole moment
  Scalar dx = 0, dy = 0, dz = 0;   // dipole offset
  Scalar Qxx = 0, Qxy = 0, Qxz = 0, Qyy = 0, Qyz = 0, Qzz = 0;
  Scalar qx = 0, qy = 0, qz = 0;   // quadrupole offset
  bool has_quad = false;
};

// Rotate the body-frame configuration to the lab frame at the given spin
// phase (= Omega*t, passed in DOUBLE like the legacy call sites so the
// trig and the promotion order — (Scalar)·(double) then cast — reproduce
// the legacy moment bitwise).
HD_INLINE stellar_moments stellar_moments_at(Scalar Bp, Scalar obliquity,
                                             const stellar_extras& e,
                                             double phase) {
  stellar_moments m;
  double c = std::cos(phase);
  double s = std::sin(phase);
  // Legacy expressions, term for term.
  m.mx = static_cast<Scalar>(Bp * std::sin(obliquity) * c);
  m.my = static_cast<Scalar>(Bp * std::sin(obliquity) * s);
  m.mz = Bp * std::cos(obliquity);
  // Offsets are rotated ONLY when nonzero so the centered configuration
  // keeps exact +0.0 components (c or s < 0 would otherwise mint -0.0,
  // and x - (-0.0) flips the sign of a zero coordinate).
  if (stellar_has_dip_off(e)) {
    m.dx = static_cast<Scalar>(c * e.dip_off[0] - s * e.dip_off[1]);
    m.dy = static_cast<Scalar>(s * e.dip_off[0] + c * e.dip_off[1]);
    m.dz = e.dip_off[2];
  }
  if (stellar_has_quad(e)) {
    m.has_quad = true;
    double Qxx = e.quad_Q[0], Qxy = e.quad_Q[1], Qxz = e.quad_Q[2];
    double Qyy = e.quad_Q[3], Qyz = e.quad_Q[4];
    double c2 = c * c, s2 = s * s, cs = c * s;
    // Q_lab = Rz(phase) Q Rz(phase)^T; the trace (and Qzz) is invariant.
    m.Qxx = static_cast<Scalar>(c2 * Qxx - 2 * cs * Qxy + s2 * Qyy);
    m.Qxy = static_cast<Scalar>(cs * (Qxx - Qyy) + (c2 - s2) * Qxy);
    m.Qyy = static_cast<Scalar>(s2 * Qxx + 2 * cs * Qxy + c2 * Qyy);
    m.Qxz = static_cast<Scalar>(c * Qxz - s * Qyz);
    m.Qyz = static_cast<Scalar>(s * Qxz + c * Qyz);
    m.Qzz = static_cast<Scalar>(-(Qxx + Qyy));
    m.qx = static_cast<Scalar>(c * e.quad_off[0] - s * e.quad_off[1]);
    m.qy = static_cast<Scalar>(s * e.quad_off[0] + c * e.quad_off[1]);
    m.qz = e.quad_off[2];
  }
  return m;
}

// Total stellar B at a lab-frame point.  With default (centered-dipole)
// moments this is bitwise dipole_B_impl — see the header note above.
HD_INLINE void stellar_B_impl(Scalar x, Scalar y, Scalar z,
                              const stellar_moments& m,
                              Scalar& Bx, Scalar& By, Scalar& Bz) {
  dipole_B_impl(x - m.dx, y - m.dy, z - m.dz, m.mx, m.my, m.mz, Bx, By, Bz);
  if (m.has_quad) {
    Scalar qbx, qby, qbz;
    quadrupole_B_impl(x - m.qx, y - m.qy, z - m.qz, m.Qxx, m.Qxy, m.Qxz,
                      m.Qyy, m.Qyz, m.Qzz, qbx, qby, qbz);
    Bx += qbx;
    By += qby;
    Bz += qbz;
  }
}

// Full retarded Deutsch solution for a rotating magnetic dipole (c = 1).
// Derived from the Hertz potential Π = m(t_r)/r, with A = ∇×Π:
//
//   A = (m_r × n̂)/r² + (ṁ_r × n̂)/r
//   B = ∇×A = [3n̂(n̂·m_r) - m_r]/r³ + [3n̂(n̂·ṁ_r) - ṁ_r]/r² + [n̂(n̂·m̈_r) - m̈_r]/r
//   E = -∂A/∂t = (n̂ × ṁ_r)/r² + (n̂ × m̈_r)/r
//
// where n̂ = r̂, m_r = m(t - r), ṁ_r = ṁ(t - r), m̈_r = m̈(t - r).
HD_INLINE void deutsch_B_impl(Scalar x, Scalar y, Scalar z, Scalar time,
                               Scalar Bp, Scalar Omega, Scalar obliquity,
                               Scalar& Bx, Scalar& By, Scalar& Bz) {
  Scalar r2 = x*x + y*y + z*z;
  Scalar r = std::sqrt(r2);
  Scalar r3 = r2*r;
  Scalar t_ret = time - r;

  Scalar m_perp = Bp * std::sin(obliquity);
  Scalar m_par = Bp * std::cos(obliquity);

  Scalar cos_phase = std::cos(Omega * t_ret);
  Scalar sin_phase = std::sin(Omega * t_ret);

  // Retarded dipole moment, its first and second time-derivatives
  Scalar mx = m_perp * cos_phase;
  Scalar my = m_perp * sin_phase;
  Scalar mz = m_par;

  Scalar dmx = -m_perp * Omega * sin_phase;
  Scalar dmy =  m_perp * Omega * cos_phase;

  Scalar ddmx = -m_perp * Omega * Omega * cos_phase;
  Scalar ddmy = -m_perp * Omega * Omega * sin_phase;

  Scalar nx = x / r, ny = y / r, nz = z / r;

  // Near field: [3n(n·m) - m] / r³
  Scalar ndotm = nx*mx + ny*my + nz*mz;
  Scalar Bnx = (Scalar(3.0)*ndotm*nx - mx) / r3;
  Scalar Bny = (Scalar(3.0)*ndotm*ny - my) / r3;
  Scalar Bnz = (Scalar(3.0)*ndotm*nz - mz) / r3;

  // Intermediate field: [3n(n·dm) - dm] / r²
  Scalar ndotdm = nx*dmx + ny*dmy;
  Scalar Bix = (Scalar(3.0)*ndotdm*nx - dmx) / r2;
  Scalar Biy = (Scalar(3.0)*ndotdm*ny - dmy) / r2;
  Scalar Biz = (Scalar(3.0)*ndotdm*nz) / r2;

  // Radiation field: [n(n·ddm) - ddm] / r
  Scalar ndotddm = nx*ddmx + ny*ddmy;
  Scalar Brx = (ndotddm*nx - ddmx) / r;
  Scalar Bry = (ndotddm*ny - ddmy) / r;
  Scalar Brz = (ndotddm*nz) / r;

  Bx = Bnx + Bix + Brx;
  By = Bny + Biy + Bry;
  Bz = Bnz + Biz + Brz;
}

HD_INLINE void deutsch_E_impl(Scalar x, Scalar y, Scalar z, Scalar time,
                               Scalar Bp, Scalar Omega, Scalar obliquity,
                               Scalar& Ex, Scalar& Ey, Scalar& Ez) {
  Scalar r2 = x*x + y*y + z*z;
  Scalar r = std::sqrt(r2);
  Scalar t_ret = time - r;

  Scalar m_perp = Bp * std::sin(obliquity);

  Scalar cos_phase = std::cos(Omega * t_ret);
  Scalar sin_phase = std::sin(Omega * t_ret);

  Scalar dmx = -m_perp * Omega * sin_phase;
  Scalar dmy =  m_perp * Omega * cos_phase;

  Scalar ddmx = -m_perp * Omega * Omega * cos_phase;
  Scalar ddmy = -m_perp * Omega * Omega * sin_phase;

  Scalar nx = x / r, ny = y / r, nz = z / r;

  // E = +(n × dm)/r² + (n × ddm)/r
  // n × dm = (-nz*dmy, nz*dmx, nx*dmy - ny*dmx)
  Scalar cx1 = -nz * dmy;
  Scalar cy1 =  nz * dmx;
  Scalar cz1 =  nx * dmy - ny * dmx;

  Scalar cx2 = -nz * ddmy;
  Scalar cy2 =  nz * ddmx;
  Scalar cz2 =  nx * ddmy - ny * ddmx;

  Ex = cx1 / r2 + cx2 / r;
  Ey = cy1 / r2 + cy2 / r;
  Ez = cz1 / r2 + cz2 / r;
}

}  // namespace Aperture
