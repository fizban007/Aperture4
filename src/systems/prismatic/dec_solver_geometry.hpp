#pragma once

#include "core/typedefs_and_constants.h"
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
HD_INLINE void vertex_unit(const prismatic_mesh_ptrs& mp, int vi,
                           Scalar& r, Scalar& ux, Scalar& uy, Scalar& uz) {
  int k = vi / mp.N_vert_s;
  int s = vi % mp.N_vert_s;
  r = mp.radii[k];
  ux = mp.sphere_vx[s];
  uy = mp.sphere_vy[s];
  uz = mp.sphere_vz[s];
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
