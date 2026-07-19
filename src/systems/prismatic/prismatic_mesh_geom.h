#pragma once

#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_mesh.h"

namespace Aperture {

// =========================================================================
// Phase 7D — per-element 3D geometry from the persisted sphere stage.
//
// Every (N_r+1)·N_s-sized quantity of the full mesh build decomposes as
// (double-precision angular factor) × (radial factor from the radii
// array).  These helpers reproduce the global arrays BIT-EXACTLY —
// including the float round-trips the global build takes (quantities
// that pass through a Scalar buffer before being reused are re-rounded
// here in the same places).  Pinned by test_prismatic_mesh_geom against
// a full build; do not "simplify" expressions.
//
// All functions are host-side build-time helpers (local mesh builders);
// they only need the sphere stage (mesh.build_sphere_only or full
// build).  Radial index arguments are GLOBAL shell/layer indices.
// =========================================================================
namespace prismatic_geom {

// ---- primal measures ----------------------------------------------------

inline Scalar tri_area(const prismatic_mesh& m, int k, int t) {
  double r = m.radii[k];
  return static_cast<Scalar>(r * r * m.sph_tri_omega[t]);
}

inline Scalar rect_area(const prismatic_mesh& m, int k, int e) {
  double r0 = m.radii[k], r1 = m.radii[k + 1];
  double dr = r1 - r0;
  return static_cast<Scalar>(dr * 0.5 * (r0 + r1) * m.sph_edge_alpha[e]);
}

inline Scalar h_edge_length(const prismatic_mesh& m, int k, int e) {
  double r = m.radii[k];
  return static_cast<Scalar>(r * m.sph_edge_alpha[e]);
}

inline Scalar v_edge_length(const prismatic_mesh& m, int k) {
  // Global: `double dr = radii[k + 1] - radii[k]` — the Scalar−Scalar
  // difference rounds BEFORE the double promotion.
  double dr = m.radii[k + 1] - m.radii[k];
  return static_cast<Scalar>(dr);
}

// ---- circumcentric dual Hodge -------------------------------------------

inline double r_mid(const prismatic_mesh& m, int k) {
  // Global: `0.5 * (radii[k] + radii[k + 1])` — the Scalar+Scalar sum
  // rounds to Scalar before the double multiply.  Keep verbatim.
  return 0.5 * (m.radii[k] + m.radii[k + 1]);
}

inline Scalar hodge2_tri(const prismatic_mesh& m, int k, int t) {
  double dist;
  if (k == 0) {
    dist = r_mid(m, 0) - m.radii[0];
  } else if (k == m.m_N_r) {
    dist = m.radii[m.m_N_r] - r_mid(m, m.m_N_r - 1);
  } else {
    dist = r_mid(m, k) - r_mid(m, k - 1);
  }
  Scalar fa = tri_area(m, k, t);  // float round-trip, as the global build
  return static_cast<Scalar>((fa > 0) ? dist / fa : 0);
}

inline Scalar hodge2_rect(const prismatic_mesh& m, int k, int e) {
  if (m.sph_edge_tri0[e] < 0 || m.sph_edge_tri1[e] < 0) return Scalar(0);
  double dist = r_mid(m, k) * m.sph_edge_beta[e];
  Scalar fa = rect_area(m, k, e);
  return static_cast<Scalar>((fa > 0) ? dist / fa : 0);
}

inline Scalar hodge1_inv_h(const prismatic_mesh& m, int k, int e) {
  double area = 0.0;
  if (m.sph_edge_tri0[e] >= 0 && m.sph_edge_tri1[e] >= 0) {
    double ang = m.sph_edge_beta[e];
    double r_below, r_above, r_avg, dr_span;
    if (k == 0) {
      r_below = m.radii[0];
      r_above = r_mid(m, 0);
      dr_span = r_above - r_below;
      r_avg = 0.5 * (r_below + r_above);
    } else if (k == m.m_N_r) {
      r_below = r_mid(m, m.m_N_r - 1);
      r_above = m.radii[m.m_N_r];
      dr_span = r_above - r_below;
      r_avg = 0.5 * (r_below + r_above);
    } else {
      r_below = r_mid(m, k - 1);
      r_above = r_mid(m, k);
      dr_span = r_above - r_below;
      r_avg = 0.5 * (r_below + r_above);
    }
    area = dr_span * r_avg * ang;
  }
  Scalar len = h_edge_length(m, k, e);  // float round-trip
  return static_cast<Scalar>((area > 0) ? len / area : 0);
}

inline Scalar hodge1_inv_v(const prismatic_mesh& m, int k, int s) {
  double rm = r_mid(m, k);
  double area = rm * rm * m.sph_vert_omega[s];
  Scalar len = v_edge_length(m, k);
  return static_cast<Scalar>((area > 0) ? len / area : 0);
}

// ---- lumped vertex dual volume ------------------------------------------
// Reproduces the global accumulation EXACTLY: for vertex (k, s), the
// layer-(k−1) v_top contributions land first (outer loop over layers),
// each layer's fan tris in ascending order, and every += rounds through
// Scalar (the global buffer element).
inline Scalar vert_dual_vol(const prismatic_mesh& m, int k, int s) {
  Scalar acc = 0;
  const int* fan = m.sph_vert_tris.data() + m.sph_vert_tri_offset[s];
  const int nf = m.sph_vert_tri_offset[s + 1] - m.sph_vert_tri_offset[s];

  if (k > 0) {
    double a = m.radii[k - 1], b = m.radii[k];
    double dr = b - a;
    for (int j = 0; j < nf; j++) {
      int t = fan[j];
      double omega = double(tri_area(m, k - 1, t)) / (a * a);
      double v_top = (omega / dr) *
          ((b * b * b * b - a * a * a * a) / 4.0 -
           a * (b * b * b - a * a * a) / 3.0);
      acc = static_cast<Scalar>(double(acc) + v_top / 3.0);
    }
  }
  if (k < m.m_N_r) {
    double a = m.radii[k], b = m.radii[k + 1];
    double dr = b - a;
    for (int j = 0; j < nf; j++) {
      int t = fan[j];
      double omega = double(tri_area(m, k, t)) / (a * a);
      double v_tot = omega * (b * b * b - a * a * a) / 3.0;
      double v_top = (omega / dr) *
          ((b * b * b * b - a * a * a * a) / 4.0 -
           a * (b * b * b - a * a * a) / 3.0);
      double v_bot = v_tot - v_top;
      acc = static_cast<Scalar>(double(acc) + v_bot / 3.0);
    }
  }
  return acc;
}

// ---- vertex coordinates --------------------------------------------------
// vert_theta/phi equal the persisted sphere_theta/phi bit-exactly (same
// expressions in the global build); vert_r is the shell radius.

inline Scalar vert_r(const prismatic_mesh& m, int k) { return m.radii[k]; }
inline Scalar vert_theta(const prismatic_mesh& m, int s) {
  return m.sphere_theta[s];
}
inline Scalar vert_phi(const prismatic_mesh& m, int s) {
  return m.sphere_phi[s];
}

// ---- boundary / radial-layer tags ---------------------------------------

inline int h_edge_boundary(const prismatic_mesh& m, int k) {
  return k == 0 ? 1 : (k == m.m_N_r ? 2 : 0);
}
inline int v_edge_boundary(const prismatic_mesh& m, int k) {
  return k == 0 ? 1 : (k == m.m_N_r - 1 ? 2 : 0);
}
inline int tri_face_boundary(const prismatic_mesh& m, int k) {
  return k == 0 ? 1 : (k == m.m_N_r ? 2 : 0);
}
inline int rect_face_boundary(const prismatic_mesh& m, int k) {
  return k == 0 ? 1 : (k == m.m_N_r - 1 ? 2 : 0);
}

// ---- GLOBAL vertex-id helpers (edge endpoints, matching the global
//      edge_v0/v1 tables' content) ---------------------------------------

inline gidx_t h_edge_gv0(const prismatic_mesh& m, int k, int e) {
  return gidx_t(k) * m.m_N_vert_s + m.sphere_edge_v0[e];
}
inline gidx_t h_edge_gv1(const prismatic_mesh& m, int k, int e) {
  return gidx_t(k) * m.m_N_vert_s + m.sphere_edge_v1[e];
}
inline gidx_t v_edge_gv0(const prismatic_mesh& m, int k, int s) {
  return gidx_t(k) * m.m_N_vert_s + s;
}
inline gidx_t v_edge_gv1(const prismatic_mesh& m, int k, int s) {
  return gidx_t(k + 1) * m.m_N_vert_s + s;
}

}  // namespace prismatic_geom

}  // namespace Aperture
