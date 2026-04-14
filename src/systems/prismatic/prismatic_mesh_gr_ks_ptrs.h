#pragma once

#include "systems/prismatic/prismatic_mesh_ptrs.h"

namespace Aperture {

// Extends prismatic_mesh_ptrs with per-element Kerr-Schild metric data.
// All metric quantities are precomputed at element midpoints/centroids and
// are time-independent (stationary spacetime).
//
// The Cartesian KS metric is  gamma_ij = delta_ij + f l_i l_j,  so all
// metric operations reduce to knowing f and l at each element.  Derived
// quantities (alpha, sqrt(gamma)*beta^i) are stored to avoid redundant
// sqrt() calls in inner kernels.
struct prismatic_mesh_gr_ks_ptrs : prismatic_mesh_ptrs {
  // --- Per-edge metric data (evaluated at edge midpoints) ---
  const Scalar* edge_f;          // f = 2r/rho^2           [N_edges]
  const Scalar* edge_lx;         // null covector l_x       [N_edges]
  const Scalar* edge_ly;         //                l_y       [N_edges]
  const Scalar* edge_lz;         //                l_z       [N_edges]
  const Scalar* edge_alpha;      // lapse = 1/sqrt(1+f)    [N_edges]
  const Scalar* edge_sgb_x;      // sqrt(gamma)*beta^x     [N_edges]
  const Scalar* edge_sgb_y;      // sqrt(gamma)*beta^y     [N_edges]
  const Scalar* edge_sgb_z;      // sqrt(gamma)*beta^z     [N_edges]
  const Scalar* edge_r;          // BL radius               [N_edges]

  // --- Per-face metric data (evaluated at face centroids) ---
  const Scalar* face_f;          // [N_faces]
  const Scalar* face_lx;         // [N_faces]
  const Scalar* face_ly;         // [N_faces]
  const Scalar* face_lz;         // [N_faces]
  const Scalar* face_alpha;      // [N_faces]
  const Scalar* face_sgb_x;      // [N_faces]
  const Scalar* face_sgb_y;      // [N_faces]
  const Scalar* face_sgb_z;      // [N_faces]
  const Scalar* face_r;          // [N_faces]

  // --- Convenience inline accessors for kernel use ---

  // Lower an upper-index vector at edge e:  v_i = gamma_ij v^j
  HD_INLINE void lower_at_edge(int e, Scalar vx, Scalar vy, Scalar vz,
                               Scalar& wx, Scalar& wy, Scalar& wz) const {
    Scalar lx = edge_lx[e], ly = edge_ly[e], lz = edge_lz[e];
    Scalar ldotv = lx * vx + ly * vy + lz * vz;
    Scalar fl = edge_f[e] * ldotv;
    wx = vx + fl * lx;
    wy = vy + fl * ly;
    wz = vz + fl * lz;
  }

  // Lower an upper-index vector at face f
  HD_INLINE void lower_at_face(int f, Scalar vx, Scalar vy, Scalar vz,
                               Scalar& wx, Scalar& wy, Scalar& wz) const {
    Scalar lx = face_lx[f], ly = face_ly[f], lz = face_lz[f];
    Scalar ldotv = lx * vx + ly * vy + lz * vz;
    Scalar fl = face_f[f] * ldotv;
    wx = vx + fl * lx;
    wy = vy + fl * ly;
    wz = vz + fl * lz;
  }

  // Raise a lower-index vector at edge e:  v^i = gamma^ij v_j
  HD_INLINE void raise_at_edge(int e, Scalar vx, Scalar vy, Scalar vz,
                               Scalar& wx, Scalar& wy, Scalar& wz) const {
    Scalar lx = edge_lx[e], ly = edge_ly[e], lz = edge_lz[e];
    Scalar fv = edge_f[e];
    Scalar ldotv = lx * vx + ly * vy + lz * vz;
    Scalar c = fv / (1.0f + fv) * ldotv;
    wx = vx - c * lx;
    wy = vy - c * ly;
    wz = vz - c * lz;
  }

  // Raise a lower-index vector at face f
  HD_INLINE void raise_at_face(int f, Scalar vx, Scalar vy, Scalar vz,
                               Scalar& wx, Scalar& wy, Scalar& wz) const {
    Scalar lx = face_lx[f], ly = face_ly[f], lz = face_lz[f];
    Scalar fv = face_f[f];
    Scalar ldotv = lx * vx + ly * vy + lz * vz;
    Scalar c = fv / (1.0f + fv) * ldotv;
    wx = vx - c * lx;
    wy = vy - c * ly;
    wz = vz - c * lz;
  }
};

}  // namespace Aperture
