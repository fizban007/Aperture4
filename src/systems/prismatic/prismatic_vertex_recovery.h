#pragma once

#include "core/buffer.hpp"
#include "core/exec_tags.h"
#include "core/math.hpp"
#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_ptrs.h"

namespace Aperture {

// =========================================================================
// Vertex least-squares recovery of B for the particle gather.
//
// The primal Whitney 2-form gather is discontinuous across prism faces
// (tangential-B jumps of O(h)), which pitch-angle-scatters particles and
// walks their guiding centers off their orbits.  This module fits, at
// every mesh vertex, a solenoidal linear field B(x) = B0 + G·(x - x_v)
// (trace(G) = 0) to the fluxes through the surrounding faces, in least
// squares, and hat-interpolates the fitted B0 vectors to particle
// positions.  Result: a fully C0, second-order B-gather.  Validated in
// python/prismatic_recovery.py; design + measurements in
// ROADMAP_NS_MAGNETOSPHERE.md (A1).
//
// The div-free constraint is essential, not cosmetic: the unconstrained
// fit's "trace" mode is exactly null at the 12 valence-5 vertices and
// weak everywhere else; with the constraint every patch has condition
// number ~O(30).
//
// Patch (canonical order, shared between the host weight build and the
// device flux gather — do not reorder):
//   interior shell k (1 <= k <= N_r-1):
//     tri-face fans at shells k-1, k, k+1   (3·val faces)
//     rect-face fans at layers k-1, k       (2·val faces)
//   inner boundary (k == 0):
//     tri fans at shells 0, 1;  rect fans at layers 0, 1
//   outer boundary (k == N_r):
//     tri fans at shells N_r-1, N_r;  rect fans at layers N_r-1, N_r-2
//
// Because the shells are exactly geometric (radii[k] = r_min·q^k), the
// interior patch at shell k is the shell-1 patch scaled by r_k/r_1, and
// the B0 weights scale as (r_1/r_k)^2.  So interior weights are stored
// once per SPHERE vertex (at reference shell 1) and rescaled at runtime;
// the two boundary classes get their own per-sphere-vertex sets.
// =========================================================================

// Trivially-copyable pointer bundle for GPU kernels.
struct prismatic_recovery_ptrs {
  int N_vert_s;
  int N_r;
  int N_verts;
  Scalar r_ref;                // radii[1]: reference radius of interior weights

  const int* valence;          // [N_vert_s] 5 or 6
  const int* tri_fan;          // [6 * N_vert_s] triangles around sphere vertex
  const int* edge_fan;         // [6 * N_vert_s] sphere edges around sphere vertex

  // Weight layout: w[(s*3 + c) * stride + j], c = Cartesian component,
  // j = patch slot in canonical order, zero-padded to stride.
  static constexpr int stride_int = 30;  // 3*6 tri + 2*6 rect
  static constexpr int stride_bnd = 24;  // 2*6 tri + 2*6 rect
  const Scalar* w_int;         // [N_vert_s * 3 * stride_int]
  const Scalar* w_inner;       // [N_vert_s * 3 * stride_bnd]
  const Scalar* w_outer;       // [N_vert_s * 3 * stride_bnd]

  Scalar* Bv;                  // [3 * N_verts], component-major

  // Fitted B at 3D vertex vi = k*N_vert_s + s from the face cochain B_f.
  HD_INLINE void compute_vertex_B(const prismatic_mesh_ptrs& mp,
                                  const Scalar* B_f, int vi) const {
    int k = vi / N_vert_s;
    int s = vi - k * N_vert_s;
    int val = valence[s];

    const Scalar* w;
    int stride;
    int tri_shells[3], rect_layers[2];
    int n_tri_shells;
    Scalar scale = Scalar(1.0);
    if (k == 0) {
      w = w_inner + s * 3 * stride_bnd;
      stride = stride_bnd;
      tri_shells[0] = 0; tri_shells[1] = 1;
      n_tri_shells = 2;
      rect_layers[0] = 0; rect_layers[1] = 1;
    } else if (k == N_r) {
      w = w_outer + s * 3 * stride_bnd;
      stride = stride_bnd;
      tri_shells[0] = N_r - 1; tri_shells[1] = N_r;
      n_tri_shells = 2;
      rect_layers[0] = N_r - 1; rect_layers[1] = N_r - 2;
    } else {
      w = w_int + s * 3 * stride_int;
      stride = stride_int;
      tri_shells[0] = k - 1; tri_shells[1] = k; tri_shells[2] = k + 1;
      n_tri_shells = 3;
      rect_layers[0] = k - 1; rect_layers[1] = k;
      // interior weights are stored at reference shell 1; fluxes scale
      // as r^2, so the weights scale as (r_ref / r_k)^2
      Scalar rk = mp.radii[k];
      scale = (r_ref / rk) * (r_ref / rk);
    }

    Scalar b0 = 0, b1 = 0, b2 = 0;
    int slot = 0;
    for (int ks = 0; ks < n_tri_shells; ks++) {
      for (int j = 0; j < val; j++, slot++) {
        Scalar f = B_f[mp.tri_face_idx(tri_shells[ks], tri_fan[s * 6 + j])];
        b0 += w[0 * stride + slot] * f;
        b1 += w[1 * stride + slot] * f;
        b2 += w[2 * stride + slot] * f;
      }
    }
    for (int kl = 0; kl < 2; kl++) {
      for (int j = 0; j < val; j++, slot++) {
        Scalar f = B_f[mp.rect_face_idx(rect_layers[kl], edge_fan[s * 6 + j])];
        b0 += w[0 * stride + slot] * f;
        b1 += w[1 * stride + slot] * f;
        b2 += w[2 * stride + slot] * f;
      }
    }
    Bv[0 * N_verts + vi] = scale * b0;
    Bv[1 * N_verts + vi] = scale * b1;
    Bv[2 * N_verts + vi] = scale * b2;
  }
};

// =========================================================================
// Phase 7C — fitted B at a LOCAL vertex tensor slot (k, s), from the
// LOCAL face cochain, using the ptc-mesh-local recovery tables
// (rec_valence / rec_tri_fan / rec_edge_fan / rec_w_*; local sphere
// ids).  Writes Bv[c * mp.N_verts + mp.vertex_idx(k, s)].  Boundary
// classes are decided on the GLOBAL shell index mp.k0 + k; the interior
// radial rescaling uses the local radii window (same values as global).
// Only OWNED slots are computed — ghost Bv arrives via the vertex-halo
// exchange (3 scalar components).  The patch/slot order is identical to
// prismatic_recovery_ptrs::compute_vertex_B (do not reorder).
// =========================================================================
template <typename MP>
HD_INLINE void compute_vertex_B_local(const MP& mp, const Scalar* B_f,
                                      Scalar* Bv, int k, int s) {
  constexpr int stride_int = 30;
  constexpr int stride_bnd = 24;
  const int val = mp.rec_valence[s];
  const int kg = mp.k0 + k;

  const Scalar* w;
  int stride;
  int tri_shells[3], rect_layers[2];
  int n_tri_shells;
  Scalar scale = Scalar(1.0);
  if (kg == 0) {
    // k0 == 0 here, so local and global shell indices coincide.
    w = mp.rec_w_inner + s * 3 * stride_bnd;
    stride = stride_bnd;
    tri_shells[0] = 0;
    tri_shells[1] = 1;
    n_tri_shells = 2;
    rect_layers[0] = 0;
    rect_layers[1] = 1;
  } else if (kg == mp.N_r_global) {
    w = mp.rec_w_outer + s * 3 * stride_bnd;
    stride = stride_bnd;
    tri_shells[0] = k - 1;
    tri_shells[1] = k;
    n_tri_shells = 2;
    rect_layers[0] = k - 1;
    rect_layers[1] = k - 2;
  } else {
    w = mp.rec_w_int + s * 3 * stride_int;
    stride = stride_int;
    tri_shells[0] = k - 1;
    tri_shells[1] = k;
    tri_shells[2] = k + 1;
    n_tri_shells = 3;
    rect_layers[0] = k - 1;
    rect_layers[1] = k;
    Scalar rk = mp.radii[k];
    scale = (mp.rec_r_ref / rk) * (mp.rec_r_ref / rk);
  }

  Scalar b0 = 0, b1 = 0, b2 = 0;
  int slot = 0;
  for (int ks = 0; ks < n_tri_shells; ks++) {
    for (int j = 0; j < val; j++, slot++) {
      Scalar f =
          B_f[mp.tri_face_idx(tri_shells[ks], mp.rec_tri_fan[s * 6 + j])];
      b0 += w[0 * stride + slot] * f;
      b1 += w[1 * stride + slot] * f;
      b2 += w[2 * stride + slot] * f;
    }
  }
  for (int kl = 0; kl < 2; kl++) {
    for (int j = 0; j < val; j++, slot++) {
      Scalar f =
          B_f[mp.rect_face_idx(rect_layers[kl], mp.rec_edge_fan[s * 6 + j])];
      b0 += w[0 * stride + slot] * f;
      b1 += w[1 * stride + slot] * f;
      b2 += w[2 * stride + slot] * f;
    }
  }
  const int vi = mp.vertex_idx(k, s);
  Bv[0 * mp.N_verts + vi] = scale * b0;
  Bv[1 * mp.N_verts + vi] = scale * b1;
  Bv[2 * mp.N_verts + vi] = scale * b2;
}

// Hat-function (barycentric x linear-in-zeta) interpolation of the
// per-vertex B vectors: the C0 second-order particle B-gather.
template <typename MP>
HD_INLINE void interpolate_B_recovery(const MP& mp,
                                      const Scalar* Bv, int tri_idx,
                                      int layer_idx, const Scalar l[3],
                                      Scalar zeta, Scalar& Bx, Scalar& By,
                                      Scalar& Bz) {
  int Nv = mp.N_verts;
  Bx = By = Bz = Scalar(0.0);
  for (int i = 0; i < 3; i++) {
    int s = mp.tri_verts[tri_idx * 3 + i];
    int vb = mp.vertex_idx(layer_idx, s);
    int vt = mp.vertex_idx(layer_idx + 1, s);
    Scalar wb = l[i] * (Scalar(1.0) - zeta);
    Scalar wt = l[i] * zeta;
    Bx += wb * Bv[0 * Nv + vb] + wt * Bv[0 * Nv + vt];
    By += wb * Bv[1 * Nv + vb] + wt * Bv[1 * Nv + vt];
    Bz += wb * Bv[2 * Nv + vb] + wt * Bv[2 * Nv + vt];
  }
}

// Hat-interpolated B AND its gradient G[i][j] = dB_i/dx_j from the
// per-vertex recovery field: within a prism B(x) = sum_i Bv_i N_i with
// N_i = lambda_i(x) phi(zeta), so G is piecewise constant per cell —
// first-order accurate, exactly what the GCA curvature / grad-B drift
// terms need.  The barycentric gradients are computed on the flattened
// triangle at the layer's radial midpoint (same convention as
// interpolate_fields).
template <typename MP>
HD_INLINE void interpolate_B_recovery_grad(
    const MP& mp, const Scalar* Bv, int tri_idx,
    int layer_idx, const Scalar l[3], Scalar zeta,
    Scalar B[3], Scalar G[3][3]) {
  int sv[3];
  for (int i = 0; i < 3; i++) sv[i] = mp.tri_verts[tri_idx * 3 + i];

  Scalar r_mid = Scalar(0.5) * (mp.radii[layer_idx] +
                                mp.radii[layer_idx + 1]);
  Scalar px[3], py[3], pz[3];
  for (int i = 0; i < 3; i++) {
    px[i] = r_mid * mp.sphere_vx[sv[i]];
    py[i] = r_mid * mp.sphere_vy[sv[i]];
    pz[i] = r_mid * mp.sphere_vz[sv[i]];
  }
  Scalar e1x = px[1]-px[0], e1y = py[1]-py[0], e1z = pz[1]-pz[0];
  Scalar e2x = px[2]-px[0], e2y = py[2]-py[0], e2z = pz[2]-pz[0];
  Scalar nx = e1y*e2z - e1z*e2y;
  Scalar ny = e1z*e2x - e1x*e2z;
  Scalar nz = e1x*e2y - e1y*e2x;
  Scalar two_A_sq = nx*nx + ny*ny + nz*nz;
  Scalar gl[3][3];
  for (int i = 0; i < 3; i++) {
    int j = (i + 1) % 3, k2 = (i + 2) % 3;
    Scalar dx = px[k2]-px[j], dy = py[k2]-py[j], dz = pz[k2]-pz[j];
    gl[i][0] = (ny*dz - nz*dy) / two_A_sq;
    gl[i][1] = (nz*dx - nx*dz) / two_A_sq;
    gl[i][2] = (nx*dy - ny*dx) / two_A_sq;
  }
  // Radial direction and d(zeta)/dx at the interpolation point
  Scalar rx = 0, ry = 0, rz = 0;
  for (int i = 0; i < 3; i++) {
    rx += l[i] * mp.sphere_vx[sv[i]];
    ry += l[i] * mp.sphere_vy[sv[i]];
    rz += l[i] * mp.sphere_vz[sv[i]];
  }
  Scalar rn = Scalar(1) / math::sqrt(rx*rx + ry*ry + rz*rz);
  Scalar dr = mp.radii[layer_idx + 1] - mp.radii[layer_idx];
  Scalar dz_[3] = {rx*rn/dr, ry*rn/dr, rz*rn/dr};

  int Nv = mp.N_verts;
  Scalar phi_b = Scalar(1) - zeta, phi_t = zeta;
  for (int c = 0; c < 3; c++) { B[c] = 0; G[c][0]=G[c][1]=G[c][2]=0; }
  for (int i = 0; i < 3; i++) {
    int vb = mp.vertex_idx(layer_idx, sv[i]);
    int vt = mp.vertex_idx(layer_idx + 1, sv[i]);
    for (int c = 0; c < 3; c++) {
      Scalar Bb = Bv[c * Nv + vb], Bt = Bv[c * Nv + vt];
      B[c] += l[i] * (phi_b * Bb + phi_t * Bt);
      for (int j = 0; j < 3; j++) {
        // grad N_bot = phi_b grad(lam) - lam grad(zeta);  top: + lam gz
        G[c][j] += Bb * (phi_b * gl[i][j] - l[i] * dz_[j]) +
                   Bt * (phi_t * gl[i][j] + l[i] * dz_[j]);
      }
    }
  }
}

// Host-side owner: builds fan tables and LSQ weights (double precision)
// at init, owns the per-vertex B buffer refreshed each step.
class prismatic_vertex_recovery {
 public:
  // Builds fan tables + weights from the (already built) mesh.  Host only.
  void build(const prismatic_mesh& mesh);

  void copy_to_device();

  prismatic_recovery_ptrs host_ptrs();
  prismatic_recovery_ptrs dev_ptrs();
  prismatic_recovery_ptrs get_ptrs(exec_tags::host) { return host_ptrs(); }
  prismatic_recovery_ptrs get_ptrs(exec_tags::device) { return dev_ptrs(); }

  // Max condition number over all patches (diagnostic, set by build()).
  double max_condition() const { return m_max_cond; }

  buffer<int> valence;      // [N_vert_s]
  buffer<int> tri_fan;      // [6 * N_vert_s]
  buffer<int> edge_fan;     // [6 * N_vert_s]
  buffer<Scalar> w_int;     // [N_vert_s * 3 * 30]
  buffer<Scalar> w_inner;   // [N_vert_s * 3 * 24]
  buffer<Scalar> w_outer;   // [N_vert_s * 3 * 24]
  buffer<Scalar> Bv;        // [3 * N_verts]

 private:
  int m_N_vert_s = 0;
  int m_N_r = 0;
  int m_N_verts = 0;
  Scalar m_r_ref = 1.0;
  double m_max_cond = 0.0;
};

}  // namespace Aperture
