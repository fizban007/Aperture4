#pragma once

#include "core/buffer.hpp"
#include "core/exec_tags.h"
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

// Hat-function (barycentric x linear-in-zeta) interpolation of the
// per-vertex B vectors: the C0 second-order particle B-gather.
HD_INLINE void interpolate_B_recovery(const prismatic_mesh_ptrs& mp,
                                      const Scalar* Bv, int tri_idx,
                                      int layer_idx, const Scalar l[3],
                                      Scalar zeta, Scalar& Bx, Scalar& By,
                                      Scalar& Bz) {
  int Nv = mp.N_verts;
  Bx = By = Bz = Scalar(0.0);
  for (int i = 0; i < 3; i++) {
    int s = mp.tri_verts[tri_idx * 3 + i];
    int vb = layer_idx * mp.N_vert_s + s;
    int vt = vb + mp.N_vert_s;
    Scalar wb = l[i] * (Scalar(1.0) - zeta);
    Scalar wt = l[i] * zeta;
    Bx += wb * Bv[0 * Nv + vb] + wt * Bv[0 * Nv + vt];
    By += wb * Bv[1 * Nv + vb] + wt * Bv[1 * Nv + vt];
    Bz += wb * Bv[2 * Nv + vb] + wt * Bv[2 * Nv + vt];
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
