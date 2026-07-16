#pragma once

#include "core/buffer.hpp"
#include "core/exec_tags.h"
#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_ptrs.h"

namespace Aperture {

// =========================================================================
// Reconstruction-corrected Hodge operators for the flat DEC solver.
//
// The diagonal circumcentric Hodge is a one-point quadrature at an
// off-center primal/dual crossing, which makes the Ampère constitutive
// chain first-order for quasi-static fields (see roadmap A2 notes).
// This module replaces the two constitutive maps by local linear
// reconstructions (validated offline in python/hodge_lab_chain.py:
// full-chain probe ratio 6.5 vs the diagonal's 2.0):
//
//   W2 : per interior face, the dual-segment circulation
//        int_{f*} H . dl  from a div-free linear LSQ fit of ~30 nearby
//        face fluxes, integrated exactly along the dual segment.
//   W1 : per interior edge, the primal-edge circulation of the curl
//        field  int_e V . dl  from a div-free linear LSQ fit of ~32
//        nearby DUAL-face fluxes (the d1t loop sums), integrated
//        exactly along the primal edge.
//
// The d1t loop-sum structure is kept between the two maps, so the
// boundary-of-boundary identity — and with it Gauss-law/charge
// conservation with deposited currents — remains topologically exact.
//
// Storage: the mesh is self-similar across (geometric) shells, and both
// row types scale as 1/r, so weights are stored once per SPHERE element
// at a reference shell and rescaled by r_ref/r_k at runtime (verified
// against a directly-built row at another shell during build()).
// Faces/edges whose fit stencil would touch the radial boundary shells
// fall back to the diagonal Hodge (W2: shells {0, N_r}; W1: shells/
// layers {0, 1, N_r-1, N_r}).
// =========================================================================

struct recon_hodge_ptrs {
  int N_r;
  int N_tri;
  int N_vert_s;
  int N_edge_s;
  Scalar r_ref;                 // radii[k_ref] of the stored templates
  const Scalar* radii;          // [N_r + 1]

  static constexpr int W2_STRIDE = 30;  // 3 tri-fan shells + 2 rect-fan layers
  static constexpr int W1_STRIDE = 32;  // 3 h-dual shells + 2 v-dual layers

  // fan tables (fixed connectivity)
  const int* valence;           // [N_vert_s]
  const int* tri_fan;           // [6 * N_vert_s]
  const int* edge_fan;          // [6 * N_vert_s]
  const int* nbr_fan;           // [6 * N_vert_s] other endpoint of edge_fan

  // designated sphere vertex per element (fit anchor)
  const int* tri_anchor;        // [N_tri]   = tri_verts[t*3]
  const int* edge_anchor;       // [N_edge_s]= sphere_edges[e][0]

  // template rows (canonical column order, zero-padded)
  const Scalar* w2_tri;         // [N_tri    * W2_STRIDE]
  const Scalar* w2_rect;        // [N_edge_s * W2_STRIDE]
  const Scalar* w1_h;           // [N_edge_s * W1_STRIDE]
  const Scalar* w1_v;           // [N_vert_s * W1_STRIDE]

  // ---- canonical column enumeration (shared by build and kernels) ----
  // W2 columns: faces of the anchor-vertex patch at shell kv:
  //   tri fans at shells kv-1, kv, kv+1, then rect fans at layers kv-1, kv.
  HD_INLINE int w2_cols(const prismatic_mesh_ptrs& mp, int s, int kv,
                        int cols[W2_STRIDE]) const {
    int val = valence[s];
    int m = 0;
    for (int o = -1; o <= 1; o++)
      for (int j = 0; j < val; j++)
        cols[m++] = mp.tri_face_idx(kv + o, tri_fan[s * 6 + j]);
    for (int o = -1; o <= 0; o++)
      for (int j = 0; j < val; j++)
        cols[m++] = mp.rect_face_idx(kv + o, edge_fan[s * 6 + j]);
    return m;
  }

  // W1 columns: dual-face fluxes (edge-indexed) of the anchor patch:
  //   h-edges of the edge fan at shells kv-1, kv, kv+1, then v-edges of
  //   {anchor, fan neighbors} at layers kv-1, kv.
  HD_INLINE int w1_cols(const prismatic_mesh_ptrs& mp, int s, int kv,
                        int cols[W1_STRIDE]) const {
    int val = valence[s];
    int m = 0;
    for (int o = -1; o <= 1; o++)
      for (int j = 0; j < val; j++)
        cols[m++] = mp.h_edge_idx(kv + o, edge_fan[s * 6 + j]);
    for (int o = -1; o <= 0; o++) {
      cols[m++] = mp.v_edge_idx(kv + o, s);
      for (int j = 0; j < val; j++)
        cols[m++] = mp.v_edge_idx(kv + o, nbr_fan[s * 6 + j]);
    }
    return m;
  }

  // ---- corrected dual-segment circulation for face f (interior) ----
  // Falls back to diagonal for boundary shells.
  HD_INLINE Scalar circ_face(const prismatic_mesh_ptrs& mp, int f,
                             const Scalar* B) const {
    int n_tf = (mp.N_r + 1) * mp.N_tri;
    int cols[W2_STRIDE];
    if (f < n_tf) {
      int k = f / mp.N_tri;
      if (k < 1 || k > mp.N_r - 1) return mp.hodge2[f] * B[f];
      int t = f - k * mp.N_tri;
      int s = tri_anchor[t];
      int m = w2_cols(mp, s, k, cols);
      const Scalar* w = w2_tri + t * W2_STRIDE;
      Scalar acc = 0;
      for (int j = 0; j < m; j++) acc += w[j] * B[cols[j]];
      return acc * (r_ref / mp.radii[k]);
    }
    int fi = f - n_tf;
    int k = fi / mp.N_edge_s;
    if (k < 1 || k > mp.N_r - 2) return mp.hodge2[f] * B[f];
    int e = fi - k * mp.N_edge_s;
    int s = edge_anchor[e];
    int m = w2_cols(mp, s, k, cols);
    const Scalar* w = w2_rect + e * W2_STRIDE;
    Scalar acc = 0;
    for (int j = 0; j < m; j++) acc += w[j] * B[cols[j]];
    // rect-face dual arcs live at r_mid[k]; scaling uses the same shell
    // ratio (r_mid tracks radii on a geometric grid)
    return acc * (r_ref / mp.radii[k]);
  }

  // ---- corrected pairing for edge e (interior): S = Phi - J tilde ----
  HD_INLINE Scalar pair_edge(const prismatic_mesh_ptrs& mp, int ei,
                             const Scalar* S) const {
    int n_h = (mp.N_r + 1) * mp.N_edge_s;
    int cols[W1_STRIDE];
    if (ei < n_h) {
      int k = ei / mp.N_edge_s;
      if (k < 2 || k > mp.N_r - 2) return mp.hodge1_inv[ei] * S[ei];
      int e = ei - k * mp.N_edge_s;
      int s = edge_anchor[e];
      int m = w1_cols(mp, s, k, cols);
      const Scalar* w = w1_h + e * W1_STRIDE;
      Scalar acc = 0;
      for (int j = 0; j < m; j++) acc += w[j] * S[cols[j]];
      return acc * (r_ref / mp.radii[k]);
    }
    int li = ei - n_h;
    int k = li / mp.N_vert_s;
    if (k < 2 || k > mp.N_r - 2) return mp.hodge1_inv[ei] * S[ei];
    int s = li - k * mp.N_vert_s;
    int m = w1_cols(mp, s, k, cols);
    const Scalar* w = w1_v + s * W1_STRIDE;
    Scalar acc = 0;
    for (int j = 0; j < m; j++) acc += w[j] * S[cols[j]];
    return acc * (r_ref / mp.radii[k]);
  }
};

class prismatic_recon_hodge {
 public:
  // Host build (double precision fits at reference shell k_ref = 2).
  // Requires a flat mesh with geometric shells and N_r >= 6.
  void build(const prismatic_mesh& mesh);

  void copy_to_device();

  recon_hodge_ptrs host_ptrs() const;
  recon_hodge_ptrs dev_ptrs() const;
  recon_hodge_ptrs get_ptrs(exec_tags::host) const { return host_ptrs(); }
  recon_hodge_ptrs get_ptrs(exec_tags::device) const { return dev_ptrs(); }

  buffer<int> valence, tri_fan, edge_fan, nbr_fan;
  buffer<int> tri_anchor, edge_anchor;
  buffer<Scalar> w2_tri, w2_rect, w1_h, w1_v;

 private:
  int m_N_r = 0, m_N_tri = 0, m_N_vert_s = 0, m_N_edge_s = 0;
  Scalar m_r_ref = 1.0;
  const prismatic_mesh* m_mesh = nullptr;
};

}  // namespace Aperture
