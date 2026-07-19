#pragma once

#include "core/buffer.hpp"
#include "core/exec_tags.h"
#include "core/gpu_translation_layer.h"
#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include <cmath>
#include <vector>

namespace Aperture {

// =========================================================================
// Phase 7C — LOCAL particle mesh (plan F5).
//
// The particle kernels keep their arithmetic shape (tensor index =
// layer·n_s + s, barycentric walks over tri_neighbor, Whitney stencils
// through h_edge_idx / v_edge_idx / ...) but operate on a LOCAL index
// space:
//
//   - Local sphere numbering: owned tris first (ascending global order),
//     then the T_halo ring (tris sharing ≥ 1 sphere-vertex with an
//     owned tri), ascending global.  Sphere-edges and sphere-vertices
//     of those tris are numbered the same way (owned block, then ghost
//     block, each ascending global).  tri_neighbor entries beyond
//     T_halo become −1.
//
//   - Local radial extent: the owned slab range padded by one ghost
//     prism on each interior side.  k0 = global index of local layer 0;
//     the local radii window is a POINTER OFFSET into the global radii
//     array (layers are contiguous), so radial arithmetic is unchanged.
//
//   - Field access goes through one int map per cochain,
//     tensor_to_layout[k·n_s + s] = index into the solver-layout local
//     field buffers ("owned ascending global, then ghosts" — untouched
//     since 4.1b).  The maps are TOTAL: with pic-depth layouts every
//     local tensor slot exists (asserted at build).  The [h|v] and
//     [tri|rect] combined-block offsets are baked into the indexing
//     helpers exactly like the global prismatic_mesh_ptrs formulas.
//
// The POD ptrs struct mirrors prismatic_mesh_ptrs' member and method
// names, so every particle kernel is simply templated on the mesh-ptrs
// type; under the identity (single-rank) partition all tables and maps
// are identity and results are bit-exact with the global path.
// =========================================================================
struct prismatic_ptc_mesh_ptrs {
  // --- Local sizes ---
  int N_r;        // local prism layers
  int N_tri;      // local tris (owned + ring)
  int N_vert_s;   // local sphere vertices
  int N_edge_s;   // local sphere edges
  int N_verts;    // vertex-cochain layout size (rho / Bv stride)
  int N_edges;    // combined [h|v] edge layout size
  int N_faces;    // combined [tri|rect] face layout size

  // --- Ownership + global anchoring ---
  int n_tri_own;       // owned tris are local ids [0, n_tri_own)
  int n_vert_s_own;    // owned sphere vertices [0, n_vert_s_own)
  int n_edge_s_own;
  int k0;              // global layer/shell index of local layer 0
  int lay_own_lo, lay_own_hi;      // owned local LAYER (slab) range
  int shell_own_lo, shell_own_hi;  // owned local SHELL range
  int N_r_global;      // global layer count (recovery boundary classes)

  // --- Radii: pointer into the global array at offset k0 ---
  const Scalar* radii;              // [N_r + 1] local window

  // --- Local sphere tables (all indices LOCAL) ---
  const Scalar* sphere_vx;          // [N_vert_s]
  const Scalar* sphere_vy;
  const Scalar* sphere_vz;
  const Scalar* sphere_theta;
  const Scalar* sphere_phi;
  const int* tri_verts;             // [N_tri * 3] local vertex ids
  const int* tri_edges_s;           // [N_tri * 3] local edge ids
  const int* tri_edge_signs;        // [N_tri * 3]
  const int* tri_neighbor;          // [N_tri * 3] local tri ids, −1 = off-halo

  // --- Tensor → layout maps (one int per local element) ---
  const int* map_h;     // [(N_r+1) * N_edge_s]
  const int* map_v;     // [N_r * N_vert_s]
  const int* map_tri;   // [(N_r+1) * N_tri]
  const int* map_rect;  // [N_r * N_edge_s]
  const int* map_vert;  // [(N_r+1) * N_vert_s]
  int e_split;          // v-block offset in the combined edge buffer
  int b_split;          // rect-block offset in the combined face buffer

  // --- Layout-indexed geometry the particle path reads ---
  const Scalar* vert_dual_vol;      // [N_verts]
  const Scalar* hodge1_inv;         // [N_edges] combined [h|v]

  // --- Vertex-recovery tables (local sphere ids; see
  //     prismatic_vertex_recovery.h for the patch convention).  Fan
  //     entries can be −1 only on NON-owned vertex rows (whose Bv comes
  //     from the halo exchange, never from a local fit). ---
  const int* rec_valence;           // [N_vert_s]
  const int* rec_tri_fan;           // [6 * N_vert_s] local tri ids
  const int* rec_edge_fan;          // [6 * N_vert_s] local edge ids
  const Scalar* rec_w_int;          // [N_vert_s * 3 * 30]
  const Scalar* rec_w_inner;        // [N_vert_s * 3 * 24]
  const Scalar* rec_w_outer;        // [N_vert_s * 3 * 24]
  Scalar rec_r_ref;

  // --- Migration ---
  const int* tri_l2g;               // [N_tri] global tri of local tri
  const int* tri_ang_rank;          // [N_tri] canonical angular rank
  int N_tri_global;                 // for the global wire cell encoding
  int mig_A;                        // angular rank count
  int mig_slab_base, mig_slab_rem;  // radial slab map (base/rem)

  // =======================================================================
  // Indexing helpers — the drop-in equivalents of the global formulas,
  // resolved through the per-cochain maps.
  // =======================================================================
  HD_INLINE int h_edge_idx(int k, int e) const {
    return map_h[k * N_edge_s + e];
  }
  HD_INLINE int v_edge_idx(int k, int s) const {
    return e_split + map_v[k * N_vert_s + s];
  }
  HD_INLINE int tri_face_idx(int k, int t) const {
    return map_tri[k * N_tri + t];
  }
  HD_INLINE int rect_face_idx(int k, int e) const {
    return b_split + map_rect[k * N_edge_s + e];
  }
  HD_INLINE int vertex_idx(int k, int s) const {
    return map_vert[k * N_vert_s + s];
  }

  HD_INLINE bool owns_cell(int tri, int layer) const {
    return tri < n_tri_own && layer >= lay_own_lo && layer < lay_own_hi;
  }

  // Destination WORLD rank (= rad·A + ang, plan F7) of a LOCAL cell, or
  // −1 when this rank owns it.  Angular rank via the per-tri table,
  // radial rank via the uniform slab map on the global layer.
  HD_INLINE int migrate_dest(uint32_t cell) const {
    const int lay = int(cell) / N_tri;
    const int tri = int(cell) - lay * N_tri;
    if (owns_cell(tri, lay)) return -1;
    const int ang = tri_ang_rank[tri];
    const int glay = k0 + lay;
    const int split = mig_slab_rem * (mig_slab_base + 1);
    const int rad = glay < split
                        ? glay / (mig_slab_base + 1)
                        : mig_slab_rem + (glay - split) / mig_slab_base;
    return rad * mig_A + ang;
  }

  // GLOBAL (rank-agnostic) wire encoding of a LOCAL cell, plan F7.
  // 64-bit: the global cell space k·N_tri_global + tri exceeds 2^32
  // near L9 (checkpoint plan appendix item 1) — the wire and the
  // checkpoint format carry uint64.  LOCAL cells stay uint32 (bounded
  // by the per-rank mesh size; guarded loudly at build).
  HD_INLINE uint64_t wire_cell(uint32_t cell) const {
    const int lay = int(cell) / N_tri;
    const int tri = int(cell) - lay * N_tri;
    return uint64_t(k0 + lay) * uint64_t(N_tri_global) +
           uint64_t(tri_l2g[tri]);
  }

  // =======================================================================
  // Geometry/topology methods — bodies identical to prismatic_mesh_ptrs
  // (they read only the local tables above).
  // =======================================================================
  HD_INLINE void prism_edge_indices(int tri_idx, int layer_idx,
                                    int edges[9]) const {
    edges[0] = h_edge_idx(layer_idx, tri_edges_s[tri_idx * 3 + 0]);
    edges[1] = h_edge_idx(layer_idx, tri_edges_s[tri_idx * 3 + 1]);
    edges[2] = h_edge_idx(layer_idx, tri_edges_s[tri_idx * 3 + 2]);
    edges[3] = h_edge_idx(layer_idx + 1, tri_edges_s[tri_idx * 3 + 0]);
    edges[4] = h_edge_idx(layer_idx + 1, tri_edges_s[tri_idx * 3 + 1]);
    edges[5] = h_edge_idx(layer_idx + 1, tri_edges_s[tri_idx * 3 + 2]);
    edges[6] = v_edge_idx(layer_idx, tri_verts[tri_idx * 3 + 0]);
    edges[7] = v_edge_idx(layer_idx, tri_verts[tri_idx * 3 + 1]);
    edges[8] = v_edge_idx(layer_idx, tri_verts[tri_idx * 3 + 2]);
  }

  HOST_DEVICE void compute_barycentric(int tri_idx, Scalar sx, Scalar sy,
                                       Scalar sz, Scalar& l1, Scalar& l2,
                                       Scalar& l3) const {
    int v0 = tri_verts[tri_idx * 3 + 0];
    int v1 = tri_verts[tri_idx * 3 + 1];
    int v2 = tri_verts[tri_idx * 3 + 2];

    Scalar p0x = sphere_vx[v0], p0y = sphere_vy[v0], p0z = sphere_vz[v0];
    Scalar p1x = sphere_vx[v1], p1y = sphere_vy[v1], p1z = sphere_vz[v1];
    Scalar p2x = sphere_vx[v2], p2y = sphere_vy[v2], p2z = sphere_vz[v2];

    Scalar c12x = p1y * p2z - p1z * p2y;
    Scalar c12y = p1z * p2x - p1x * p2z;
    Scalar c12z = p1x * p2y - p1y * p2x;
    Scalar c20x = p2y * p0z - p2z * p0y;
    Scalar c20y = p2z * p0x - p2x * p0z;
    Scalar c20z = p2x * p0y - p2y * p0x;
    Scalar c01x = p0y * p1z - p0z * p1y;
    Scalar c01y = p0z * p1x - p0x * p1z;
    Scalar c01z = p0x * p1y - p0y * p1x;

    Scalar c0 = c12x * sx + c12y * sy + c12z * sz;
    Scalar c1 = c20x * sx + c20y * sy + c20z * sz;
    Scalar c2 = c01x * sx + c01y * sy + c01z * sz;

    Scalar det = c12x * p0x + c12y * p0y + c12z * p0z;
    Scalar sum = c0 + c1 + c2;
    Scalar denom = std::abs(sum);
    if (denom < Scalar(1e-30)) denom = Scalar(1e-30);
    Scalar inv = (det >= Scalar(0.0) ? Scalar(1.0) : Scalar(-1.0)) / denom;

    l1 = c0 * inv;
    l2 = c1 * inv;
    l3 = c2 * inv;
  }

  HD_INLINE int find_radial_layer(Scalar r) const {
    if (r < radii[0] || r > radii[N_r]) return -1;
    int lo = 0, hi = N_r - 1;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (r < radii[mid + 1]) hi = mid;
      else lo = mid + 1;
    }
    return lo;
  }

  HD_INLINE Scalar compute_zeta(int k, Scalar r) const {
    return (r - radii[k]) / (radii[k + 1] - radii[k]);
  }

  HOST_DEVICE int find_triangle(Scalar sx, Scalar sy, Scalar sz,
                                int tri_hint = -1) const {
    int t = tri_hint;
    if (t < 0 || t >= N_tri) t = 0;

    const int opposite_edge[3] = {1, 2, 0};

    for (int iter = 0; iter < N_tri; iter++) {
      Scalar l1, l2, l3;
      compute_barycentric(t, sx, sy, sz, l1, l2, l3);

      if (l1 >= Scalar(-1e-10) && l2 >= Scalar(-1e-10) && l3 >= Scalar(-1e-10))
        return t;

      Scalar lam[3] = {l1, l2, l3};
      int min_idx = 0;
      if (lam[1] < lam[min_idx]) min_idx = 1;
      if (lam[2] < lam[min_idx]) min_idx = 2;

      int next = tri_neighbor[t * 3 + opposite_edge[min_idx]];
      if (next < 0) return t;
      t = next;
    }
    return t;
  }
};

// =========================================================================
// Host-side owner: builds the local numbering, remapped tables, maps and
// recovery rows from (global mesh, pic-depth mesh_partition).  Also keeps
// the host lookups migration unpack and add_particle need.
// =========================================================================
class prismatic_vertex_recovery;

class prismatic_ptc_mesh_local {
 public:
  // `recovery` may be null when the recovery gather is disabled; the
  // rec_* tables are then left empty (kernels receive null pointers).
  void build(const prismatic_mesh& mesh, const prismatic_mesh_partition& mp,
             const prismatic_vertex_recovery* recovery = nullptr,
             MemType mem = MemType::host_only);

  void copy_to_device();

  prismatic_ptc_mesh_ptrs host_ptrs() const;
  prismatic_ptc_mesh_ptrs dev_ptrs() const;
  prismatic_ptc_mesh_ptrs get_ptrs(exec_tags::host) const {
    return host_ptrs();
  }
  prismatic_ptc_mesh_ptrs get_ptrs(exec_tags::device) const {
    return dev_ptrs();
  }

  // ---- Host-side lookups ----
  // Global tri -> local tri (−1 outside T_halo); size N_tri_global.
  const std::vector<int>& tri_g2l() const { return m_tri_g2l; }
  const std::vector<int>& tri_l2g_host() const { return m_tri_l2g_host; }
  // Sphere vertex / edge local -> global.
  const std::vector<int>& vert_l2g() const { return m_vert_l2g; }
  const std::vector<int>& edge_l2g() const { return m_edge_l2g; }

  int n_tri_local() const { return m_n_tri_local; }
  int n_tri_own() const { return m_n_tri_own; }
  int n_layers() const { return m_n_layers; }
  int k0() const { return m_k0; }
  int lay_own_lo() const { return m_lay_own_lo; }
  int lay_own_hi() const { return m_lay_own_hi; }
  int n_tri_global() const { return m_n_tri_global; }
  size_t max_cell() const { return size_t(m_n_tri_local) * m_n_layers; }

  // World-rank migration parameters (canonical A·K decomposition).
  int n_angular_ranks() const { return m_A; }
  int slab_base() const { return m_slab_base; }
  int slab_rem() const { return m_slab_rem; }

  // ---- Buffers (public for tests) ----
  buffer<Scalar> sphere_vx, sphere_vy, sphere_vz, sphere_theta, sphere_phi;
  buffer<int> tri_verts, tri_edges_s, tri_edge_signs, tri_neighbor;
  buffer<int> map_h, map_v, map_tri, map_rect, map_vert;
  buffer<Scalar> vert_dual_vol, hodge1_inv;
  buffer<int> rec_valence, rec_tri_fan, rec_edge_fan;
  buffer<Scalar> rec_w_int, rec_w_inner, rec_w_outer;
  buffer<int> tri_l2g, tri_ang_rank;

 private:
  const prismatic_mesh* m_mesh = nullptr;

  int m_n_tri_local = 0, m_n_tri_own = 0;
  int m_n_vert_s_local = 0, m_n_vert_s_own = 0;
  int m_n_edge_s_local = 0, m_n_edge_s_own = 0;
  int m_n_layers = 0, m_k0 = 0;
  int m_lay_own_lo = 0, m_lay_own_hi = 0;
  int m_shell_own_lo = 0, m_shell_own_hi = 0;
  int m_n_tri_global = 0, m_N_r_global = 0;
  int m_e_split = 0, m_b_split = 0;
  int m_n_verts_layout = 0, m_n_edges_layout = 0, m_n_faces_layout = 0;
  int m_A = 1, m_slab_base = 0, m_slab_rem = 0;
  Scalar m_rec_r_ref = Scalar(1);
  bool m_has_recovery = false;

  std::vector<int> m_tri_g2l;
  std::vector<int> m_tri_l2g_host;
  std::vector<int> m_vert_l2g;
  std::vector<int> m_edge_l2g;
};

}  // namespace Aperture
