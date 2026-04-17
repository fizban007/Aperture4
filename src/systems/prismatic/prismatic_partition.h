#pragma once

#include <array>
#include <vector>

namespace Aperture {

class icosphere_topology;

// =========================================================================
// Prismatic mesh MPI partition descriptor.
//
// Describes which mesh elements a given MPI rank owns, which it reads as
// ghosts, and which peer ranks it exchanges halos with.  All indices are
// GLOBAL — the same on every rank — so a partition change doesn't shift
// the meaning of any index (critical for parallel HDF5 and restart files).
//
// Partition scheme (see PARALLELIZATION_PLAN.md):
//   Angular: each rank owns one or more of the 20 original icosahedron
//            faces, which after L subdivisions group sub-triangles into
//            20 contiguous ranges of length 4^L.
//   Radial:  each rank owns a range of shells [shell_k_lo, shell_k_hi).
//
// Phase 1 (current): single-rank degenerate case only — the rank owns
// everything, there are no halos, no neighbors.  Downstream code can
// use this API and will run identically to the pre-MPI code.  MPI-aware
// construction and halo tables are populated in later phases.
// =========================================================================
class prismatic_partition {
 public:
  // -----------------------------------------------------------------------
  // Global subdivision parameters (identical on every rank).
  // -----------------------------------------------------------------------
  int L = 0;                // subdivision level
  int N_r_global = 0;       // total physics + ghost radial shells
  int N_tri_global = 0;     // = 20 * 4^L
  int N_vert_s_global = 0;  // = 10 * 4^L + 2
  int N_edge_s_global = 0;  // = 30 * 4^L

  // Total cochain counts across the full mesh (used for global-indexing).
  int N_tri_faces_global = 0;    // = (N_r_global + 1) * N_tri_global
  int N_rect_faces_global = 0;   // = N_r_global * N_edge_s_global
  int N_h_edges_global = 0;      // = (N_r_global + 1) * N_edge_s_global
  int N_v_edges_global = 0;      // = N_r_global * N_vert_s_global
  int N_verts_global = 0;        // = (N_r_global + 1) * N_vert_s_global

  // -----------------------------------------------------------------------
  // Angular partition: this rank owns sub-triangles in ico-face indices
  // [ico_face_lo, ico_face_hi).  Single-rank default: [0, 20).
  // -----------------------------------------------------------------------
  int ico_face_lo = 0;
  int ico_face_hi = 20;

  // -----------------------------------------------------------------------
  // Radial partition: this rank owns physics shells [shell_k_lo, shell_k_hi).
  // Ghost shells read from radial neighbors are in
  //   [shell_k_lo - n_ghost_lower, shell_k_lo)  and
  //   [shell_k_hi, shell_k_hi + n_ghost_upper).
  // These do NOT overlap with the mesh-level n_ghost_inner / n_ghost_outer
  // from prismatic_mesh::build() — those are physical-domain ghosts at the
  // radial boundary, included in N_r_global.  The ghosts here are MPI
  // halo ghosts internal to the partition.
  // -----------------------------------------------------------------------
  int shell_k_lo = 0;
  int shell_k_hi = 0;
  int n_ghost_lower = 0;
  int n_ghost_upper = 0;

  // -----------------------------------------------------------------------
  // MPI rank identity (Phase 3: populated from MPI_Comm_rank on each
  // sub-communicator).  Single-rank fallback: both 0, world_size = 1.
  // -----------------------------------------------------------------------
  int angular_rank = 0;   // rank within comm_angular (0..19 in full MPI)
  int radial_rank = 0;    // rank within comm_radial   (0..K-1)
  int n_angular_ranks = 1;
  int n_radial_ranks = 1;

  // -----------------------------------------------------------------------
  // Angular neighbor list (Phase 3: populated from icosahedron adjacency).
  // Each entry: (global ico-face index of the neighbor, halo class).
  // Halo classes:
  //   edge:     shares an ico-edge (3 per rank in full decomposition)
  //   diagonal: shares only a valence-5 vertex (6 per rank)
  // Single-rank fallback: empty.
  // -----------------------------------------------------------------------
  enum class halo_class : unsigned char { edge, diagonal };
  struct angular_neighbor {
    int ico_face;        // global ico-face index of the neighbor
    halo_class kind;
  };
  std::vector<angular_neighbor> angular_neighbors;

  // -----------------------------------------------------------------------
  // Construction.
  // -----------------------------------------------------------------------

  // Radial-only decomposition factory.  Splits N_r_global radial SLABS
  // evenly across K radial slabs and returns the partition for slab
  // `radial_rank` (one ico-face cover, full angular range).
  //
  // Note: there are N_r_global slabs, but N_r_global + 1 shells.  Under
  // the "slab k owned by the rank owning shell k" convention plus a
  // boundary override (shell N_r_global is owned by the last rank),
  // each rank r's shell range is [slab_lo_r, slab_lo_{r+1}), and the
  // LAST rank's shell range extends one further to include the
  // topmost shell.  Slab range is always [slab_lo_r, slab_lo_{r+1}).
  //
  // Ghost layers of depth 1 are configured on interior slabs; the
  // boundary slabs have no ghost on the exterior side.
  static prismatic_partition radial_slab(int L, int N_r_global,
                                         int n_radial_ranks,
                                         int radial_rank) {
    prismatic_partition p = single_rank(L, N_r_global);
    p.n_radial_ranks = n_radial_ranks;
    p.radial_rank = radial_rank;
    // Even split; low-index slabs absorb the remainder.
    int base = N_r_global / n_radial_ranks;
    int rem = N_r_global - base * n_radial_ranks;
    auto slab_lo = [&](int r) { return r * base + (r < rem ? r : rem); };
    p.shell_k_lo = slab_lo(radial_rank);
    // Last rank absorbs the top shell (there are N_r + 1 shells but N_r
    // slabs; the extra shell at the top is owned by the last rank).
    const bool is_last = (radial_rank == n_radial_ranks - 1);
    p.shell_k_hi = slab_lo(radial_rank + 1) + (is_last ? 1 : 0);
    p.n_ghost_lower = (radial_rank > 0) ? 1 : 0;
    p.n_ghost_upper = is_last ? 0 : 1;
    return p;
  }

  // Angular-only decomposition factory.  20-way split by ico-face, with
  // full radial range owned.  Rank k owns ico-face k.
  static prismatic_partition ico_face_angular(int L, int N_r_global,
                                              int ico_face_idx) {
    auto p = single_rank(L, N_r_global);
    p.ico_face_lo = ico_face_idx;
    p.ico_face_hi = ico_face_idx + 1;
    p.angular_rank = ico_face_idx;
    p.n_angular_ranks = 20;
    return p;
  }

  // Combined decomposition: K radial slabs × 20 angular ico-faces.
  // Rank (radial_rank, ico_face_idx) owns sub-triangles from one ico-face
  // on shells of one radial slab.  Used by production-target partitions
  // (20·K ranks total) and combined-decomposition tests.
  static prismatic_partition combined(int L, int N_r_global,
                                      int n_radial_ranks, int radial_rank,
                                      int ico_face_idx) {
    auto p = radial_slab(L, N_r_global, n_radial_ranks, radial_rank);
    p.ico_face_lo = ico_face_idx;
    p.ico_face_hi = ico_face_idx + 1;
    p.angular_rank = ico_face_idx;
    p.n_angular_ranks = 20;
    return p;
  }

  // Single-rank fallback: rank owns the full mesh.  No MPI, no halos.
  // This is the constructor used until Phase 3 wires in MPI.
  static prismatic_partition single_rank(int L, int N_r_global) {
    prismatic_partition p;
    p.L = L;
    p.N_r_global = N_r_global;
    p.N_tri_global = 20 * pow4L(L);
    p.N_vert_s_global = 10 * pow4L(L) + 2;
    p.N_edge_s_global = 30 * pow4L(L);
    p.N_tri_faces_global  = (N_r_global + 1) * p.N_tri_global;
    p.N_rect_faces_global = N_r_global * p.N_edge_s_global;
    p.N_h_edges_global    = (N_r_global + 1) * p.N_edge_s_global;
    p.N_v_edges_global    = N_r_global * p.N_vert_s_global;
    p.N_verts_global      = (N_r_global + 1) * p.N_vert_s_global;

    p.ico_face_lo = 0;
    p.ico_face_hi = 20;
    p.shell_k_lo = 0;
    // Shells live at k ∈ [0, N_r_global] — single rank owns all N_r+1 of
    // them.  Slab index range [0, N_r_global) is derived via owns_slab().
    p.shell_k_hi = N_r_global + 1;
    p.n_ghost_lower = 0;
    p.n_ghost_upper = 0;
    p.angular_rank = 0;
    p.radial_rank = 0;
    p.n_angular_ranks = 1;
    p.n_radial_ranks = 1;
    // No angular_neighbors (empty).
    return p;
  }

  // True if this is the single-rank fallback (no MPI overhead needed).
  bool is_single_rank() const {
    return n_angular_ranks == 1 && n_radial_ranks == 1;
  }

  // -----------------------------------------------------------------------
  // Ownership queries — primitive axes.
  // -----------------------------------------------------------------------
  bool owns_shell(int shell_k) const {
    return shell_k >= shell_k_lo && shell_k < shell_k_hi;
  }

  // Slab k spans shells [k, k+1].  Convention: slab k owned by rank
  // owning shell k (the lower endpoint).  Valid slab indices are
  // k ∈ [0, N_r_global); owns_slab(k) for k outside that range is
  // always false — the "extra" top shell N_r has no slab above it.
  bool owns_slab(int slab_k) const {
    return slab_k >= 0 && slab_k < N_r_global && owns_shell(slab_k);
  }

  bool owns_ico_face(int ico_face) const {
    return ico_face >= ico_face_lo && ico_face < ico_face_hi;
  }

  bool owns_sub_tri(int global_tri_idx) const {
    const int sz = N_tri_global / 20;  // = 4^L
    int ico_face = global_tri_idx / sz;
    return owns_ico_face(ico_face);
  }

  // True if this rank covers the full angular span (no angular partition),
  // equivalently: angular ownership questions degenerate to "everyone
  // owns everything angular".  Purely radial decomposition sets this.
  bool owns_all_angular() const {
    return ico_face_lo == 0 && ico_face_hi == 20;
  }

  // -----------------------------------------------------------------------
  // Topology-dependent ownership — sphere-edges and sphere-vertices.
  //
  // Attach an icosphere_topology via set_topology() to enable these.
  // Without a topology, these queries fall back to owns_all_angular() —
  // they only work correctly for the radial-only (full-angular) case.
  // -----------------------------------------------------------------------
  void set_topology(const icosphere_topology* t) { m_topology = t; }
  const icosphere_topology* topology() const { return m_topology; }

  // Angular ownership of a sphere-edge.  Rule: lowest-index incident
  // ico-face owns the edge.  Interior sphere-edges are in exactly one
  // ico-face; ico-edge boundary edges are in two and go to the smaller
  // ico-face index.
  bool owns_sphere_edge(int sphere_edge_idx) const;

  // Angular ownership of a sphere-vertex.  Rule: lowest-index incident
  // ico-face owns the vertex.  Interior vertices: 1 face; ico-edge
  // vertices: 2 faces; valence-5 icosahedron corners: 5 faces.  Lowest
  // index always wins.
  bool owns_sphere_vertex(int sphere_vertex_idx) const;

  // -----------------------------------------------------------------------
  // Ownership queries — full cochain index space.
  //
  // Each cochain type has its own global-index scheme documented in
  // PARALLELIZATION_PLAN.md.  Composite ownership = radial-axis check
  // AND angular-axis check.
  //
  // NOTE: for cochains that live on sphere-edges (rect_face, h_edge) or
  // sphere-vertices (v_edge, vertex), the angular-axis ownership depends
  // on the subdivision topology (which sphere-edges sit on ico-edges,
  // which sphere-vertices are at valence-5 corners, etc.).  That
  // topology is populated in Phase 3; until then, these queries are
  // only valid when this rank covers the full angular span.
  // -----------------------------------------------------------------------

  bool owns_tri_face_cochain(int global_idx) const {
    int shell_k = global_idx / N_tri_global;
    int tri_idx = global_idx % N_tri_global;
    return owns_shell(shell_k) && owns_sub_tri(tri_idx);
  }

  bool owns_rect_face_cochain(int global_idx) const {
    int slab_k = global_idx / N_edge_s_global;
    int sphere_edge = global_idx % N_edge_s_global;
    return owns_slab(slab_k) && owns_sphere_edge(sphere_edge);
  }

  bool owns_h_edge_cochain(int global_idx) const {
    int shell_k = global_idx / N_edge_s_global;
    int sphere_edge = global_idx % N_edge_s_global;
    return owns_shell(shell_k) && owns_sphere_edge(sphere_edge);
  }

  bool owns_v_edge_cochain(int global_idx) const {
    int slab_k = global_idx / N_vert_s_global;
    int sphere_vert = global_idx % N_vert_s_global;
    return owns_slab(slab_k) && owns_sphere_vertex(sphere_vert);
  }

  bool owns_vertex_cochain(int global_idx) const {
    int shell_k = global_idx / N_vert_s_global;
    int sphere_vert = global_idx % N_vert_s_global;
    return owns_shell(shell_k) && owns_sphere_vertex(sphere_vert);
  }

  // -----------------------------------------------------------------------
  // Icosahedron face adjacency tables (static — same on every rank).
  //
  // Indexed by ico_face ∈ [0, 20).  edge_neighbors[f] gives the 3
  // ico-faces sharing an ico-edge with f; diagonal_neighbors[f] gives
  // the 6 ico-faces sharing only an ico-vertex (valence-5 corner) with
  // f.  Populated by populate_ico_adjacency() on first use.
  // -----------------------------------------------------------------------
  static const std::array<std::array<int, 3>, 20>& edge_neighbors();
  static const std::array<std::array<int, 6>, 20>& diagonal_neighbors();

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------
 private:
  static constexpr int pow4L(int L) {
    int p = 1;
    for (int i = 0; i < L; ++i) p *= 4;
    return p;
  }

  const icosphere_topology* m_topology = nullptr;
};

}  // namespace Aperture
