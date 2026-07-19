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
// Partition scheme (see PHASE_7_SCALABLE_PIC_PLAN.md F3):
//   Angular: the sphere is decomposed into U = 20·4^m congruent level-m
//            "patch units" (m = patch_level); each angular rank owns a
//            contiguous range of units in the canonical path ordering
//            below.  m = 0 with one unit per rank reproduces the
//            historical one-ico-face-per-rank decomposition.
//   Radial:  each rank owns a range of shells [shell_k_lo, shell_k_hi).
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
  // Angular partition — generalized level-m patch units (Phase 7A).
  //
  // The angular axis is decomposed into U = 20·4^m congruent units,
  // m = patch_level ∈ [0, L].  Unit u (GLOBAL numbering) covers the
  // contiguous triangle block [u·4^(L−m), (u+1)·4^(L−m)) — subdivide()
  // pushes the 4 children of a parent consecutively, so a triangle's
  // global unit index is pure arithmetic: unit_of_tri(t) = t >> 2(L−m).
  // At m = 0 units are the 20 icosahedron faces.
  //
  // For rank assignment, units are ordered along a fixed Hamiltonian
  // cycle of the icosahedron's face-adjacency graph (face_path()),
  // times base-4 child order within each face:
  //   path_of_unit(u) = face_path_pos()[u / 4^m]·4^m + u % 4^m.
  // This rank owns PATH units [unit_lo, unit_hi).  The path ordering
  // makes every whole-face group (A ∈ {1,2,4,5,10,20}) a connected band
  // and any aligned power-of-4 unit range a single connected patch.
  //
  // Ownership of shared sphere-edges / sphere-vertices: the LOWEST
  // incident unit in GLOBAL unit numbering owns the element (min over
  // its incident triangles of unit_of_tri).  At m = 0 this reduces
  // exactly to the historical lowest-incident-ico-face rule, and the
  // owning RANK is independent of the m used at fixed A (refining m
  // splits each unit into 4 consecutive global units, preserving the
  // relative order across distinct coarser units).
  // -----------------------------------------------------------------------
  int patch_level = 0;  // m
  int unit_lo = 0;      // owned PATH-ordered unit range [unit_lo, unit_hi)
  int unit_hi = 20;

  // LEGACY whole-face view, consumed by the per-ico-face angular halo
  // plan builder (replaced in 7A.3) and a few tests.  Synced by the
  // factories: [0, 20) when the rank owns the full sphere, [f, f+1)
  // when it owns exactly ico-face f, [-1, -1) otherwise (generalized
  // unit partitions the legacy builder cannot serve).
  int ico_face_lo = 0;
  int ico_face_hi = 20;

  // True when angular ranks are assigned by canonical path-ordered unit
  // ranges (the angular_units()/combined() factories) — peer ranks in
  // halo plans are then angular_rank_of_path_unit values.  False for
  // the legacy identity rank→face factories (ico_face_angular /
  // combined_ico_face), whose peer ranks are ico-face indices.  Decides
  // which angular plan builder prismatic_mesh_partition uses; the
  // legacy wiring is removed in 7A.4.
  bool canonical_rank_order = false;

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
  // full radial range owned.  Rank k owns ico-face k (IDENTITY rank→face
  // order — the pre-7A comm wiring; the generalized factories below use
  // the canonical path order instead).
  static prismatic_partition ico_face_angular(int L, int N_r_global,
                                              int ico_face_idx) {
    auto p = single_rank(L, N_r_global);
    p.ico_face_lo = ico_face_idx;
    p.ico_face_hi = ico_face_idx + 1;
    p.unit_lo = face_path_pos()[ico_face_idx];
    p.unit_hi = p.unit_lo + 1;
    p.angular_rank = ico_face_idx;
    p.n_angular_ranks = 20;
    return p;
  }

  // LEGACY combined decomposition: K radial slabs × 20 angular ico-faces
  // with identity rank→face order.  Kept while the per-face halo plan
  // builder and the 20·K comm wiring remain (removed in 7A.3/7A.4).
  static prismatic_partition combined_ico_face(int L, int N_r_global,
                                               int n_radial_ranks,
                                               int radial_rank,
                                               int ico_face_idx) {
    auto p = radial_slab(L, N_r_global, n_radial_ranks, radial_rank);
    p.ico_face_lo = ico_face_idx;
    p.ico_face_hi = ico_face_idx + 1;
    p.unit_lo = face_path_pos()[ico_face_idx];
    p.unit_hi = p.unit_lo + 1;
    p.angular_rank = ico_face_idx;
    p.n_angular_ranks = 20;
    return p;
  }

  // Generalized angular-only decomposition (Phase 7A): A angular ranks
  // over U = 20·4^m level-m patch units, A | U.  Angular rank a owns
  // path units [a·U/A, (a+1)·U/A).  m defaults to the smallest level
  // with A | 20·4^m; an explicit m must satisfy A | 20·4^m and m ≤ L.
  // Throws std::invalid_argument on an unsatisfiable A or m.
  static prismatic_partition angular_units(int L, int N_r_global, int A,
                                           int angular_rank, int m = -1);

  // Generalized combined decomposition: A angular × K radial ranks,
  // world_rank = radial_rank·A + angular_rank (Phase 7 convention,
  // generalizing the historical radial·20 + face).
  static prismatic_partition combined(int L, int N_r_global, int A, int K,
                                      int world_rank, int m = -1);

  // Smallest patch level m ≥ 0 with A | 20·4^m, or -1 if none exists
  // (A must be of the form 2^j or 5·2^j).
  static int min_patch_level_for(int A);

  // Phase 7E: suggest an angular rank count for a given world size and
  // mesh — the ANGULAR-MAJOR heuristic (plan F10: particles concentrate
  // radially and, for oblique rotators, in latitude, so radial slabs
  // load-imbalance first): the LARGEST valid A with
  //   A | world_size,  A = 2^j or 5·2^j with patch level ≤ L,
  //   K = world/A ≤ N_r, and (when `pic`) N_r / K ≥ 2
  // (the pic-depth radial halos need ≥ 2 shells per slab).
  // Returns 0 when no valid factorization exists.  Cluster-agnostic:
  // shapes are chosen from (world, L, N_r) alone; config
  // n_angular_ranks overrides.
  static int suggest_angular_ranks(int world_size, int L, int N_r,
                                   bool pic = true);

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
    p.patch_level = 0;
    p.unit_lo = 0;
    p.unit_hi = 20;
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

  // -----------------------------------------------------------------------
  // Angular unit arithmetic (Phase 7A).  All O(1): shifts plus the
  // static 20-entry face-path table.
  // -----------------------------------------------------------------------
  int units_per_face() const { return pow4L(patch_level); }  // = 4^m
  int n_units() const { return 20 * units_per_face(); }      // = U

  // GLOBAL unit index of a global sub-triangle.
  int unit_of_tri(int global_tri_idx) const {
    return global_tri_idx >> (2 * (L - patch_level));
  }

  // Canonical (Hamiltonian-path) position of a GLOBAL unit index.
  int path_of_unit(int unit) const {
    const int upf = units_per_face();
    return face_path_pos()[unit / upf] * upf + unit % upf;
  }

  bool owns_path_unit(int path_unit) const {
    return path_unit >= unit_lo && path_unit < unit_hi;
  }
  bool owns_unit(int unit) const { return owns_path_unit(path_of_unit(unit)); }

  // Angular rank owning a path unit under the uniform A-way split used
  // by angular_units()/combined().  NOT meaningful for the legacy
  // identity-ordered factories (ico_face_angular / combined_ico_face),
  // whose rank→face map bypasses the path ordering.
  int angular_rank_of_path_unit(int path_unit) const {
    return static_cast<int>(static_cast<long>(path_unit) * n_angular_ranks /
                            n_units());
  }

  // True iff this rank owns EVERY unit of the given ico-face.
  bool owns_ico_face(int ico_face) const {
    const int upf = units_per_face();
    const int p0 = face_path_pos()[ico_face] * upf;
    return p0 >= unit_lo && p0 + upf <= unit_hi;
  }

  bool owns_sub_tri(int global_tri_idx) const {
    return owns_unit(unit_of_tri(global_tri_idx));
  }

  // True if this rank covers the full angular span (no angular partition),
  // equivalently: angular ownership questions degenerate to "everyone
  // owns everything angular".  Purely radial decomposition sets this.
  bool owns_all_angular() const {
    return unit_lo == 0 && unit_hi == n_units();
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

  // Angular ownership of a sphere-edge.  Rule: lowest incident unit (in
  // GLOBAL unit numbering, via the edge's 2 adjacent triangles) owns the
  // edge.  At m = 0 this is the historical lowest-incident-ico-face rule.
  bool owns_sphere_edge(int sphere_edge_idx) const;

  // Angular ownership of a sphere-vertex.  Rule: lowest incident unit
  // over the vertex's triangle fan (valence 5 at icosahedron corners,
  // 6 elsewhere).  At m = 0: lowest incident ico-face.
  bool owns_sphere_vertex(int sphere_vertex_idx) const;

  // Owner in GLOBAL unit numbering (min incident unit); -1 without an
  // attached topology.
  int owner_unit_of_sphere_edge(int sphere_edge_idx) const;
  int owner_unit_of_sphere_vertex(int sphere_vertex_idx) const;

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
  // Canonical face ordering (Phase 7A): a fixed Hamiltonian cycle on the
  // icosahedron's face-adjacency graph (the dodecahedral graph).
  // face_path()[p] = ico-face at path position p; face_path_pos() is the
  // inverse.  Consecutive entries (cyclically) share an ico-edge, so any
  // contiguous run of whole faces in path order is a connected band.
  // -----------------------------------------------------------------------
  static const std::array<int, 20>& face_path();
  static const std::array<int, 20>& face_path_pos();

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------
  static constexpr int pow4L(int L) {
    int p = 1;
    for (int i = 0; i < L; ++i) p *= 4;
    return p;
  }

 private:
  // Set the generalized angular-unit fields (validates A, m; syncs the
  // legacy ico_face_lo/hi view).  Shared by angular_units()/combined().
  void set_angular_units(int A, int angular_rank, int m);

  const icosphere_topology* m_topology = nullptr;
};

}  // namespace Aperture
