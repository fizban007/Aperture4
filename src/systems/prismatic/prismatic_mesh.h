#pragma once

#include "core/buffer.hpp"
#include "core/exec_tags.h"
#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_mesh_ptrs.h"
#include <map>
#include <vector>

namespace Aperture {

class prismatic_mesh {
 public:
  prismatic_mesh() = default;
  ~prismatic_mesh() = default;

  // Build the prismatic mesh with a log-spaced radial grid.
  //
  // `n_ghost_inner` / `n_ghost_outer` prepend / append ghost radial
  // layers below r_min / above r_max.  The physical domain is shells
  // [n_ghost_inner, n_ghost_inner + N_r], and the ghosts give symmetric
  // face-averaging in the shift cross-term at what would otherwise be
  // the one-sided physical boundary.  r_min, r_max, N_r in the config
  // refer to the physical domain; internally m_N_r = N_r + n_ghost_inner
  // + n_ghost_outer.
  void build(int L, int N_r, double r_min, double r_max,
             int n_ghost_inner = 0, int n_ghost_outer = 0);

  // Phase 7D: SPHERE-ONLY build for distributed runs.  Persists the
  // sphere tables, radii, counts and the double-precision angular
  // geometry (solid angles, arc angles, dual-cell angles) — everything
  // O(4^L) — but never allocates a (N_r+1)·N_s-sized 3D array.  Local
  // builders compute per-element 3D geometry from these tables via
  // prismatic_mesh_geom.h (bit-identical to the global arrays, pinned
  // by test).
  void build_sphere_only(int L, int N_r, double r_min, double r_max,
                         int n_ghost_inner = 0, int n_ghost_outer = 0);

  // True when the global 3D per-cochain arrays exist (full build()).
  bool has_3d() const { return m_has_3d; }

  // --- Mesh parameters ---
  int m_L = 0;       // subdivision level
  int m_N_r = 0;     // number of radial layers

  double m_r_min = 1.0;
  double m_r_max = 10.0;

  // --- Counts on the unit sphere ---
  int m_N_tri = 0;     // number of triangles = 20 * 4^L
  int m_N_vert_s = 0;  // vertices per shell = 10 * 4^L + 2
  int m_N_edge_s = 0;  // edges per shell = 30 * 4^L

  // --- Total 3D counts ---
  int m_N_verts = 0;   // = m_N_vert_s * (N_r + 1)
  int m_N_edges = 0;   // = m_N_edge_s * (N_r+1) + m_N_vert_s * N_r
  int m_N_faces = 0;   // = m_N_tri * (N_r+1) + m_N_edge_s * N_r

  // --- Geometry (spherical coordinates) ---
  // Vertex positions in (r, θ, φ).  Cartesian (x, y, z) is derivable as
  //   x = r sin(θ) cos(φ),  y = r sin(θ) sin(φ),  z = r cos(θ).
  // All face areas / edge lengths are intrinsic spherical quantities
  // (arc lengths on shells, spherical-triangle areas via Girard, ruled
  //  trapezoid areas between shells).
  buffer<Scalar> vert_r, vert_theta, vert_phi;  // vertex positions, size N_verts
  buffer<Scalar> radii;                          // shell radii, size N_r+1

  // --- Edge data ---
  buffer<Scalar> edge_length;  // size N_edges
  buffer<int> edge_v0, edge_v1;  // endpoint vertex indices, size N_edges

  // --- Face data ---
  buffer<Scalar> face_area;  // size N_faces
  // Lumped dual (co-volume) of each vertex: sum over adjacent prisms of
  // the hat-function volume integral, computed exactly for the
  // radially-projected prisms (cross-section = solid angle * s^2).
  // Divides the deposited vertex charge cochain to give a density.
  buffer<Scalar> vert_dual_vol;  // size N_verts

  // --- Incidence matrix d1 (face -> edges) in CSR ---
  buffer<int> d1_row_ptr;    // size N_faces + 1
  buffer<int> d1_col_idx;    // nonzeros
  buffer<Scalar> d1_val;     // +1 or -1

  // --- Transpose d1^T (edge -> faces) in CSR ---
  buffer<int> d1t_row_ptr;   // size N_edges + 1
  buffer<int> d1t_col_idx;
  buffer<Scalar> d1t_val;

  // --- Circumcentric dual Hodge star (diagonal) ---
  // Computed from Voronoi dual using prism circumcenters.
  // hodge1_inv[e] = |e| / |e*|  (edge length / dual face area)
  // hodge2[f] = |f*| / |f|     (dual edge length / face area)
  buffer<Scalar> hodge1_inv;  // size N_edges
  buffer<Scalar> hodge2;      // size N_faces

  // --- Boundary tags ---
  // 0 = interior, 1 = inner boundary, 2 = outer boundary
  buffer<int> edge_boundary;  // size N_edges
  buffer<int> face_boundary;  // size N_faces

  // --- Radial layer index for each edge/face ---
  buffer<int> edge_radial_layer;  // size N_edges
  buffer<int> face_radial_layer;  // size N_faces

  // --- Helper: face vertex indices for output ---
  buffer<int> tri_face_v0, tri_face_v1, tri_face_v2;   // size N_tri*(N_r+1)
  buffer<int> rect_face_v0, rect_face_v1, rect_face_v2, rect_face_v3;  // size N_edge_s*N_r

  // --- Persistent sphere mesh data (needed for particle operations) ---
  buffer<Scalar> sphere_vx, sphere_vy, sphere_vz;  // unit sphere vertex positions [N_vert_s]
  buffer<Scalar> sphere_theta, sphere_phi;         // unit sphere angular coords [N_vert_s]
  buffer<int> tri_verts;       // [N_tri * 3]: sphere vertex indices per triangle
  buffer<int> tri_edges_s;     // [N_tri * 3]: sphere edge indices per triangle
  buffer<int> tri_edge_signs;  // [N_tri * 3]: orientation signs (+1 or -1)

  // --- Triangle adjacency ---
  // tri_neighbor[t * 3 + j] = triangle across edge j of triangle t (-1 if none)
  buffer<int> tri_neighbor;    // [N_tri * 3]

  // --- Sphere-edge endpoints (7D; v0 < v1 by construction) ---
  buffer<int> sphere_edge_v0, sphere_edge_v1;   // [N_edge_s]

  // --- Persisted double-precision angular geometry (7D) ---
  // Everything a local builder needs to reproduce the 3D per-cochain
  // geometry bit-exactly: per-element angular factors from the (double)
  // sphere mesh, combined with the radii at build time by
  // prismatic_mesh_geom.h.  Host-only, O(4^L).
  std::vector<double> sph_tri_omega;   // [N_tri] solid angle
  std::vector<double> sph_edge_alpha;  // [N_edge_s] endpoint arc angle
  std::vector<double> sph_edge_beta;   // [N_edge_s] circumcenter-dir arc
  std::vector<double> sph_vert_omega;  // [N_vert_s] dual polygon solid angle
  std::vector<int> sph_edge_tri0, sph_edge_tri1;  // [N_edge_s] adjacent tris
  // Vertex fan CSR (tris ascending — the global accumulation order).
  std::vector<int> sph_vert_tri_offset;  // [N_vert_s + 1]
  std::vector<int> sph_vert_tris;

  // --- Indexing helpers ---
  int h_edge_idx(int k, int e) const { return k * m_N_edge_s + e; }
  int v_edge_idx(int k, int s) const {
    return (m_N_r + 1) * m_N_edge_s + k * m_N_vert_s + s;
  }
  int tri_face_idx(int k, int t) const { return k * m_N_tri + t; }
  int rect_face_idx(int k, int e) const {
    return (m_N_r + 1) * m_N_tri + k * m_N_edge_s + e;
  }
  int vert_idx(int k, int s) const { return k * m_N_vert_s + s; }

  // --- Particle-related queries ---

  // Compute barycentric coordinates of a unit-sphere point in triangle t.
  // Returns (l1, l2, l3) where l1 + l2 + l3 = 1.
  void compute_barycentric(int tri_idx, Scalar sx, Scalar sy, Scalar sz,
                           Scalar& l1, Scalar& l2, Scalar& l3) const;

  // Find radial layer k such that radii[k] <= r < radii[k+1].
  // Returns -1 if r is out of range.
  int find_radial_layer(Scalar r) const;

  // Compute normalized radial coordinate within layer k:
  // zeta = (r - radii[k]) / (radii[k+1] - radii[k])
  Scalar compute_zeta(int k, Scalar r) const;

  // Find which sphere triangle contains the unit-sphere point (sx, sy, sz).
  // Uses a walk algorithm starting from tri_hint (or brute force if hint < 0).
  int find_triangle(Scalar sx, Scalar sy, Scalar sz, int tri_hint = -1) const;

  // Get the 9 global edge indices for prism (tri_idx, layer_idx).
  // edges[0..2] = bottom horizontal (shell k)
  // edges[3..5] = top horizontal (shell k+1)
  // edges[6..8] = vertical
  void prism_edge_indices(int tri_idx, int layer_idx, int edges[9]) const;

  // Number of spherical Lloyd (SCVT) relaxation iterations applied to
  // the subdivided icosahedron before extrusion (set BEFORE build()).
  // The raw subdivided icosahedron's circumcentric duals are offset
  // from primal-edge midpoints by O(h), which makes the diagonal Hodge
  // first-order for quasi-static fields (measured: uniform 2x/level
  // spurious-curl convergence on the exact static dipole).  SCVT
  // relaxation — the standard cure in icosahedral C-grid dynamical
  // cores (Heikes & Randall 1995; MPAS SCVT grids) — re-centers the
  // duals.  0 (default) preserves the historical mesh exactly.
  int sphere_optimize_iters = 0;

 private:
  bool m_has_3d = false;

 public:

  // Get a prismatic_mesh_ptrs struct filled with host pointers.
  prismatic_mesh_ptrs host_ptrs() const;

  // Tag-dispatched pointer access (for ExecPolicy::exec_tag)
  prismatic_mesh_ptrs get_ptrs(exec_tags::host) const { return host_ptrs(); }

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  // Get a prismatic_mesh_ptrs struct filled with device pointers.
  // Requires copy_to_device() to have been called first.
  prismatic_mesh_ptrs dev_ptrs() const;

  prismatic_mesh_ptrs get_ptrs(exec_tags::device) const { return dev_ptrs(); }

  // Copy all mesh data from host to device.
  void copy_to_device();
#endif

 private:
  struct sphere_mesh {
    std::vector<double> vx, vy, vz;
    std::vector<std::array<int, 3>> triangles;
    std::vector<std::array<int, 2>> edges;
    std::vector<std::array<int, 3>> tri_edges;
    std::vector<std::array<int, 3>> tri_edge_orient;
    std::map<std::pair<int, int>, int> edge_map;

    int add_vertex(double x, double y, double z);
    int get_or_create_edge(int a, int b);
    void build_icosahedron();
    void subdivide();
  };

  void build_stages(int L, int N_r, double r_min, double r_max,
                    int n_ghost_inner, int n_ghost_outer, bool with_3d);
  void compute_radii_and_counts();
  void persist_sphere_geometry(const sphere_mesh& sm);
  void build_sphere_mesh(int L, sphere_mesh& sm);
  void optimize_sphere_mesh(sphere_mesh& sm, int iters);
  void extrude_to_3d(const sphere_mesh& sm);
  void build_incidence(const sphere_mesh& sm);
  void transpose_d1();
  void compute_geometric_dual(const sphere_mesh& sm);
  void tag_boundaries();
  void persist_sphere_data(const sphere_mesh& sm);
};

}  // namespace Aperture
