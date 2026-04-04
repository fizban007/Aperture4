#pragma once

#include "core/buffer.hpp"
#include "core/typedefs_and_constants.h"
#include <map>
#include <vector>

namespace Aperture {

class prismatic_mesh {
 public:
  prismatic_mesh() = default;
  ~prismatic_mesh() = default;

  void build(int L, int N_r, double r_min, double r_max);

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

  // --- Geometry ---
  buffer<Scalar> vert_x, vert_y, vert_z;  // vertex positions, size N_verts
  buffer<Scalar> radii;                     // shell radii, size N_r+1

  // --- Edge data ---
  buffer<Scalar> edge_length;  // size N_edges
  buffer<int> edge_v0, edge_v1;  // endpoint vertex indices, size N_edges

  // --- Face data ---
  buffer<Scalar> face_area;  // size N_faces

  // --- Incidence matrix d1 (face -> edges) in CSR ---
  buffer<int> d1_row_ptr;    // size N_faces + 1
  buffer<int> d1_col_idx;    // nonzeros
  buffer<Scalar> d1_val;     // +1 or -1

  // --- Transpose d1^T (edge -> faces) in CSR ---
  buffer<int> d1t_row_ptr;   // size N_edges + 1
  buffer<int> d1t_col_idx;
  buffer<Scalar> d1t_val;

  // --- Galerkin mass matrices in CSR ---
  // M1: 1-form mass matrix (N_edges x N_edges), symmetric positive definite
  buffer<int> M1_row_ptr;
  buffer<int> M1_col_idx;
  buffer<Scalar> M1_val;
  buffer<Scalar> M1_diag;  // diagonal entries for preconditioner, size N_edges

  // M2: 2-form mass matrix (N_faces x N_faces), symmetric positive definite
  buffer<int> M2_row_ptr;
  buffer<int> M2_col_idx;
  buffer<Scalar> M2_val;

  // M1_inv: precomputed sparse approximate inverse of M1 (block-Jacobi)
  // Same sparsity pattern as M1. Applied as a single spmv.
  buffer<int> M1inv_row_ptr;
  buffer<int> M1inv_col_idx;
  buffer<Scalar> M1inv_val;

  // Sparse matvec helpers
  void spmv_M1(const buffer<Scalar>& x, buffer<Scalar>& y) const;
  void spmv_M1inv(const buffer<Scalar>& x, buffer<Scalar>& y) const;
  void spmv_M2(const buffer<Scalar>& x, buffer<Scalar>& y) const;

  // --- Geometric dual Hodge star (diagonal) ---
  // Computed from actual dual mesh geometry (prism centroids).
  // hodge1_inv[e] = |e| / |e*|  (edge length / dual face area)
  // hodge2[f] = |f*| / |f|     (dual edge length / face area)
  buffer<Scalar> hodge1_inv;  // size N_edges
  buffer<Scalar> hodge2;      // size N_faces

  // --- Boundary tags ---
  // 0 = interior, 1 = inner boundary, 2 = outer boundary
  buffer<int> edge_boundary;  // size N_edges
  buffer<int> face_boundary;  // size N_faces

  // --- Radial layer index for each edge/face ---
  // For damping: which radial layer does this element belong to?
  buffer<int> edge_radial_layer;  // size N_edges, layer index (0..N_r-1 or shell 0..N_r)
  buffer<int> face_radial_layer;  // size N_faces

  // --- Helper: face vertex indices for output ---
  // Triangular faces: 3 vertices. Rectangular faces: 4 vertices.
  // Store separately for simplicity.
  buffer<int> tri_face_v0, tri_face_v1, tri_face_v2;   // size N_tri*(N_r+1)
  buffer<int> rect_face_v0, rect_face_v1, rect_face_v2, rect_face_v3;  // size N_edge_s*N_r

  // --- Indexing helpers ---
  // Horizontal edge global index: shell k, sphere edge e
  int h_edge_idx(int k, int e) const { return k * m_N_edge_s + e; }
  // Vertical edge global index: layer k, sphere vertex s
  int v_edge_idx(int k, int s) const {
    return (m_N_r + 1) * m_N_edge_s + k * m_N_vert_s + s;
  }
  // Triangular face global index: shell k, sphere triangle t
  int tri_face_idx(int k, int t) const { return k * m_N_tri + t; }
  // Rectangular face global index: layer k, sphere edge e
  int rect_face_idx(int k, int e) const {
    return (m_N_r + 1) * m_N_tri + k * m_N_edge_s + e;
  }
  // Vertex global index: shell k, sphere vertex s
  int vert_idx(int k, int s) const { return k * m_N_vert_s + s; }

 private:
  // --- Sphere mesh generation helpers ---
  struct sphere_mesh {
    std::vector<double> vx, vy, vz;           // unit sphere vertex positions
    std::vector<std::array<int, 3>> triangles; // triangle vertex indices
    std::vector<std::array<int, 2>> edges;     // edge vertex indices (a < b)
    // For each triangle, which 3 edges (indices into edges[])
    std::vector<std::array<int, 3>> tri_edges;
    // For each triangle, orientation of each edge (+1 or -1)
    std::vector<std::array<int, 3>> tri_edge_orient;
    // Map from vertex pair to edge index
    std::map<std::pair<int, int>, int> edge_map;

    int add_vertex(double x, double y, double z);
    int get_or_create_edge(int a, int b);
    void build_icosahedron();
    void subdivide();
  };

  void build_sphere_mesh(int L, sphere_mesh& sm);
  void extrude_to_3d(const sphere_mesh& sm);
  void build_incidence(const sphere_mesh& sm);
  void transpose_d1();
  void assemble_mass_matrices(const sphere_mesh& sm);
  void compute_M1_inverse();
  void compute_geometric_dual(const sphere_mesh& sm);
  void tag_boundaries();
};

}  // namespace Aperture
