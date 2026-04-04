#pragma once

#include "core/buffer.hpp"
#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_mesh_ptrs.h"
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
  buffer<int> tri_verts;       // [N_tri * 3]: sphere vertex indices per triangle
  buffer<int> tri_edges_s;     // [N_tri * 3]: sphere edge indices per triangle
  buffer<int> tri_edge_signs;  // [N_tri * 3]: orientation signs (+1 or -1)

  // --- Triangle adjacency ---
  // tri_neighbor[t * 3 + j] = triangle across edge j of triangle t (-1 if none)
  buffer<int> tri_neighbor;    // [N_tri * 3]

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

  // Get a prismatic_mesh_ptrs struct filled with host pointers.
  prismatic_mesh_ptrs host_ptrs() const;

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  // Get a prismatic_mesh_ptrs struct filled with device pointers.
  // Requires copy_to_device() to have been called first.
  prismatic_mesh_ptrs dev_ptrs() const;

  // Copy all mesh data from host to device.
  // Switches buffers to MemType::host_device if needed.
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

  void build_sphere_mesh(int L, sphere_mesh& sm);
  void extrude_to_3d(const sphere_mesh& sm);
  void build_incidence(const sphere_mesh& sm);
  void transpose_d1();
  void compute_geometric_dual(const sphere_mesh& sm);
  void tag_boundaries();
  void persist_sphere_data(const sphere_mesh& sm);
};

}  // namespace Aperture
