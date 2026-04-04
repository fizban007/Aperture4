#include "systems/prismatic/prismatic_mesh.h"
#include "utils/logger.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

namespace Aperture {

// ============================================================================
// Sphere mesh helpers
// ============================================================================

int prismatic_mesh::sphere_mesh::add_vertex(double x, double y, double z) {
  double r = std::sqrt(x * x + y * y + z * z);
  vx.push_back(x / r);
  vy.push_back(y / r);
  vz.push_back(z / r);
  return vx.size() - 1;
}

int prismatic_mesh::sphere_mesh::get_or_create_edge(int a, int b) {
  auto key = std::make_pair(std::min(a, b), std::max(a, b));
  auto it = edge_map.find(key);
  if (it != edge_map.end()) return it->second;
  int idx = edges.size();
  edges.push_back({key.first, key.second});
  edge_map[key] = idx;
  return idx;
}

void prismatic_mesh::sphere_mesh::build_icosahedron() {
  // Golden ratio
  double phi = (1.0 + std::sqrt(5.0)) / 2.0;

  // 12 vertices of the icosahedron
  add_vertex(-1, phi, 0);
  add_vertex(1, phi, 0);
  add_vertex(-1, -phi, 0);
  add_vertex(1, -phi, 0);

  add_vertex(0, -1, phi);
  add_vertex(0, 1, phi);
  add_vertex(0, -1, -phi);
  add_vertex(0, 1, -phi);

  add_vertex(phi, 0, -1);
  add_vertex(phi, 0, 1);
  add_vertex(-phi, 0, -1);
  add_vertex(-phi, 0, 1);

  // 20 triangular faces (consistent CCW orientation when viewed from outside)
  int faces[20][3] = {
      {0, 11, 5},  {0, 5, 1},   {0, 1, 7},   {0, 7, 10},  {0, 10, 11},
      {1, 5, 9},   {5, 11, 4},  {11, 10, 2},  {10, 7, 6},  {7, 1, 8},
      {3, 9, 4},   {3, 4, 2},   {3, 2, 6},    {3, 6, 8},   {3, 8, 9},
      {4, 9, 5},   {2, 4, 11},  {6, 2, 10},   {8, 6, 7},   {9, 8, 1},
  };

  for (int i = 0; i < 20; i++) {
    int a = faces[i][0], b = faces[i][1], c = faces[i][2];
    triangles.push_back({a, b, c});

    int e0 = get_or_create_edge(a, b);
    int e1 = get_or_create_edge(b, c);
    int e2 = get_or_create_edge(a, c);
    tri_edges.push_back({e0, e1, e2});

    // Determine orientation: edge (min,max) vs triangle vertex order
    // Edge e0 connects a-b. If a < b, edge is oriented a->b, same as tri => +1
    // If a > b, edge is oriented b->a (since min first), opposite => -1
    auto orient = [](int from, int to) -> int {
      return (from < to) ? +1 : -1;
    };
    // Triangle boundary circuit: a→b→c→a
    // Third edge goes c→a, stored as (min,max), so sign is orient(c,a)
    tri_edge_orient.push_back({orient(a, b), orient(b, c), orient(c, a)});
  }
}

void prismatic_mesh::sphere_mesh::subdivide() {
  std::vector<std::array<int, 3>> new_triangles;
  std::vector<std::array<int, 3>> new_tri_edges;
  std::vector<std::array<int, 3>> new_tri_edge_orient;

  // Cache of edge midpoints: old_edge_index -> new_vertex_index
  std::map<int, int> edge_midpoint;

  auto get_midpoint = [&](int edge_idx) -> int {
    auto it = edge_midpoint.find(edge_idx);
    if (it != edge_midpoint.end()) return it->second;
    int a = edges[edge_idx][0], b = edges[edge_idx][1];
    int mid = add_vertex(vx[a] + vx[b], vy[a] + vy[b], vz[a] + vz[b]);
    edge_midpoint[edge_idx] = mid;
    return mid;
  };

  for (size_t i = 0; i < triangles.size(); i++) {
    int a = triangles[i][0], b = triangles[i][1], c = triangles[i][2];
    int e_ab = tri_edges[i][0], e_bc = tri_edges[i][1], e_ac = tri_edges[i][2];

    int m_ab = get_midpoint(e_ab);
    int m_bc = get_midpoint(e_bc);
    int m_ac = get_midpoint(e_ac);

    // 4 sub-triangles, preserving CCW orientation
    // Triangle (a, m_ab, m_ac), (m_ab, b, m_bc), (m_ac, m_bc, c), (m_ab, m_bc, m_ac)
    std::array<int, 3> sub_tris[4] = {
        {a, m_ab, m_ac},
        {m_ab, b, m_bc},
        {m_ac, m_bc, c},
        {m_ab, m_bc, m_ac},
    };

    for (auto& tri : sub_tris) {
      new_triangles.push_back(tri);
      int v0 = tri[0], v1 = tri[1], v2 = tri[2];
      int ne0 = get_or_create_edge(v0, v1);
      int ne1 = get_or_create_edge(v1, v2);
      int ne2 = get_or_create_edge(v0, v2);
      new_tri_edges.push_back({ne0, ne1, ne2});

      auto orient = [](int from, int to) -> int {
        return (from < to) ? +1 : -1;
      };
      // Triangle boundary: v0→v1→v2→v0. Third edge goes v2→v0.
      new_tri_edge_orient.push_back(
          {orient(v0, v1), orient(v1, v2), orient(v2, v0)});
    }
  }

  triangles = std::move(new_triangles);
  tri_edges = std::move(new_tri_edges);
  tri_edge_orient = std::move(new_tri_edge_orient);

  // Rebuild edge list: keep only edges referenced by current triangles
  std::set<int> used_edges;
  for (auto& te : tri_edges) {
    used_edges.insert(te[0]);
    used_edges.insert(te[1]);
    used_edges.insert(te[2]);
  }

  // Build old-to-new index mapping
  std::map<int, int> edge_remap;
  std::vector<std::array<int, 2>> new_edges;
  std::map<std::pair<int, int>, int> new_edge_map;
  for (int old_idx : used_edges) {
    int new_idx = new_edges.size();
    edge_remap[old_idx] = new_idx;
    new_edges.push_back(edges[old_idx]);
    auto key = std::make_pair(edges[old_idx][0], edges[old_idx][1]);
    new_edge_map[key] = new_idx;
  }

  // Remap tri_edges
  for (auto& te : tri_edges) {
    te[0] = edge_remap[te[0]];
    te[1] = edge_remap[te[1]];
    te[2] = edge_remap[te[2]];
  }

  edges = std::move(new_edges);
  edge_map = std::move(new_edge_map);
}

// ============================================================================
// Main build method
// ============================================================================

void prismatic_mesh::build(int L, int N_r, double r_min, double r_max) {
  m_L = L;
  m_N_r = N_r;
  m_r_min = r_min;
  m_r_max = r_max;

  // Step 1: Build the sphere mesh
  sphere_mesh sm;
  build_sphere_mesh(L, sm);

  // Step 2: Extrude to 3D
  extrude_to_3d(sm);

  // Step 3: Build incidence matrix d1
  build_incidence(sm);

  // Step 4: Transpose d1 -> d1^T
  transpose_d1();

  // Step 5: Compute lumped Hodge star
  compute_hodge(sm);

  // Step 6: Tag boundaries
  tag_boundaries();

  Logger::print_info(
      "Prismatic mesh built: L={}, N_r={}, {} vertices, {} edges, {} faces",
      m_L, m_N_r, m_N_verts, m_N_edges, m_N_faces);
}

void prismatic_mesh::build_sphere_mesh(int L, sphere_mesh& sm) {
  sm.build_icosahedron();
  for (int l = 0; l < L; l++) {
    sm.subdivide();
  }

  m_N_tri = sm.triangles.size();
  m_N_vert_s = sm.vx.size();
  m_N_edge_s = sm.edges.size();

  Logger::print_info(
      "Sphere mesh: {} vertices, {} edges, {} triangles",
      m_N_vert_s, m_N_edge_s, m_N_tri);
}

void prismatic_mesh::extrude_to_3d(const sphere_mesh& sm) {
  // Compute radii (geometric spacing)
  radii.resize(m_N_r + 1);
  double log_ratio = std::log(m_r_max / m_r_min) / m_N_r;
  for (int k = 0; k <= m_N_r; k++) {
    radii[k] = m_r_min * std::exp(k * log_ratio);
  }

  // Compute total counts
  m_N_verts = m_N_vert_s * (m_N_r + 1);
  m_N_edges = m_N_edge_s * (m_N_r + 1) + m_N_vert_s * m_N_r;
  m_N_faces = m_N_tri * (m_N_r + 1) + m_N_edge_s * m_N_r;

  // Allocate and fill vertex positions
  vert_x.resize(m_N_verts);
  vert_y.resize(m_N_verts);
  vert_z.resize(m_N_verts);

  for (int k = 0; k <= m_N_r; k++) {
    double r = radii[k];
    for (int s = 0; s < m_N_vert_s; s++) {
      int idx = vert_idx(k, s);
      vert_x[idx] = r * sm.vx[s];
      vert_y[idx] = r * sm.vy[s];
      vert_z[idx] = r * sm.vz[s];
    }
  }

  // Build edge list with lengths and endpoints
  edge_length.resize(m_N_edges);
  edge_v0.resize(m_N_edges);
  edge_v1.resize(m_N_edges);

  // Horizontal edges
  for (int k = 0; k <= m_N_r; k++) {
    double r = radii[k];
    for (int e = 0; e < m_N_edge_s; e++) {
      int a = sm.edges[e][0], b = sm.edges[e][1];
      int idx = h_edge_idx(k, e);
      int va = vert_idx(k, a), vb = vert_idx(k, b);
      edge_v0[idx] = va;
      edge_v1[idx] = vb;
      // Arc length on shell: r * angle between unit vectors
      double dot = sm.vx[a] * sm.vx[b] + sm.vy[a] * sm.vy[b] +
                   sm.vz[a] * sm.vz[b];
      dot = std::max(-1.0, std::min(1.0, dot));
      edge_length[idx] = r * std::acos(dot);
    }
  }

  // Vertical edges
  for (int k = 0; k < m_N_r; k++) {
    double dr = radii[k + 1] - radii[k];
    for (int s = 0; s < m_N_vert_s; s++) {
      int idx = v_edge_idx(k, s);
      edge_v0[idx] = vert_idx(k, s);
      edge_v1[idx] = vert_idx(k + 1, s);
      edge_length[idx] = dr;
    }
  }

  // Build face vertex arrays for output
  int n_tri_faces = m_N_tri * (m_N_r + 1);
  int n_rect_faces = m_N_edge_s * m_N_r;
  tri_face_v0.resize(n_tri_faces);
  tri_face_v1.resize(n_tri_faces);
  tri_face_v2.resize(n_tri_faces);
  rect_face_v0.resize(n_rect_faces);
  rect_face_v1.resize(n_rect_faces);
  rect_face_v2.resize(n_rect_faces);
  rect_face_v3.resize(n_rect_faces);

  face_area.resize(m_N_faces);

  // Triangular faces
  for (int k = 0; k <= m_N_r; k++) {
    double r = radii[k];
    for (int t = 0; t < m_N_tri; t++) {
      int a = sm.triangles[t][0], b = sm.triangles[t][1],
          c = sm.triangles[t][2];
      int fi = tri_face_idx(k, t);
      int local = k * m_N_tri + t;
      tri_face_v0[local] = vert_idx(k, a);
      tri_face_v1[local] = vert_idx(k, b);
      tri_face_v2[local] = vert_idx(k, c);

      // Area of spherical triangle: r^2 * excess angle
      // For small triangles, approximate as flat triangle area scaled by r^2
      // Cross product of two edge vectors on the unit sphere
      double ax = sm.vx[b] - sm.vx[a], ay = sm.vy[b] - sm.vy[a],
             az = sm.vz[b] - sm.vz[a];
      double bx = sm.vx[c] - sm.vx[a], by = sm.vy[c] - sm.vy[a],
             bz = sm.vz[c] - sm.vz[a];
      double cx = ay * bz - az * by;
      double cy = az * bx - ax * bz;
      double cz = ax * by - ay * bx;
      double flat_area = 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
      face_area[fi] = r * r * flat_area;
    }
  }

  // Rectangular faces
  for (int k = 0; k < m_N_r; k++) {
    double r0 = radii[k], r1 = radii[k + 1];
    double dr = r1 - r0;
    for (int e = 0; e < m_N_edge_s; e++) {
      int a = sm.edges[e][0], b = sm.edges[e][1];
      int fi = rect_face_idx(k, e);
      int local = k * m_N_edge_s + e;
      rect_face_v0[local] = vert_idx(k, a);
      rect_face_v1[local] = vert_idx(k, b);
      rect_face_v2[local] = vert_idx(k + 1, b);
      rect_face_v3[local] = vert_idx(k + 1, a);

      // Area: trapezoid = dr * avg_arc_length
      double dot = sm.vx[a] * sm.vx[b] + sm.vy[a] * sm.vy[b] +
                   sm.vz[a] * sm.vz[b];
      dot = std::max(-1.0, std::min(1.0, dot));
      double angle = std::acos(dot);
      face_area[fi] = dr * 0.5 * (r0 + r1) * angle;
    }
  }
}

void prismatic_mesh::build_incidence(const sphere_mesh& sm) {
  // Build d1 in CSR format: for each face, list its boundary edges with signs
  // Triangular faces have 3 edges, rectangular faces have 4 edges
  int n_tri_faces = m_N_tri * (m_N_r + 1);
  int n_rect_faces = m_N_edge_s * m_N_r;
  int nnz = 3 * n_tri_faces + 4 * n_rect_faces;

  d1_row_ptr.resize(m_N_faces + 1);
  d1_col_idx.resize(nnz);
  d1_val.resize(nnz);

  int ptr = 0;

  // Triangular faces on each shell
  for (int k = 0; k <= m_N_r; k++) {
    for (int t = 0; t < m_N_tri; t++) {
      int fi = tri_face_idx(k, t);
      d1_row_ptr[fi] = ptr;

      // Three edges of this triangle on shell k
      for (int j = 0; j < 3; j++) {
        int sphere_edge = sm.tri_edges[t][j];
        int orient = sm.tri_edge_orient[t][j];
        d1_col_idx[ptr] = h_edge_idx(k, sphere_edge);
        d1_val[ptr] = orient;
        ptr++;
      }
    }
  }

  // Rectangular faces in each layer
  for (int k = 0; k < m_N_r; k++) {
    for (int e = 0; e < m_N_edge_s; e++) {
      int fi = rect_face_idx(k, e);
      d1_row_ptr[fi] = ptr;

      int a = sm.edges[e][0], b = sm.edges[e][1];  // a < b by construction

      // Boundary circuit (right-hand rule for outward normal):
      // bottom horizontal (a->b, +1) -> right vertical (b, k->k+1, +1)
      // -> top horizontal (b->a, so edge a->b with -1) -> left vertical (a, k+1->k, so -1)

      // Bottom horizontal edge: sphere edge e at shell k, oriented a->b = +1
      d1_col_idx[ptr] = h_edge_idx(k, e);
      d1_val[ptr] = +1;
      ptr++;

      // Right vertical: vertex b, layer k, oriented upward = +1
      d1_col_idx[ptr] = v_edge_idx(k, b);
      d1_val[ptr] = +1;
      ptr++;

      // Top horizontal edge: sphere edge e at shell k+1, oriented a->b
      // but circuit goes b->a here, so sign = -1
      d1_col_idx[ptr] = h_edge_idx(k + 1, e);
      d1_val[ptr] = -1;
      ptr++;

      // Left vertical: vertex a, layer k, oriented upward
      // but circuit goes from top to bottom here, so sign = -1
      d1_col_idx[ptr] = v_edge_idx(k, a);
      d1_val[ptr] = -1;
      ptr++;
    }
  }

  d1_row_ptr[m_N_faces] = ptr;

  if (ptr != nnz) {
    Logger::print_err("d1 CSR build error: ptr={} != nnz={}", ptr, nnz);
  }
}

void prismatic_mesh::transpose_d1() {
  // Standard CSR transpose: d1 is (N_faces x N_edges), d1^T is (N_edges x N_faces)
  int nnz = d1_row_ptr[m_N_faces];

  d1t_row_ptr.resize(m_N_edges + 1);
  d1t_col_idx.resize(nnz);
  d1t_val.resize(nnz);

  // Count entries per column (= per edge)
  std::vector<int> count(m_N_edges, 0);
  for (int i = 0; i < nnz; i++) {
    count[d1_col_idx[i]]++;
  }

  // Prefix sum for row pointers
  d1t_row_ptr[0] = 0;
  for (int e = 0; e < m_N_edges; e++) {
    d1t_row_ptr[e + 1] = d1t_row_ptr[e] + count[e];
  }

  // Fill in values
  std::vector<int> offset(m_N_edges, 0);
  for (int f = 0; f < m_N_faces; f++) {
    for (int j = d1_row_ptr[f]; j < d1_row_ptr[f + 1]; j++) {
      int e = d1_col_idx[j];
      int pos = d1t_row_ptr[e] + offset[e];
      d1t_col_idx[pos] = f;
      d1t_val[pos] = d1_val[j];
      offset[e]++;
    }
  }
}

void prismatic_mesh::compute_hodge(const sphere_mesh& sm) {
  // Lumped Hodge star using barycentric dual.
  //
  // For the 1-form Hodge (edges):
  //   hodge1[e] = dual_area[e] / edge_length[e]
  //   hodge1_inv[e] = edge_length[e] / dual_area[e]
  //
  // The dual area for an edge is the sum of (face_area / num_edges_of_face)
  // for all faces incident to that edge. This is the "barycentric" dual area.
  //
  // For the 2-form Hodge (faces):
  //   hodge2[f] = dual_length[f] / face_area[f]
  //   hodge2_inv[f] = face_area[f] / dual_length[f]
  //
  // The dual length for a face is the sum of (edge_length / num_faces_of_edge)
  // for all edges of that face. This is the barycentric dual length.

  hodge1_inv.resize(m_N_edges);
  hodge2.resize(m_N_faces);

  // Compute dual area for each edge (using d1^T: each edge knows its faces)
  buffer<Scalar> dual_area(m_N_edges, MemType::host_only);
  dual_area.assign(0, m_N_edges, 0.0);

  for (int f = 0; f < m_N_faces; f++) {
    int n_edges_of_face = d1_row_ptr[f + 1] - d1_row_ptr[f];
    Scalar contribution = face_area[f] / n_edges_of_face;
    for (int j = d1_row_ptr[f]; j < d1_row_ptr[f + 1]; j++) {
      int e = d1_col_idx[j];
      dual_area[e] += contribution;
    }
  }

  for (int e = 0; e < m_N_edges; e++) {
    if (dual_area[e] > 0) {
      hodge1_inv[e] = edge_length[e] / dual_area[e];
    } else {
      hodge1_inv[e] = 0.0;
    }
  }

  // Compute dual length for each face
  // Using d1: each face knows its edges
  // Count how many faces each edge belongs to (from d1^T)
  std::vector<int> edge_face_count(m_N_edges, 0);
  for (int e = 0; e < m_N_edges; e++) {
    edge_face_count[e] = d1t_row_ptr[e + 1] - d1t_row_ptr[e];
  }

  for (int f = 0; f < m_N_faces; f++) {
    Scalar dual_length = 0.0;
    for (int j = d1_row_ptr[f]; j < d1_row_ptr[f + 1]; j++) {
      int e = d1_col_idx[j];
      if (edge_face_count[e] > 0) {
        dual_length += edge_length[e] / edge_face_count[e];
      }
    }
    if (face_area[f] > 0) {
      hodge2[f] = dual_length / face_area[f];
    } else {
      hodge2[f] = 0.0;
    }
  }
}

void prismatic_mesh::tag_boundaries() {
  edge_boundary.resize(m_N_edges);
  face_boundary.resize(m_N_faces);
  edge_radial_layer.resize(m_N_edges);
  face_radial_layer.resize(m_N_faces);

  edge_boundary.assign(0, m_N_edges, 0);
  face_boundary.assign(0, m_N_faces, 0);

  // Horizontal edges: on shell k
  for (int k = 0; k <= m_N_r; k++) {
    for (int e = 0; e < m_N_edge_s; e++) {
      int idx = h_edge_idx(k, e);
      edge_radial_layer[idx] = k;
      if (k == 0) edge_boundary[idx] = 1;       // inner boundary
      if (k == m_N_r) edge_boundary[idx] = 2;   // outer boundary
    }
  }

  // Vertical edges: in layer k (between shell k and k+1)
  for (int k = 0; k < m_N_r; k++) {
    for (int s = 0; s < m_N_vert_s; s++) {
      int idx = v_edge_idx(k, s);
      edge_radial_layer[idx] = k;
      if (k == 0) edge_boundary[idx] = 1;
      if (k == m_N_r - 1) edge_boundary[idx] = 2;
    }
  }

  // Triangular faces: on shell k
  for (int k = 0; k <= m_N_r; k++) {
    for (int t = 0; t < m_N_tri; t++) {
      int idx = tri_face_idx(k, t);
      face_radial_layer[idx] = k;
      if (k == 0) face_boundary[idx] = 1;
      if (k == m_N_r) face_boundary[idx] = 2;
    }
  }

  // Rectangular faces: in layer k
  for (int k = 0; k < m_N_r; k++) {
    for (int e = 0; e < m_N_edge_s; e++) {
      int idx = rect_face_idx(k, e);
      face_radial_layer[idx] = k;
      if (k == 0) face_boundary[idx] = 1;
      if (k == m_N_r - 1) face_boundary[idx] = 2;
    }
  }
}

}  // namespace Aperture
