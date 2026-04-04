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

  // Step 5: Compute circumcentric dual Hodge star
  compute_geometric_dual(sm);

  // Step 6: Tag boundaries
  tag_boundaries();

  // Step 7: Persist sphere data for particle operations
  persist_sphere_data(sm);

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


void prismatic_mesh::compute_geometric_dual(const sphere_mesh& sm) {
  // Compute proper diagonal Hodge star from geometric dual mesh.
  //
  // For each prism, compute the centroid. Then:
  //   hodge2[f] = |f*| / |f|  where |f*| = distance between centroids of
  //     the two prisms sharing face f (= dual edge length)
  //   hodge1_inv[e] = |e| / |e*| where |e*| = area of the polygon formed
  //     by centroids of prisms sharing edge e (= dual face area)

  // --- Step 1: Compute prism circumcenters ---
  // For the Voronoi dual, each prism's dual vertex is at the circumcenter:
  //   Angular position: circumcenter of the triangle (equidistant from 3 vertices)
  //   Radial position: midpoint of the radial interval
  // This guarantees dual faces are perpendicular to primal edges.

  // Helper: compute circumcenter of triangle (p0, p1, p2) in 3D
  auto tri_circumcenter = [](double p0x, double p0y, double p0z,
                             double p1x, double p1y, double p1z,
                             double p2x, double p2y, double p2z,
                             double& ccx, double& ccy, double& ccz) {
    // Circumcenter = p0 + s*(p1-p0) + t*(p2-p0) where s,t solve:
    //   2 * dot(p1-p0, p1-p0) * s + 2 * dot(p1-p0, p2-p0) * t = dot(p1-p0, p1-p0)
    //   2 * dot(p1-p0, p2-p0) * s + 2 * dot(p2-p0, p2-p0) * t = dot(p2-p0, p2-p0)
    // (from |cc - p0|² = |cc - p1|² = |cc - p2|²)
    double d10x = p1x - p0x, d10y = p1y - p0y, d10z = p1z - p0z;
    double d20x = p2x - p0x, d20y = p2y - p0y, d20z = p2z - p0z;
    double a11 = 2.0 * (d10x*d10x + d10y*d10y + d10z*d10z);
    double a12 = 2.0 * (d10x*d20x + d10y*d20y + d10z*d20z);
    double a22 = 2.0 * (d20x*d20x + d20y*d20y + d20z*d20z);
    double b1 = d10x*d10x + d10y*d10y + d10z*d10z;
    double b2 = d20x*d20x + d20y*d20y + d20z*d20z;
    double det = a11 * a22 - a12 * a12;
    if (std::abs(det) < 1e-30) {
      // Degenerate: fall back to centroid
      ccx = (p0x + p1x + p2x) / 3.0;
      ccy = (p0y + p1y + p2y) / 3.0;
      ccz = (p0z + p1z + p2z) / 3.0;
      return;
    }
    double s = (a22 * b1 - a12 * b2) / det;
    double t = (a11 * b2 - a12 * b1) / det;
    ccx = p0x + s * d10x + t * d20x;
    ccy = p0y + s * d10y + t * d20y;
    ccz = p0z + s * d10z + t * d20z;
  };

  int N_prisms = m_N_tri * m_N_r;
  std::vector<double> cx(N_prisms), cy(N_prisms), cz(N_prisms);

  for (int k = 0; k < m_N_r; k++) {
    // Radial midpoint
    double r_mid = 0.5 * (radii[k] + radii[k + 1]);

    for (int t = 0; t < m_N_tri; t++) {
      int pid = k * m_N_tri + t;
      int a = sm.triangles[t][0];
      int b = sm.triangles[t][1];
      int c = sm.triangles[t][2];

      // Compute circumcenter of the triangle on the unit sphere
      double ccx_unit, ccy_unit, ccz_unit;
      tri_circumcenter(sm.vx[a], sm.vy[a], sm.vz[a],
                       sm.vx[b], sm.vy[b], sm.vz[b],
                       sm.vx[c], sm.vy[c], sm.vz[c],
                       ccx_unit, ccy_unit, ccz_unit);

      // Project onto unit sphere (circumcenter of a spherical triangle
      // is approximately the projected circumcenter for small triangles)
      double r_cc = std::sqrt(ccx_unit*ccx_unit + ccy_unit*ccy_unit +
                              ccz_unit*ccz_unit);
      if (r_cc > 0) {
        ccx_unit /= r_cc;
        ccy_unit /= r_cc;
        ccz_unit /= r_cc;
      }

      // Place at radial midpoint
      cx[pid] = r_mid * ccx_unit;
      cy[pid] = r_mid * ccy_unit;
      cz[pid] = r_mid * ccz_unit;
    }
  }

  // --- Step 2: Compute hodge2[f] = |f*| / |f| ---
  // For each face, find the two prisms sharing it and compute centroid distance.
  hodge2.resize(m_N_faces);

  // Build a map: for each sphere triangle, which sphere triangles are its
  // neighbors (sharing an edge). Each triangle has 3 edges, each shared by
  // exactly 2 triangles.
  // neighbor_tri[t][j] = triangle index sharing edge j of triangle t
  // For each sphere edge, find the 2 triangles sharing it
  std::vector<std::array<int, 2>> edge_tris(m_N_edge_s, {-1, -1});
  for (int t = 0; t < m_N_tri; t++) {
    for (int j = 0; j < 3; j++) {
      int e = sm.tri_edges[t][j];
      if (edge_tris[e][0] == -1) {
        edge_tris[e][0] = t;
      } else {
        edge_tris[e][1] = t;
      }
    }
  }

  // Triangular faces (on shells): shared by prisms in layers k-1 and k
  for (int k = 0; k <= m_N_r; k++) {
    for (int t = 0; t < m_N_tri; t++) {
      int fi = tri_face_idx(k, t);
      if (k == 0) {
        // Inner boundary: only one prism (layer 0, above)
        // Use distance from face centroid to prism centroid, doubled
        int pid_above = 0 * m_N_tri + t;
        double fx = (vert_x[vert_idx(0, sm.triangles[t][0])] +
                     vert_x[vert_idx(0, sm.triangles[t][1])] +
                     vert_x[vert_idx(0, sm.triangles[t][2])]) / 3.0;
        double fy = (vert_y[vert_idx(0, sm.triangles[t][0])] +
                     vert_y[vert_idx(0, sm.triangles[t][1])] +
                     vert_y[vert_idx(0, sm.triangles[t][2])]) / 3.0;
        double fz = (vert_z[vert_idx(0, sm.triangles[t][0])] +
                     vert_z[vert_idx(0, sm.triangles[t][1])] +
                     vert_z[vert_idx(0, sm.triangles[t][2])]) / 3.0;
        double dx = cx[pid_above] - fx;
        double dy = cy[pid_above] - fy;
        double dz = cz[pid_above] - fz;
        double dist = std::sqrt(dx*dx + dy*dy + dz*dz) * 2.0;
        hodge2[fi] = (face_area[fi] > 0) ? dist / face_area[fi] : 0;
      } else if (k == m_N_r) {
        // Outer boundary: only one prism (layer N_r-1, below)
        int pid_below = (m_N_r - 1) * m_N_tri + t;
        double fx = (vert_x[vert_idx(m_N_r, sm.triangles[t][0])] +
                     vert_x[vert_idx(m_N_r, sm.triangles[t][1])] +
                     vert_x[vert_idx(m_N_r, sm.triangles[t][2])]) / 3.0;
        double fy = (vert_y[vert_idx(m_N_r, sm.triangles[t][0])] +
                     vert_y[vert_idx(m_N_r, sm.triangles[t][1])] +
                     vert_y[vert_idx(m_N_r, sm.triangles[t][2])]) / 3.0;
        double fz = (vert_z[vert_idx(m_N_r, sm.triangles[t][0])] +
                     vert_z[vert_idx(m_N_r, sm.triangles[t][1])] +
                     vert_z[vert_idx(m_N_r, sm.triangles[t][2])]) / 3.0;
        double dx = cx[pid_below] - fx;
        double dy = cy[pid_below] - fy;
        double dz = cz[pid_below] - fz;
        double dist = std::sqrt(dx*dx + dy*dy + dz*dz) * 2.0;
        hodge2[fi] = (face_area[fi] > 0) ? dist / face_area[fi] : 0;
      } else {
        // Interior: two prisms (layer k-1 below, layer k above)
        int pid_below = (k - 1) * m_N_tri + t;
        int pid_above = k * m_N_tri + t;
        double dx = cx[pid_above] - cx[pid_below];
        double dy = cy[pid_above] - cy[pid_below];
        double dz = cz[pid_above] - cz[pid_below];
        double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        hodge2[fi] = (face_area[fi] > 0) ? dist / face_area[fi] : 0;
      }
    }
  }

  // Rectangular faces (between shells): shared by 2 angular neighbor prisms
  for (int k = 0; k < m_N_r; k++) {
    for (int e = 0; e < m_N_edge_s; e++) {
      int fi = rect_face_idx(k, e);
      int t0 = edge_tris[e][0];
      int t1 = edge_tris[e][1];
      if (t0 == -1 || t1 == -1) {
        hodge2[fi] = 0;
        continue;
      }
      int pid0 = k * m_N_tri + t0;
      int pid1 = k * m_N_tri + t1;
      double dx = cx[pid1] - cx[pid0];
      double dy = cy[pid1] - cy[pid0];
      double dz = cz[pid1] - cz[pid0];
      double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
      hodge2[fi] = (face_area[fi] > 0) ? dist / face_area[fi] : 0;
    }
  }

  // --- Step 3: Compute hodge1_inv[e] = |e| / |e*| ---
  hodge1_inv.resize(m_N_edges);

  // Horizontal edges: dual face is a rectangle (approx) formed by
  // 4 prism centroids (2 angular × 2 radial layers).
  // More precisely: the 2 triangle neighbors at layers k-1 and k.
  for (int k = 0; k <= m_N_r; k++) {
    for (int e = 0; e < m_N_edge_s; e++) {
      int ei = h_edge_idx(k, e);

      int t0 = edge_tris[e][0];
      int t1 = edge_tris[e][1];

      // Collect centroids of all prisms containing this edge
      // Layer k-1 (below shell k) and layer k (above shell k)
      std::vector<double> px, py, pz;
      if (k > 0 && t0 >= 0) {
        int pid = (k-1) * m_N_tri + t0;
        px.push_back(cx[pid]); py.push_back(cy[pid]); pz.push_back(cz[pid]);
      }
      if (k > 0 && t1 >= 0) {
        int pid = (k-1) * m_N_tri + t1;
        px.push_back(cx[pid]); py.push_back(cy[pid]); pz.push_back(cz[pid]);
      }
      if (k < m_N_r && t1 >= 0) {
        int pid = k * m_N_tri + t1;
        px.push_back(cx[pid]); py.push_back(cy[pid]); pz.push_back(cz[pid]);
      }
      if (k < m_N_r && t0 >= 0) {
        int pid = k * m_N_tri + t0;
        px.push_back(cx[pid]); py.push_back(cy[pid]); pz.push_back(cz[pid]);
      }

      // Compute polygon area by fan triangulation from centroid
      int np = px.size();
      double area = 0.0;
      if (np >= 3) {
        double mx = 0, my = 0, mz = 0;
        for (int i = 0; i < np; i++) { mx += px[i]; my += py[i]; mz += pz[i]; }
        mx /= np; my /= np; mz /= np;
        for (int i = 0; i < np; i++) {
          int j = (i + 1) % np;
          double ax = px[i] - mx, ay = py[i] - my, az = pz[i] - mz;
          double bx = px[j] - mx, by = py[j] - my, bz = pz[j] - mz;
          double cx_v = ay * bz - az * by;
          double cy_v = az * bx - ax * bz;
          double cz_v = ax * by - ay * bx;
          area += 0.5 * std::sqrt(cx_v*cx_v + cy_v*cy_v + cz_v*cz_v);
        }
      }

      // For boundary edges with only 2 prisms, the fan triangulation gives
      // zero area (degenerate polygon). Approximate the dual face as a
      // rectangle: distance between the 2 centroids × half the radial span.
      if (np == 2 && area < 1e-30) {
        double dx = px[1] - px[0], dy = py[1] - py[0], dz = pz[1] - pz[0];
        double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        // Half radial span: distance from shell to nearest prism centroid
        double half_dr = 0;
        if (k == 0) half_dr = radii[1] - radii[0];
        else half_dr = radii[k] - radii[k-1];
        area = dist * half_dr * 0.5;
      } else if (np < 2) {
        // Single prism: mirror to estimate area
        // Use interior value from neighboring shell
        area = 0;
      }

      hodge1_inv[ei] = (area > 0) ? edge_length[ei] / area : 0;
    }
  }

  // Vertical edges: dual face is a pentagon/hexagon formed by
  // centroids of the 5-6 prisms sharing this vertex at this layer.
  // Build vertex-to-triangle adjacency for the sphere mesh.
  std::vector<std::vector<int>> vert_tris(m_N_vert_s);
  for (int t = 0; t < m_N_tri; t++) {
    for (int j = 0; j < 3; j++) {
      vert_tris[sm.triangles[t][j]].push_back(t);
    }
  }

  for (int k = 0; k < m_N_r; k++) {
    for (int s = 0; s < m_N_vert_s; s++) {
      int ei = v_edge_idx(k, s);
      auto& tris = vert_tris[s];
      int np = tris.size();  // 5 or 6

      // Collect centroids of prisms around this vertex in layer k
      std::vector<double> px(np), py(np), pz(np);
      for (int i = 0; i < np; i++) {
        int pid = k * m_N_tri + tris[i];
        px[i] = cx[pid];
        py[i] = cy[pid];
        pz[i] = cz[pid];
      }

      // Sort centroids by angle around the vertical edge
      // The edge midpoint is at the vertex position at mid-radius
      double emx = 0.5 * (vert_x[vert_idx(k, s)] + vert_x[vert_idx(k+1, s)]);
      double emy = 0.5 * (vert_y[vert_idx(k, s)] + vert_y[vert_idx(k+1, s)]);
      double emz = 0.5 * (vert_z[vert_idx(k, s)] + vert_z[vert_idx(k+1, s)]);

      // Use the radial direction as the normal for the polygon plane
      double nr = std::sqrt(emx*emx + emy*emy + emz*emz);
      double nx = emx / nr, ny = emy / nr, nz = emz / nr;

      // Build a local 2D coordinate system tangent to the sphere
      // u = arbitrary tangent, v = n × u
      double ux, uy, uz;
      if (std::abs(nx) < 0.9) {
        ux = 0; uy = -nz; uz = ny;  // n × x_hat
      } else {
        ux = nz; uy = 0; uz = -nx;  // n × y_hat
      }
      double unorm = std::sqrt(ux*ux + uy*uy + uz*uz);
      ux /= unorm; uy /= unorm; uz /= unorm;
      double vx = ny*uz - nz*uy;
      double vy = nz*ux - nx*uz;
      double vz = nx*uy - ny*ux;

      // Project centroids to 2D and sort by angle
      std::vector<double> angles(np);
      for (int i = 0; i < np; i++) {
        double dx = px[i] - emx, dy = py[i] - emy, dz = pz[i] - emz;
        double u_coord = dx*ux + dy*uy + dz*uz;
        double v_coord = dx*vx + dy*vy + dz*vz;
        angles[i] = std::atan2(v_coord, u_coord);
      }
      // Sort by angle
      std::vector<int> order(np);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&](int a, int b) { return angles[a] < angles[b]; });

      // Compute polygon area by fan triangulation
      double area = 0.0;
      for (int i = 0; i < np; i++) {
        int j = (i + 1) % np;
        int i0 = order[i], i1 = order[j];
        double ax = px[i0] - emx, ay = py[i0] - emy, az = pz[i0] - emz;
        double bx = px[i1] - emx, by = py[i1] - emy, bz = pz[i1] - emz;
        double cx_v = ay * bz - az * by;
        double cy_v = az * bx - ax * bz;
        double cz_v = ax * by - ay * bx;
        area += 0.5 * std::sqrt(cx_v*cx_v + cy_v*cy_v + cz_v*cz_v);
      }

      hodge1_inv[ei] = (area > 0) ? edge_length[ei] / area : 0;
    }
  }

  Logger::print_info("Geometric dual Hodge computed: hodge1_inv [{:.4f}, {:.4f}], "
                     "hodge2 [{:.6f}, {:.6f}]",
                     *std::min_element(&hodge1_inv[0], &hodge1_inv[m_N_edges-1]),
                     *std::max_element(&hodge1_inv[0], &hodge1_inv[m_N_edges-1]),
                     *std::min_element(&hodge2[0], &hodge2[m_N_faces-1]),
                     *std::max_element(&hodge2[0], &hodge2[m_N_faces-1]));
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

void prismatic_mesh::persist_sphere_data(const sphere_mesh& sm) {
  // Store unit sphere vertex positions
  sphere_vx.resize(m_N_vert_s);
  sphere_vy.resize(m_N_vert_s);
  sphere_vz.resize(m_N_vert_s);
  for (int s = 0; s < m_N_vert_s; s++) {
    sphere_vx[s] = sm.vx[s];
    sphere_vy[s] = sm.vy[s];
    sphere_vz[s] = sm.vz[s];
  }

  // Store triangle vertex/edge/orientation data
  tri_verts.resize(m_N_tri * 3);
  tri_edges_s.resize(m_N_tri * 3);
  tri_edge_signs.resize(m_N_tri * 3);
  for (int t = 0; t < m_N_tri; t++) {
    for (int j = 0; j < 3; j++) {
      tri_verts[t * 3 + j] = sm.triangles[t][j];
      tri_edges_s[t * 3 + j] = sm.tri_edges[t][j];
      tri_edge_signs[t * 3 + j] = sm.tri_edge_orient[t][j];
    }
  }

  // Build triangle adjacency: for each triangle edge, find the neighboring
  // triangle across that edge.
  tri_neighbor.resize(m_N_tri * 3);
  tri_neighbor.assign(0, m_N_tri * 3, -1);

  // Map each sphere edge to its two incident triangles
  std::vector<std::array<int, 2>> edge_tris(m_N_edge_s, {-1, -1});
  for (int t = 0; t < m_N_tri; t++) {
    for (int j = 0; j < 3; j++) {
      int e = sm.tri_edges[t][j];
      if (edge_tris[e][0] == -1) {
        edge_tris[e][0] = t;
      } else {
        edge_tris[e][1] = t;
      }
    }
  }

  for (int t = 0; t < m_N_tri; t++) {
    for (int j = 0; j < 3; j++) {
      int e = sm.tri_edges[t][j];
      tri_neighbor[t * 3 + j] =
          (edge_tris[e][0] == t) ? edge_tris[e][1] : edge_tris[e][0];
    }
  }

  Logger::print_info("Sphere data persisted: {} triangles, {} adjacency entries",
                     m_N_tri, m_N_tri * 3);
}

// ============================================================================
// Particle-related queries
// ============================================================================

void prismatic_mesh::compute_barycentric(int tri_idx, Scalar sx, Scalar sy,
                                         Scalar sz, Scalar& l1, Scalar& l2,
                                         Scalar& l3) const {
  // Compute barycentric coordinates of point (sx,sy,sz) in triangle tri_idx
  // on the unit sphere. Uses the cross-product formula:
  //   n = (v1-v0) x (v2-v0)  [face normal]
  //   l0 = [(v1-p) x (v2-p)] . n / (n . n)
  //   l1 = [(v2-p) x (v0-p)] . n / (n . n)
  //   l2 = 1 - l0 - l1
  int v0 = tri_verts[tri_idx * 3 + 0];
  int v1 = tri_verts[tri_idx * 3 + 1];
  int v2 = tri_verts[tri_idx * 3 + 2];

  Scalar p0x = sphere_vx[v0], p0y = sphere_vy[v0], p0z = sphere_vz[v0];
  Scalar p1x = sphere_vx[v1], p1y = sphere_vy[v1], p1z = sphere_vz[v1];
  Scalar p2x = sphere_vx[v2], p2y = sphere_vy[v2], p2z = sphere_vz[v2];

  // Face normal n = (p1-p0) x (p2-p0)
  Scalar e1x = p1x - p0x, e1y = p1y - p0y, e1z = p1z - p0z;
  Scalar e2x = p2x - p0x, e2y = p2y - p0y, e2z = p2z - p0z;
  Scalar nx = e1y * e2z - e1z * e2y;
  Scalar ny = e1z * e2x - e1x * e2z;
  Scalar nz = e1x * e2y - e1y * e2x;
  Scalar n_dot_n = nx * nx + ny * ny + nz * nz;

  // Sub-triangle (p, v1, v2): cross product (v1-p) x (v2-p)
  Scalar d1x = p1x - sx, d1y = p1y - sy, d1z = p1z - sz;
  Scalar d2x = p2x - sx, d2y = p2y - sy, d2z = p2z - sz;
  Scalar cx0 = d1y * d2z - d1z * d2y;
  Scalar cy0 = d1z * d2x - d1x * d2z;
  Scalar cz0 = d1x * d2y - d1y * d2x;
  l1 = (cx0 * nx + cy0 * ny + cz0 * nz) / n_dot_n;

  // Sub-triangle (p, v2, v0): cross product (v2-p) x (v0-p)
  Scalar d0x = p0x - sx, d0y = p0y - sy, d0z = p0z - sz;
  Scalar cx1 = d2y * d0z - d2z * d0y;
  Scalar cy1 = d2z * d0x - d2x * d0z;
  Scalar cz1 = d2x * d0y - d2y * d0x;
  l2 = (cx1 * nx + cy1 * ny + cz1 * nz) / n_dot_n;

  l3 = 1.0 - l1 - l2;
}

int prismatic_mesh::find_radial_layer(Scalar r) const {
  if (r < radii[0] || r > radii[m_N_r]) return -1;
  // Binary search for k such that radii[k] <= r < radii[k+1]
  int lo = 0, hi = m_N_r - 1;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (r < radii[mid + 1]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

Scalar prismatic_mesh::compute_zeta(int k, Scalar r) const {
  return (r - radii[k]) / (radii[k + 1] - radii[k]);
}

int prismatic_mesh::find_triangle(Scalar sx, Scalar sy, Scalar sz,
                                  int tri_hint) const {
  // Walk algorithm: start from tri_hint, compute barycentric coords,
  // walk toward the most negative coordinate.
  // Falls back to brute force if hint is invalid.
  int t = tri_hint;
  if (t < 0 || t >= m_N_tri) t = 0;

  // Opposite edge for each local vertex index:
  // l0 < 0 -> cross edge 1 (v1-v2), l1 < 0 -> cross edge 2 (v0-v2),
  // l2 < 0 -> cross edge 0 (v0-v1)
  static const int opposite_edge[3] = {1, 2, 0};

  for (int iter = 0; iter < m_N_tri; iter++) {
    Scalar l1, l2, l3;
    compute_barycentric(t, sx, sy, sz, l1, l2, l3);

    if (l1 >= -1e-10 && l2 >= -1e-10 && l3 >= -1e-10) {
      return t;  // Found the containing triangle
    }

    // Walk toward the most negative coordinate
    Scalar lam[3] = {l1, l2, l3};
    int min_idx = 0;
    if (lam[1] < lam[min_idx]) min_idx = 1;
    if (lam[2] < lam[min_idx]) min_idx = 2;

    int next = tri_neighbor[t * 3 + opposite_edge[min_idx]];
    if (next < 0) return t;  // At mesh boundary, best we can do
    t = next;
  }

  // Should not reach here, but brute-force fallback
  Scalar best_min = -1e30;
  int best_t = 0;
  for (int ti = 0; ti < m_N_tri; ti++) {
    Scalar l1, l2, l3;
    compute_barycentric(ti, sx, sy, sz, l1, l2, l3);
    Scalar min_l = std::min({l1, l2, l3});
    if (min_l > best_min) {
      best_min = min_l;
      best_t = ti;
    }
  }
  return best_t;
}

void prismatic_mesh::prism_edge_indices(int tri_idx, int layer_idx,
                                        int edges[9]) const {
  // Bottom horizontal edges (shell layer_idx)
  edges[0] = h_edge_idx(layer_idx, tri_edges_s[tri_idx * 3 + 0]);
  edges[1] = h_edge_idx(layer_idx, tri_edges_s[tri_idx * 3 + 1]);
  edges[2] = h_edge_idx(layer_idx, tri_edges_s[tri_idx * 3 + 2]);

  // Top horizontal edges (shell layer_idx + 1)
  edges[3] = h_edge_idx(layer_idx + 1, tri_edges_s[tri_idx * 3 + 0]);
  edges[4] = h_edge_idx(layer_idx + 1, tri_edges_s[tri_idx * 3 + 1]);
  edges[5] = h_edge_idx(layer_idx + 1, tri_edges_s[tri_idx * 3 + 2]);

  // Vertical edges
  edges[6] = v_edge_idx(layer_idx, tri_verts[tri_idx * 3 + 0]);
  edges[7] = v_edge_idx(layer_idx, tri_verts[tri_idx * 3 + 1]);
  edges[8] = v_edge_idx(layer_idx, tri_verts[tri_idx * 3 + 2]);
}

}  // namespace Aperture
