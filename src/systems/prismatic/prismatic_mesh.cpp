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

  // Step 5: Assemble Galerkin mass matrices (optional, skip for diagonal Hodge)
  // assemble_mass_matrices(sm);

  // Step 6: Compute sparse approximate inverse of M1 (optional, skip for speed)
  // compute_M1_inverse();

  // Step 7: Compute geometric dual Hodge star
  compute_geometric_dual(sm);

  // Step 8: Tag boundaries
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

void prismatic_mesh::assemble_mass_matrices(const sphere_mesh& sm) {
  // Galerkin mass matrices for Whitney forms on the prismatic mesh.
  //
  // Key property: horizontal-vertical cross terms vanish because
  // horizontal Whitney 1-forms are tangential to the sphere and vertical
  // forms are radial. Similarly for 2-forms (tri faces radial, rect faces
  // tangential). So both M₁ and M₂ are block-diagonal in h/v (or tri/rect).
  //
  // For M₁ horizontal block: the integral factors as
  //   M₁[h_{ij,k}, h_{mn,p}] = T(ij,mn) × Z(k,p) × Δr
  // where T is a per-triangle 3×3 matrix (radius-independent) and
  // Z(k,p) = {1/3 if k=p, 1/6 if k≠p} is the interval hat integral.
  //
  // For M₁ vertical block:
  //   M₁[v_i, v_j] = |T_unit| × (1+δ_{ij})/12 × R²_avg / Δr

  // --- Step 1: Compute per-triangle 3×3 edge mass matrices on unit sphere ---
  // T(ij, mn) = ∫_T (λ_i∇λ_j - λ_j∇λ_i)·(λ_m∇λ_n - λ_n∇λ_m) dA
  //
  // Using ∇λ constant on the triangle:
  // T(ij,mn) = (∇λ_j·∇λ_n)I(i,m) - (∇λ_j·∇λ_m)I(i,n)
  //          - (∇λ_i·∇λ_n)I(j,m) + (∇λ_i·∇λ_m)I(j,n)
  // where I(a,b) = |T|(1+δ_{ab})/12

  int N_tri_s = sm.triangles.size();

  // Per-triangle: 3 edges, so 3×3 matrix. Store as flat array [N_tri_s × 9].
  std::vector<double> tri_mass(N_tri_s * 9, 0.0);

  // Pre-compute ∇λ for each unit-sphere triangle
  for (int t = 0; t < N_tri_s; t++) {
    int a = sm.triangles[t][0];
    int b = sm.triangles[t][1];
    int c = sm.triangles[t][2];

    // Triangle vertices on unit sphere
    double p[3][3] = {
        {sm.vx[a], sm.vy[a], sm.vz[a]},
        {sm.vx[b], sm.vy[b], sm.vz[b]},
        {sm.vx[c], sm.vy[c], sm.vz[c]},
    };

    // Edge vectors
    double e01[3] = {p[1][0]-p[0][0], p[1][1]-p[0][1], p[1][2]-p[0][2]};
    double e02[3] = {p[2][0]-p[0][0], p[2][1]-p[0][1], p[2][2]-p[0][2]};

    // Face normal (unnormalized) = e01 × e02
    double n[3] = {
        e01[1]*e02[2] - e01[2]*e02[1],
        e01[2]*e02[0] - e01[0]*e02[2],
        e01[0]*e02[1] - e01[1]*e02[0],
    };
    double area2 = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    double area = area2 / 2.0;

    // Unit normal
    double nn[3] = {n[0]/area2, n[1]/area2, n[2]/area2};

    // ∇λ_i = (edge opposite to i) × normal / (2 * area)
    // Edge opposite to vertex 0: p[2]-p[1]
    // Edge opposite to vertex 1: p[0]-p[2]
    // Edge opposite to vertex 2: p[1]-p[0]
    double opp[3][3];
    for (int d = 0; d < 3; d++) {
      opp[0][d] = p[2][d] - p[1][d];
      opp[1][d] = p[0][d] - p[2][d];
      opp[2][d] = p[1][d] - p[0][d];
    }

    double grad_lambda[3][3];  // grad_lambda[vertex][xyz]
    for (int i = 0; i < 3; i++) {
      // ∇λ_i = (opp_i × n_hat) / (2 * area)
      double cross[3] = {
          opp[i][1]*nn[2] - opp[i][2]*nn[1],
          opp[i][2]*nn[0] - opp[i][0]*nn[2],
          opp[i][0]*nn[1] - opp[i][1]*nn[0],
      };
      for (int d = 0; d < 3; d++) {
        grad_lambda[i][d] = cross[d] / (2.0 * area);
      }
    }

    // Dot products: ∇λ_a · ∇λ_b
    double gdot[3][3];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        gdot[i][j] = 0;
        for (int d = 0; d < 3; d++) {
          gdot[i][j] += grad_lambda[i][d] * grad_lambda[j][d];
        }
      }
    }

    // The 3 triangle edges are (from tri_edges):
    // edge 0: vertices (tri[0], tri[1]) → ij = (0,1)
    // edge 1: vertices (tri[1], tri[2]) → ij = (1,2)
    // edge 2: vertices (tri[0], tri[2]) → ij = (0,2)
    // (using the local vertex indices 0,1,2 within the triangle)
    int ij_pairs[3][2] = {{0,1}, {1,2}, {0,2}};

    // I(a,b) = area * (1 + δ_{ab}) / 12
    auto I_ab = [area](int aa, int bb) -> double {
      return area * (1.0 + (aa == bb ? 1.0 : 0.0)) / 12.0;
    };

    // Compute 3×3 mass matrix T
    for (int ei = 0; ei < 3; ei++) {
      int ii = ij_pairs[ei][0], jj = ij_pairs[ei][1];
      for (int ej = 0; ej < 3; ej++) {
        int mm = ij_pairs[ej][0], nn_idx = ij_pairs[ej][1];
        double val =
            gdot[jj][nn_idx] * I_ab(ii, mm) -
            gdot[jj][mm]     * I_ab(ii, nn_idx) -
            gdot[ii][nn_idx] * I_ab(jj, mm) +
            gdot[ii][mm]     * I_ab(jj, nn_idx);
        tri_mass[t * 9 + ei * 3 + ej] = val;
      }
    }
  }

  // --- Step 2: Build M₁ using COO format, then convert to CSR ---
  // Each prism contributes:
  //   Horizontal block: 6×6 = 36 entries (3 edges × 2 levels)
  //   Vertical block: 3×3 = 9 entries
  //   Total: 45 entries per prism
  // But many are merged when converting to CSR.

  struct coo_entry {
    int row, col;
    Scalar val;
  };
  std::vector<coo_entry> M1_coo;
  std::vector<coo_entry> M2_coo;

  // Reserve approximate space
  M1_coo.reserve(45L * m_N_tri * m_N_r);
  M2_coo.reserve(13L * m_N_tri * m_N_r);

  for (int k = 0; k < m_N_r; k++) {
    double dr = radii[k + 1] - radii[k];
    double r0 = radii[k];
    double r1 = radii[k + 1];
    double R2_avg = (r0 * r0 + r0 * r1 + r1 * r1) / 3.0;

    // Z(k_level, p_level): integral of φ_k φ_p over [0,1]
    // k_level, p_level ∈ {0,1} for bottom/top of this prism
    // Z(0,0) = Z(1,1) = 1/3, Z(0,1) = Z(1,0) = 1/6
    double Z[2][2] = {{1.0/3.0, 1.0/6.0}, {1.0/6.0, 1.0/3.0}};

    for (int t = 0; t < m_N_tri; t++) {
      // --- M₁ horizontal block ---
      // The 6 horizontal edges of this prism:
      // edges[ei] at level kl: global index = h_edge_idx(k + kl, tri_edges[t][ei])
      for (int ei = 0; ei < 3; ei++) {
        for (int kl_i = 0; kl_i < 2; kl_i++) {
          int row = h_edge_idx(k + kl_i, sm.tri_edges[t][ei]);
          for (int ej = 0; ej < 3; ej++) {
            for (int kl_j = 0; kl_j < 2; kl_j++) {
              int col = h_edge_idx(k + kl_j, sm.tri_edges[t][ej]);
              double val = tri_mass[t * 9 + ei * 3 + ej] * Z[kl_i][kl_j] * dr;
              if (std::abs(val) > 1e-20) {
                M1_coo.push_back({row, col, static_cast<Scalar>(val)});
              }
            }
          }
        }
      }

      // --- M₁ vertical block ---
      // 3 vertical edges: v_edge_idx(k, tri_vertex[i])
      int verts[3] = {sm.triangles[t][0], sm.triangles[t][1], sm.triangles[t][2]};
      double unit_area = face_area[tri_face_idx(0, t)];  // r=1 shell area ≈ unit sphere area

      // Actually, face_area at shell 0 is r_min^2 * unit_area. We need the unit sphere area.
      // Unit area = face_area[tri_face_idx(0,t)] / (r_min * r_min)
      double unit_tri_area = unit_area / (m_r_min * m_r_min);

      for (int vi = 0; vi < 3; vi++) {
        int row = v_edge_idx(k, verts[vi]);
        for (int vj = 0; vj < 3; vj++) {
          int col = v_edge_idx(k, verts[vj]);
          double val = unit_tri_area * (1.0 + (vi == vj ? 1.0 : 0.0)) / 12.0
                       * R2_avg / dr;
          M1_coo.push_back({row, col, static_cast<Scalar>(val)});
        }
      }

      // --- M₂ triangular face block (2×2) ---
      // Faces: tri_face_idx(k, t) and tri_face_idx(k+1, t)
      // M₂_tt[kl_i, kl_j] = Δr / (4|T_unit|) × ∫₀¹ φ_ki φ_kj / r(ζ)² dζ
      // ≈ Δr / (4|T_unit|) × Z(ki,kj) / R2_avg  (approximate: 1/r² ≈ 1/R²_avg)
      // More precisely: ∫₀¹ φ_ki φ_kj / r(ζ)² dζ needs exact computation.
      // For now, use the midpoint approximation: 1/r² ≈ 1/((r0+r1)/2)²
      double r_mid2 = 0.25 * (r0 + r1) * (r0 + r1);
      for (int ki = 0; ki < 2; ki++) {
        int row = tri_face_idx(k + ki, t);
        for (int kj = 0; kj < 2; kj++) {
          int col = tri_face_idx(k + kj, t);
          double val = dr / (4.0 * unit_tri_area) * Z[ki][kj] / r_mid2;
          M2_coo.push_back({row, col, static_cast<Scalar>(val)});
        }
      }

      // --- M₂ rectangular face block (3×3) ---
      // Faces: rect_face_idx(k, tri_edges[t][ei])
      // These involve the same triangle integrals as M₁ horizontal.
      // M₂_rr[ei, ej] = T(ij, mn) × 1/Δr  (the Z integral for dζ∧dζ is gone,
      //   replaced by the 1-form wedge structure)
      // Actually: W²_{ij} = (λ_i dλ_j - λ_j dλ_i) ∧ dζ
      // ⟨W²_{ij}, W²_{mn}⟩ = [(λ_i∇λ_j-λ_j∇λ_i)·(λ_m∇λ_n-λ_n∇λ_m)] × (dζ·dζ)
      // dζ·dζ = 1/Δr² in physical space
      // d³x = r² × area × Δr dζ
      // So M₂_rr = (1/Δr²) × ∫₀¹ r² Δr dζ × ∫_T T_angular dA
      //          = (1/Δr) × R²_avg × T(ei,ej)
      // But wait, the r² should cancel like for M₁... let me check.
      // The angular part of W²_{ij} scales as 1/r, and dζ = 1/Δr.
      // |W²_{ij}|² ~ 1/(r² Δr²). d³x ~ r² Δr. So integral ~ 1/Δr. r² cancels!
      for (int ei = 0; ei < 3; ei++) {
        int row = rect_face_idx(k, sm.tri_edges[t][ei]);
        for (int ej = 0; ej < 3; ej++) {
          int col = rect_face_idx(k, sm.tri_edges[t][ej]);
          double val = tri_mass[t * 9 + ei * 3 + ej] / dr;
          if (std::abs(val) > 1e-20) {
            M2_coo.push_back({row, col, static_cast<Scalar>(val)});
          }
        }
      }
    }
  }

  Logger::print_info("M1 COO entries: {}, M2 COO entries: {}",
                     M1_coo.size(), M2_coo.size());

  // --- Step 3: Convert COO to CSR with merging duplicates ---
  auto coo_to_csr = [](std::vector<coo_entry>& coo, int nrows,
                        buffer<int>& row_ptr, buffer<int>& col_idx,
                        buffer<Scalar>& val, buffer<Scalar>* diag = nullptr) {
    // Sort by (row, col)
    std::sort(coo.begin(), coo.end(), [](const coo_entry& a, const coo_entry& b) {
      return (a.row < b.row) || (a.row == b.row && a.col < b.col);
    });

    // Merge duplicates
    std::vector<coo_entry> merged;
    merged.reserve(coo.size());
    for (size_t i = 0; i < coo.size(); ) {
      int r = coo[i].row, c = coo[i].col;
      Scalar s = 0;
      while (i < coo.size() && coo[i].row == r && coo[i].col == c) {
        s += coo[i].val;
        i++;
      }
      merged.push_back({r, c, s});
    }

    // Build CSR
    int nnz = merged.size();
    row_ptr.resize(nrows + 1);
    col_idx.resize(nnz);
    val.resize(nnz);

    row_ptr.assign(0, nrows + 1, 0);
    for (auto& e : merged) {
      row_ptr[e.row + 1]++;
    }
    for (int i = 1; i <= nrows; i++) {
      row_ptr[i] += row_ptr[i - 1];
    }

    for (int i = 0; i < nnz; i++) {
      col_idx[i] = merged[i].col;
      val[i] = merged[i].val;
    }

    // Extract diagonal
    if (diag) {
      diag->resize(nrows);
      diag->assign(0, nrows, 0.0);
      for (int r = 0; r < nrows; r++) {
        for (int j = row_ptr[r]; j < row_ptr[r + 1]; j++) {
          if (col_idx[j] == r) {
            (*diag)[r] = val[j];
            break;
          }
        }
      }
    }
  };

  coo_to_csr(M1_coo, m_N_edges, M1_row_ptr, M1_col_idx, M1_val, &M1_diag);
  coo_to_csr(M2_coo, m_N_faces, M2_row_ptr, M2_col_idx, M2_val);

  Logger::print_info("M1 CSR: {} rows, {} nnz, M2 CSR: {} rows, {} nnz",
                     m_N_edges, M1_row_ptr[m_N_edges],
                     m_N_faces, M2_row_ptr[m_N_faces]);

  // Diagnostic: check M1 diagonal
  {
    Scalar min_diag = 1e30, max_diag = -1e30;
    int zero_count = 0, neg_count = 0;
    for (int i = 0; i < m_N_edges; i++) {
      if (M1_diag[i] <= 0) {
        if (M1_diag[i] == 0) zero_count++;
        else neg_count++;
      }
      min_diag = std::min(min_diag, M1_diag[i]);
      max_diag = std::max(max_diag, M1_diag[i]);
    }
    Logger::print_info("M1 diagonal: min={}, max={}, zeros={}, negatives={}",
                       min_diag, max_diag, zero_count, neg_count);
  }
}

void prismatic_mesh::spmv_M1(const buffer<Scalar>& x, buffer<Scalar>& y) const {
  for (int i = 0; i < m_N_edges; i++) {
    Scalar s = 0;
    for (int j = M1_row_ptr[i]; j < M1_row_ptr[i + 1]; j++) {
      s += M1_val[j] * x[M1_col_idx[j]];
    }
    y[i] = s;
  }
}

void prismatic_mesh::spmv_M2(const buffer<Scalar>& x, buffer<Scalar>& y) const {
  for (int i = 0; i < m_N_faces; i++) {
    Scalar s = 0;
    for (int j = M2_row_ptr[i]; j < M2_row_ptr[i + 1]; j++) {
      s += M2_val[j] * x[M2_col_idx[j]];
    }
    y[i] = s;
  }
}

void prismatic_mesh::spmv_M1inv(const buffer<Scalar>& x,
                                buffer<Scalar>& y) const {
  for (int i = 0; i < m_N_edges; i++) {
    Scalar s = 0;
    for (int j = M1inv_row_ptr[i]; j < M1inv_row_ptr[i + 1]; j++) {
      s += M1inv_val[j] * x[M1inv_col_idx[j]];
    }
    y[i] = s;
  }
}

void prismatic_mesh::compute_M1_inverse() {
  // SPAI (Sparse Approximate Inverse) following Kim & Teixeira (2011).
  //
  // Sparsity pattern P = pattern of M^2 (second-ring neighbors, ~5x fill-in).
  // Each column computed independently via least-squares: min ||A m_k - e_k||_2.
  // Result is symmetrized for leapfrog stability.

  int N = m_N_edges;

  // --- Step 1: Compute sparsity pattern of M² (symbolic) ---
  Logger::print_info("SPAI: computing M^2 sparsity pattern...");

  std::vector<int> P_row_ptr(N + 1, 0);
  std::vector<int> P_col_idx;

  for (int i = 0; i < N; i++) {
    std::set<int> cols;
    for (int ji = M1_row_ptr[i]; ji < M1_row_ptr[i + 1]; ji++) {
      int j = M1_col_idx[ji];
      for (int ki = M1_row_ptr[j]; ki < M1_row_ptr[j + 1]; ki++) {
        cols.insert(M1_col_idx[ki]);
      }
    }
    P_row_ptr[i + 1] = P_row_ptr[i] + static_cast<int>(cols.size());
    for (int c : cols) {
      P_col_idx.push_back(c);
    }
  }

  int P_nnz = P_col_idx.size();
  double fill_ratio = static_cast<double>(P_nnz) / M1_row_ptr[N];
  Logger::print_info("SPAI: M^2 pattern: {} nnz ({:.1f}x fill-in, "
                     "{:.1f} avg nnz/row)",
                     P_nnz, fill_ratio, static_cast<double>(P_nnz) / N);

  // --- Step 2: SPAI column-by-column ---
  Logger::print_info("SPAI: computing approximate inverse...");

  M1inv_row_ptr.resize(N + 1);
  M1inv_col_idx.resize(P_nnz);
  M1inv_val.resize(P_nnz);

  for (int i = 0; i <= N; i++) M1inv_row_ptr[i] = P_row_ptr[i];
  for (int i = 0; i < P_nnz; i++) M1inv_col_idx[i] = P_col_idx[i];
  M1inv_val.assign(0, P_nnz, 0.0);

  for (int k = 0; k < N; k++) {
    // J = column pattern of row k in P (= sparsity of column k since symmetric)
    int J_start = P_row_ptr[k];
    int J_end = P_row_ptr[k + 1];
    int nJ = J_end - J_start;
    if (nJ == 0) continue;

    // Build J and mapping
    std::vector<int> J(nJ);
    std::map<int, int> J_map;
    for (int jj = 0; jj < nJ; jj++) {
      J[jj] = P_col_idx[J_start + jj];
      J_map[J[jj]] = jj;
    }

    // I = union of nonzero row indices in columns J of M1
    std::set<int> I_set;
    for (int jj = 0; jj < nJ; jj++) {
      int j = J[jj];
      for (int mi = M1_row_ptr[j]; mi < M1_row_ptr[j + 1]; mi++) {
        I_set.insert(M1_col_idx[mi]);
      }
    }
    std::vector<int> I_vec(I_set.begin(), I_set.end());
    int nI = I_vec.size();
    std::map<int, int> I_map;
    for (int ii = 0; ii < nI; ii++) I_map[I_vec[ii]] = ii;

    // Build submatrix Abar = M1(I, J), size nI x nJ
    std::vector<double> Abar(nI * nJ, 0.0);
    for (int jj = 0; jj < nJ; jj++) {
      int j = J[jj];
      for (int mi = M1_row_ptr[j]; mi < M1_row_ptr[j + 1]; mi++) {
        auto it = I_map.find(M1_col_idx[mi]);
        if (it != I_map.end()) {
          Abar[it->second * nJ + jj] = M1_val[mi];
        }
      }
    }

    // Build e_k(I)
    std::vector<double> ek(nI, 0.0);
    auto it_k = I_map.find(k);
    if (it_k != I_map.end()) ek[it_k->second] = 1.0;

    // Normal equations: ATA m = ATe, where ATA = nJ x nJ
    std::vector<double> ATA(nJ * nJ, 0.0);
    std::vector<double> ATe(nJ, 0.0);

    for (int ii = 0; ii < nI; ii++) {
      for (int jj = 0; jj < nJ; jj++) {
        double aij = Abar[ii * nJ + jj];
        if (aij == 0.0) continue;
        ATe[jj] += aij * ek[ii];
        for (int kk = jj; kk < nJ; kk++) {
          ATA[jj * nJ + kk] += aij * Abar[ii * nJ + kk];
        }
      }
    }
    for (int jj = 0; jj < nJ; jj++) {
      for (int kk = 0; kk < jj; kk++) {
        ATA[jj * nJ + kk] = ATA[kk * nJ + jj];
      }
    }

    // Cholesky solve
    std::vector<double> Lch(ATA);
    for (int jj = 0; jj < nJ; jj++) {
      for (int kk = 0; kk < jj; kk++) {
        double s = 0;
        for (int ll = 0; ll < kk; ll++) s += Lch[jj * nJ + ll] * Lch[kk * nJ + ll];
        Lch[jj * nJ + kk] = (Lch[jj * nJ + kk] - s) / Lch[kk * nJ + kk];
      }
      double s = 0;
      for (int kk = 0; kk < jj; kk++) s += Lch[jj * nJ + kk] * Lch[jj * nJ + kk];
      double d = Lch[jj * nJ + jj] - s;
      Lch[jj * nJ + jj] = (d > 0) ? std::sqrt(d) : 1e-15;
    }

    // Forward solve: L y = ATe
    std::vector<double> y(nJ);
    for (int jj = 0; jj < nJ; jj++) {
      double s = ATe[jj];
      for (int kk = 0; kk < jj; kk++) s -= Lch[jj * nJ + kk] * y[kk];
      y[jj] = s / Lch[jj * nJ + jj];
    }

    // Back solve: L^T m = y
    std::vector<double> m(nJ);
    for (int jj = nJ - 1; jj >= 0; jj--) {
      double s = y[jj];
      for (int kk = jj + 1; kk < nJ; kk++) s -= Lch[kk * nJ + jj] * m[kk];
      m[jj] = s / Lch[jj * nJ + jj];
    }

    // Store row k
    for (int jj = 0; jj < nJ; jj++) {
      M1inv_val[J_start + jj] = static_cast<Scalar>(m[jj]);
    }
  }

  // --- Step 3: Symmetrize ---
  for (int i = 0; i < N; i++) {
    for (int ji = M1inv_row_ptr[i]; ji < M1inv_row_ptr[i + 1]; ji++) {
      int j = M1inv_col_idx[ji];
      if (j <= i) continue;
      for (int ki = M1inv_row_ptr[j]; ki < M1inv_row_ptr[j + 1]; ki++) {
        if (M1inv_col_idx[ki] == i) {
          Scalar avg = 0.5f * (M1inv_val[ji] + M1inv_val[ki]);
          M1inv_val[ji] = avg;
          M1inv_val[ki] = avg;
          break;
        }
      }
    }
  }

  Logger::print_info("SPAI M1_inv: {} rows, {} nnz ({:.1f}x fill-in)",
                     N, P_nnz, fill_ratio);
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

}  // namespace Aperture
