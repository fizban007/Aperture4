#include "systems/prismatic/prismatic_mesh.h"
#include "utils/logger.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

namespace Aperture {

// ============================================================================
// Spherical geometry helpers (used by extrude_to_3d and compute_geometric_dual)
// ============================================================================

// Convert (r, theta, phi) → (x, y, z).
static inline void sph_to_cart(double r, double theta, double phi,
                               double& x, double& y, double& z) {
  double sth = std::sin(theta);
  x = r * sth * std::cos(phi);
  y = r * sth * std::sin(phi);
  z = r * std::cos(theta);
}

// Arc angle between two unit vectors (clamped dot + acos).
static inline double arc_angle(double ax, double ay, double az,
                               double bx, double by, double bz) {
  double dot = ax * bx + ay * by + az * bz;
  if (dot > 1.0) dot = 1.0;
  if (dot < -1.0) dot = -1.0;
  return std::acos(dot);
}

// Spherical excess of the triangle with unit-vector vertices (a, b, c),
// via the Van Oosterom–Strang formula.  Result is the solid angle Ω;
// multiply by r² to get surface area on a sphere of radius r.
static inline double sph_triangle_area(double ax, double ay, double az,
                                       double bx, double by, double bz,
                                       double cx, double cy, double cz) {
  // Numerator: |a · (b × c)| (absolute value — area is unsigned)
  double num = ax * (by * cz - bz * cy)
             + ay * (bz * cx - bx * cz)
             + az * (bx * cy - by * cx);
  double denom = 1.0
               + (ax * bx + ay * by + az * bz)
               + (bx * cx + by * cy + bz * cz)
               + (cx * ax + cy * ay + cz * az);
  double omega = 2.0 * std::atan2(std::fabs(num), denom);
  // atan2 returns in [0, π] here because denom can be negative for large
  // triangles (solid angle > π); the formula is valid in that regime too.
  return omega;
}

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

void prismatic_mesh::build(int L, int N_r, double r_min, double r_max,
                           int n_ghost_inner, int n_ghost_outer) {
  build_stages(L, N_r, r_min, r_max, n_ghost_inner, n_ghost_outer, true);
}

void prismatic_mesh::build_sphere_only(int L, int N_r, double r_min,
                                       double r_max, int n_ghost_inner,
                                       int n_ghost_outer) {
  build_stages(L, N_r, r_min, r_max, n_ghost_inner, n_ghost_outer, false);
}

void prismatic_mesh::build_stages(int L, int N_r, double r_min, double r_max,
                                  int n_ghost_inner, int n_ghost_outer,
                                  bool with_3d) {
  // Prepend / append ghost radial layers below r_min / above r_max
  // preserving the log-spacing.  The physical domain is shells
  // [n_ghost_inner, n_ghost_inner + N_r].
  if (n_ghost_inner > 0 || n_ghost_outer > 0) {
    double log_ratio = std::log(r_max / r_min) / N_r;
    r_min *= std::exp(-n_ghost_inner * log_ratio);
    r_max *= std::exp(n_ghost_outer * log_ratio);
    N_r += n_ghost_inner + n_ghost_outer;
  }
  m_L = L;
  m_N_r = N_r;
  m_r_min = r_min;
  m_r_max = r_max;

  // ---- SPHERE STAGE (always; O(4^L) + O(N_r) state only) ----
  sphere_mesh sm;
  build_sphere_mesh(L, sm);
  compute_radii_and_counts();
  persist_sphere_data(sm);
  persist_sphere_geometry(sm);

  // ---- 3D STAGE (skipped under a partition — local builders compute
  //      the per-element geometry via prismatic_mesh_geom.h) ----
  if (with_3d) {
    extrude_to_3d(sm);
    build_incidence(sm);
    transpose_d1();
    compute_geometric_dual(sm);
    tag_boundaries();
    m_has_3d = true;
    Logger::print_info(
        "Prismatic mesh built: L={}, N_r={}, {} vertices, {} edges, {} faces",
        m_L, m_N_r, m_N_verts, m_N_edges, m_N_faces);
  } else {
    Logger::print_info(
        "Prismatic mesh built (sphere stage only): L={}, N_r={}, {} sphere "
        "vertices, {} sphere edges, {} tris",
        m_L, m_N_r, m_N_vert_s, m_N_edge_s, m_N_tri);
  }
}

void prismatic_mesh::compute_radii_and_counts() {
  radii.resize(m_N_r + 1);
  double log_ratio = std::log(m_r_max / m_r_min) / m_N_r;
  for (int k = 0; k <= m_N_r; k++) {
    radii[k] = m_r_min * std::exp(k * log_ratio);
  }
  m_N_verts = gidx_t(m_N_vert_s) * (m_N_r + 1);
  m_N_edges = gidx_t(m_N_edge_s) * (m_N_r + 1) + gidx_t(m_N_vert_s) * m_N_r;
  m_N_faces = gidx_t(m_N_tri) * (m_N_r + 1) + gidx_t(m_N_edge_s) * m_N_r;
}

// Double-precision angular geometry, persisted so local builders can
// reproduce every 3D per-cochain quantity bit-exactly without the global
// arrays.  Expressions are copied VERBATIM from extrude_to_3d /
// compute_geometric_dual (do not "simplify" — bit-identity is the
// contract, pinned by test_prismatic_mesh_geom).
void prismatic_mesh::persist_sphere_geometry(const sphere_mesh& sm) {
  // Tri solid angles (extrude_to_3d's face-area factor).
  sph_tri_omega.resize(m_N_tri);
  for (int t = 0; t < m_N_tri; t++) {
    int a = sm.triangles[t][0], b = sm.triangles[t][1],
        c = sm.triangles[t][2];
    sph_tri_omega[t] = sph_triangle_area(
        sm.vx[a], sm.vy[a], sm.vz[a],
        sm.vx[b], sm.vy[b], sm.vz[b],
        sm.vx[c], sm.vy[c], sm.vz[c]);
  }

  // Edge endpoint arc angles (h-edge lengths / rect areas).
  sph_edge_alpha.resize(m_N_edge_s);
  sphere_edge_v0.resize(m_N_edge_s);
  sphere_edge_v1.resize(m_N_edge_s);
  for (int e = 0; e < m_N_edge_s; e++) {
    int a = sm.edges[e][0], b = sm.edges[e][1];
    sphere_edge_v0[e] = a;
    sphere_edge_v1[e] = b;
    double dot = sm.vx[a] * sm.vx[b] + sm.vy[a] * sm.vy[b] +
                 sm.vz[a] * sm.vz[b];
    dot = std::max(-1.0, std::min(1.0, dot));
    sph_edge_alpha[e] = std::acos(dot);
  }

  // Circumcenter directions (transient) — same expressions as
  // compute_geometric_dual step 1.
  std::vector<double> circ_ux(m_N_tri), circ_uy(m_N_tri), circ_uz(m_N_tri);
  for (int t = 0; t < m_N_tri; t++) {
    int a = sm.triangles[t][0];
    int b = sm.triangles[t][1];
    int c = sm.triangles[t][2];
    double e1x = sm.vx[b] - sm.vx[a];
    double e1y = sm.vy[b] - sm.vy[a];
    double e1z = sm.vz[b] - sm.vz[a];
    double e2x = sm.vx[c] - sm.vx[a];
    double e2y = sm.vy[c] - sm.vy[a];
    double e2z = sm.vz[c] - sm.vz[a];
    double nx = e1y * e2z - e1z * e2y;
    double ny = e1z * e2x - e1x * e2z;
    double nz = e1x * e2y - e1y * e2x;
    if (dual_centroid) {
      // Keep VERBATIM in sync with compute_geometric_dual step 1.
      nx = sm.vx[a] + sm.vx[b] + sm.vx[c];
      ny = sm.vy[a] + sm.vy[b] + sm.vy[c];
      nz = sm.vz[a] + sm.vz[b] + sm.vz[c];
    }
    double nlen = std::sqrt(nx * nx + ny * ny + nz * nz);
    double ux = 0.0, uy = 0.0, uz = 0.0;
    if (nlen > 0) {
      ux = nx / nlen;
      uy = ny / nlen;
      uz = nz / nlen;
      double mx = (sm.vx[a] + sm.vx[b] + sm.vx[c]) / 3.0;
      double my = (sm.vy[a] + sm.vy[b] + sm.vy[c]) / 3.0;
      double mz = (sm.vz[a] + sm.vz[b] + sm.vz[c]) / 3.0;
      if (ux * mx + uy * my + uz * mz < 0) {
        ux = -ux;
        uy = -uy;
        uz = -uz;
      }
    }
    circ_ux[t] = ux;
    circ_uy[t] = uy;
    circ_uz[t] = uz;
  }

  // Sphere-edge adjacent tris (first-come in ascending-t order, matching
  // compute_geometric_dual) + the dual-edge arc between their
  // circumcenter directions.
  sph_edge_tri0.assign(m_N_edge_s, -1);
  sph_edge_tri1.assign(m_N_edge_s, -1);
  for (int t = 0; t < m_N_tri; t++) {
    for (int j = 0; j < 3; j++) {
      int e = sm.tri_edges[t][j];
      if (sph_edge_tri0[e] == -1) {
        sph_edge_tri0[e] = t;
      } else {
        sph_edge_tri1[e] = t;
      }
    }
  }
  sph_edge_beta.assign(m_N_edge_s, 0.0);
  for (int e = 0; e < m_N_edge_s; e++) {
    int t0 = sph_edge_tri0[e], t1 = sph_edge_tri1[e];
    if (t0 >= 0 && t1 >= 0) {
      sph_edge_beta[e] = arc_angle(circ_ux[t0], circ_uy[t0], circ_uz[t0],
                                   circ_ux[t1], circ_uy[t1], circ_uz[t1]);
    }
  }

  // Vertex fans (tris ascending — the global accumulation order) and the
  // dual polygon solid angle around each vertex (compute_geometric_dual's
  // sorted-fan construction, verbatim).
  std::vector<std::vector<int>> vert_tris(m_N_vert_s);
  for (int t = 0; t < m_N_tri; t++) {
    for (int j = 0; j < 3; j++) {
      vert_tris[sm.triangles[t][j]].push_back(t);
    }
  }
  sph_vert_tri_offset.assign(m_N_vert_s + 1, 0);
  sph_vert_tris.clear();
  for (int v = 0; v < m_N_vert_s; v++) {
    sph_vert_tri_offset[v] = int(sph_vert_tris.size());
    sph_vert_tris.insert(sph_vert_tris.end(), vert_tris[v].begin(),
                         vert_tris[v].end());
  }
  sph_vert_tri_offset[m_N_vert_s] = int(sph_vert_tris.size());

  sph_vert_omega.resize(m_N_vert_s);
  for (int sv = 0; sv < m_N_vert_s; sv++) {
    auto& tris = vert_tris[sv];
    int np = int(tris.size());
    double anchor_x = sm.vx[sv], anchor_y = sm.vy[sv], anchor_z = sm.vz[sv];
    double tx, ty, tz;
    if (std::fabs(anchor_x) < 0.9) {
      tx = 0.0; ty = -anchor_z; tz = anchor_y;
    } else {
      tx = anchor_z; ty = 0.0; tz = -anchor_x;
    }
    double tn = std::sqrt(tx * tx + ty * ty + tz * tz);
    tx /= tn; ty /= tn; tz /= tn;
    double bx = anchor_y * tz - anchor_z * ty;
    double by = anchor_z * tx - anchor_x * tz;
    double bz = anchor_x * ty - anchor_y * tx;

    std::vector<double> angles(np);
    for (int i = 0; i < np; i++) {
      int t = tris[i];
      double dx = circ_ux[t] - anchor_x;
      double dy = circ_uy[t] - anchor_y;
      double dz = circ_uz[t] - anchor_z;
      double u_coord = dx * tx + dy * ty + dz * tz;
      double v_coord = dx * bx + dy * by + dz * bz;
      angles[i] = std::atan2(v_coord, u_coord);
    }
    std::vector<int> order(np);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int ia, int ib) { return angles[ia] < angles[ib]; });

    double omega = 0.0;
    for (int i = 0; i < np; i++) {
      int j = (i + 1) % np;
      int t0 = tris[order[i]];
      int t1 = tris[order[j]];
      omega += sph_triangle_area(
          anchor_x, anchor_y, anchor_z,
          circ_ux[t0], circ_uy[t0], circ_uz[t0],
          circ_ux[t1], circ_uy[t1], circ_uz[t1]);
    }
    sph_vert_omega[sv] = omega;
  }
}

void prismatic_mesh::build_sphere_mesh(int L, sphere_mesh& sm) {
  sm.build_icosahedron();
  for (int l = 0; l < L; l++) {
    sm.subdivide();
  }

  if (sphere_optimize_iters > 0) {
    optimize_sphere_mesh(sm, sphere_optimize_iters);
  }

  m_N_tri = sm.triangles.size();
  m_N_vert_s = sm.vx.size();
  m_N_edge_s = sm.edges.size();

  Logger::print_info(
      "Sphere mesh: {} vertices, {} edges, {} triangles",
      m_N_vert_s, m_N_edge_s, m_N_tri);
}

// Spherical Lloyd (SCVT) relaxation: iteratively move each vertex to the
// (spherical) centroid of its Voronoi cell, whose corners are the
// circumcenter directions of the incident triangles.  Connectivity is
// untouched; the 12 original icosahedron vertices are held fixed (they
// are stationary points of the flow by symmetry; pinning avoids drift).
void prismatic_mesh::optimize_sphere_mesh(sphere_mesh& sm, int iters) {
  const int n_v = static_cast<int>(sm.vx.size());
  const int n_t = static_cast<int>(sm.triangles.size());

  // vertex -> incident triangles (fixed connectivity)
  std::vector<std::vector<int>> vtris(n_v);
  for (int t = 0; t < n_t; t++) {
    for (int j = 0; j < 3; j++) vtris[sm.triangles[t][j]].push_back(t);
  }

  auto normalize3 = [](double& x, double& y, double& z) {
    double n = std::sqrt(x * x + y * y + z * z);
    x /= n; y /= n; z /= n;
  };

  std::vector<double> cx(n_t), cy(n_t), cz(n_t);
  double move_rms = 0.0;
  for (int it = 0; it < iters; it++) {
    // circumcenter directions of all triangles
    for (int t = 0; t < n_t; t++) {
      int a = sm.triangles[t][0], b = sm.triangles[t][1],
          c = sm.triangles[t][2];
      double e1x = sm.vx[b] - sm.vx[a], e1y = sm.vy[b] - sm.vy[a],
             e1z = sm.vz[b] - sm.vz[a];
      double e2x = sm.vx[c] - sm.vx[a], e2y = sm.vy[c] - sm.vy[a],
             e2z = sm.vz[c] - sm.vz[a];
      double ux = e1y * e2z - e1z * e2y;
      double uy = e1z * e2x - e1x * e2z;
      double uz = e1x * e2y - e1y * e2x;
      normalize3(ux, uy, uz);
      // orient outward
      if (ux * sm.vx[a] + uy * sm.vy[a] + uz * sm.vz[a] < 0) {
        ux = -ux; uy = -uy; uz = -uz;
      }
      cx[t] = ux; cy[t] = uy; cz[t] = uz;
    }

    move_rms = 0.0;
    for (int s = 12; s < n_v; s++) {  // keep the 12 icosahedron corners
      // Order the incident-triangle fan by azimuth in the tangent plane.
      double vxs = sm.vx[s], vys = sm.vy[s], vzs = sm.vz[s];
      // tangent basis
      double ax = (std::abs(vzs) < 0.9) ? 0 : 1, ay = 0,
             az = (std::abs(vzs) < 0.9) ? 1 : 0;
      double t1x = ay * vzs - az * vys, t1y = az * vxs - ax * vzs,
             t1z = ax * vys - ay * vxs;
      normalize3(t1x, t1y, t1z);
      double t2x = vys * t1z - vzs * t1y, t2y = vzs * t1x - vxs * t1z,
             t2z = vxs * t1y - vys * t1x;

      auto& fan = vtris[s];
      std::vector<std::pair<double, int>> order;
      order.reserve(fan.size());
      for (int t : fan) {
        double dx = cx[t] - vxs, dy = cy[t] - vys, dz = cz[t] - vzs;
        order.push_back(
            {std::atan2(dx * t2x + dy * t2y + dz * t2z,
                        dx * t1x + dy * t1y + dz * t1z), t});
      }
      std::sort(order.begin(), order.end());

      // Voronoi-cell spherical centroid: fan of wedges (v, c_i, c_{i+1}),
      // weight = wedge solid angle, position = normalized wedge mean.
      double gx = 0, gy = 0, gz = 0;
      int m = static_cast<int>(order.size());
      for (int i = 0; i < m; i++) {
        int ta = order[i].second, tb = order[(i + 1) % m].second;
        // wedge solid angle via the vector triple formula (Van Oosterom)
        double d1 = vxs * cx[ta] + vys * cy[ta] + vzs * cz[ta];
        double d2 = vxs * cx[tb] + vys * cy[tb] + vzs * cz[tb];
        double d3 = cx[ta] * cx[tb] + cy[ta] * cy[tb] + cz[ta] * cz[tb];
        double trip = vxs * (cy[ta] * cz[tb] - cz[ta] * cy[tb]) +
                      vys * (cz[ta] * cx[tb] - cx[ta] * cz[tb]) +
                      vzs * (cx[ta] * cy[tb] - cy[ta] * cx[tb]);
        double omega = 2.0 * std::atan2(trip, 1.0 + d1 + d2 + d3);
        double mx = vxs + cx[ta] + cx[tb], my = vys + cy[ta] + cy[tb],
               mz = vzs + cz[ta] + cz[tb];
        normalize3(mx, my, mz);
        gx += omega * mx; gy += omega * my; gz += omega * mz;
      }
      normalize3(gx, gy, gz);
      double dx = gx - vxs, dy = gy - vys, dz = gz - vzs;
      move_rms += dx * dx + dy * dy + dz * dz;
      sm.vx[s] = gx; sm.vy[s] = gy; sm.vz[s] = gz;
    }
  }
  Logger::print_info(
      "Sphere mesh SCVT relaxation: {} iterations, final rms move {:.3e}",
      iters, std::sqrt(move_rms / std::max(1, n_v - 12)));
}

void prismatic_mesh::extrude_to_3d(const sphere_mesh& sm) {
  // Compute radii (geometric spacing)
  radii.resize(m_N_r + 1);
  double log_ratio = std::log(m_r_max / m_r_min) / m_N_r;
  for (int k = 0; k <= m_N_r; k++) {
    radii[k] = m_r_min * std::exp(k * log_ratio);
  }

  // Compute total counts
  m_N_verts = gidx_t(m_N_vert_s) * (m_N_r + 1);
  m_N_edges = gidx_t(m_N_edge_s) * (m_N_r + 1) + gidx_t(m_N_vert_s) * m_N_r;
  m_N_faces = gidx_t(m_N_tri) * (m_N_r + 1) + gidx_t(m_N_edge_s) * m_N_r;

  // Allocate and fill vertex positions in spherical coordinates.
  //   r     = radii[k]
  //   theta = acos(sphere_vz[s])        (polar angle)
  //   phi   = atan2(sphere_vy[s], sphere_vx[s])  (azimuth)
  vert_r.resize(m_N_verts);
  vert_theta.resize(m_N_verts);
  vert_phi.resize(m_N_verts);

  for (int k = 0; k <= m_N_r; k++) {
    double r = radii[k];
    for (int s = 0; s < m_N_vert_s; s++) {
      int idx = vert_idx(k, s);
      double cth = sm.vz[s];
      if (cth > 1.0) cth = 1.0;
      if (cth < -1.0) cth = -1.0;
      vert_r[idx] = r;
      vert_theta[idx] = std::acos(cth);
      vert_phi[idx] = std::atan2(sm.vy[s], sm.vx[s]);
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

      // Exact spherical-triangle area on shell of radius r, via Van
      // Oosterom–Strang (Girard's theorem in stable atan2 form).  No chord
      // approximation: the face is the region of the sphere bounded by
      // the three great-circle arcs.
      double omega = sph_triangle_area(
          sm.vx[a], sm.vy[a], sm.vz[a],
          sm.vx[b], sm.vy[b], sm.vz[b],
          sm.vx[c], sm.vy[c], sm.vz[c]);
      face_area[fi] = r * r * omega;
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

  // Lumped vertex dual volumes.  Prism (t, k) subtends the solid angle
  // omega of triangle t; its cross-section at radius s is omega*s^2, so
  //   V           = omega (b^3 - a^3)/3,
  //   int zeta dV = (omega/dr) [ (b^4 - a^4)/4 - a (b^3 - a^3)/3 ],
  // with a = r_k, b = r_k+1.  Each of the 3 bottom (top) vertices gets
  // a third of the (1 - zeta) (zeta) share — the lumped counterpart of
  // the deposit's hat weights.  Sum over vertices = total volume.
  vert_dual_vol.resize(m_N_verts);
  for (int i = 0; i < m_N_verts; i++) vert_dual_vol[i] = 0;
  for (int k = 0; k < m_N_r; k++) {
    double a = radii[k], b = radii[k + 1];
    double dr = b - a;
    for (int t = 0; t < m_N_tri; t++) {
      double omega = face_area[tri_face_idx(k, t)] / (a * a);
      double v_tot = omega * (b * b * b - a * a * a) / 3.0;
      double v_top = (omega / dr) *
          ((b * b * b * b - a * a * a * a) / 4.0 -
           a * (b * b * b - a * a * a) / 3.0);
      double v_bot = v_tot - v_top;
      for (int vi = 0; vi < 3; vi++) {
        int sv = sm.triangles[t][vi];
        vert_dual_vol[vert_idx(k, sv)] += v_bot / 3.0;
        vert_dual_vol[vert_idx(k + 1, sv)] += v_top / 3.0;
      }
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
  // Compute proper diagonal Hodge star from the (r, θ, φ) geometric dual.
  //
  // Each prism's dual vertex is at its spherical circumcenter:
  //   angular direction û_circ = unit normal to the chord triangle
  //                               (= spherical circumcenter of the triangle)
  //   radial position r_mid    = midpoint of the radial interval
  //
  // All dual distances and polygon areas are computed directly in spherical
  // terms — no Cartesian subtraction, no tangent-plane projection.
  //
  //   hodge2[f]    = |f*| / |f|
  //   hodge1_inv[e] = |e| / |e*|

  // --- Step 1: circumcenter angular directions (unit vectors) per sphere
  // triangle and radial midpoints per layer ---
  std::vector<double> circ_ux(m_N_tri), circ_uy(m_N_tri), circ_uz(m_N_tri);
  std::vector<double> r_mid_layer(m_N_r);
  for (int k = 0; k < m_N_r; k++) {
    r_mid_layer[k] = 0.5 * (radii[k] + radii[k + 1]);
  }

  for (int t = 0; t < m_N_tri; t++) {
    int a = sm.triangles[t][0];
    int b = sm.triangles[t][1];
    int c = sm.triangles[t][2];

    // Spherical circumcenter: unit normal to the chord plane of (a, b, c).
    // For three points on the unit sphere, this is exactly the point equi-
    // distant (in great-circle metric) from all three — n·a = n·b = n·c iff
    // n ⟂ (b−a) and n ⟂ (c−a).  For the icosahedral subdivision (spherically
    // Delaunay) this makes the dual the spherical Voronoi diagram, with the
    // diagonal Hodge star uniformly second-order accurate at every vertex.
    double e1x = sm.vx[b] - sm.vx[a];
    double e1y = sm.vy[b] - sm.vy[a];
    double e1z = sm.vz[b] - sm.vz[a];
    double e2x = sm.vx[c] - sm.vx[a];
    double e2y = sm.vy[c] - sm.vy[a];
    double e2z = sm.vz[c] - sm.vz[a];
    double nx = e1y * e2z - e1z * e2y;
    double ny = e1z * e2x - e1x * e2z;
    double nz = e1x * e2y - e1y * e2x;
    if (dual_centroid) {
      // Diagnostic centroidal dual (mesh_dual_centroid): use the spherical
      // centroid direction instead — loses edge ⟂ dual-face.
      nx = sm.vx[a] + sm.vx[b] + sm.vx[c];
      ny = sm.vy[a] + sm.vy[b] + sm.vy[c];
      nz = sm.vz[a] + sm.vz[b] + sm.vz[c];
    }
    double nlen = std::sqrt(nx * nx + ny * ny + nz * nz);
    double ux = 0.0, uy = 0.0, uz = 0.0;
    if (nlen > 0) {
      ux = nx / nlen;
      uy = ny / nlen;
      uz = nz / nlen;
      // Orient outward (same hemisphere as the triangle centroid).
      double mx = (sm.vx[a] + sm.vx[b] + sm.vx[c]) / 3.0;
      double my = (sm.vy[a] + sm.vy[b] + sm.vy[c]) / 3.0;
      double mz = (sm.vz[a] + sm.vz[b] + sm.vz[c]) / 3.0;
      if (ux * mx + uy * my + uz * mz < 0) {
        ux = -ux;
        uy = -uy;
        uz = -uz;
      }
    }
    circ_ux[t] = ux;
    circ_uy[t] = uy;
    circ_uz[t] = uz;
  }

  // Build sphere-edge -> (triangle_0, triangle_1) map.
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

  // --- Step 2: hodge2[f] = |f*| / |f| ---
  hodge2.resize(m_N_faces);

  // Triangular faces on shell k: the two adjacent prisms (layers k-1, k) share
  // the same circumcenter direction û_circ; their dual vertices differ only
  // in radius.  The dual edge length is therefore the pure radial span.
  for (int k = 0; k <= m_N_r; k++) {
    for (int t = 0; t < m_N_tri; t++) {
      int fi = tri_face_idx(k, t);
      double dist;
      if (k == 0) {
        // Inner boundary: only one prism above.  Use the TRUNCATED dual
        // edge (r_mid - r_shell), consistent with the half-trapezoid
        // convention for boundary h-edges below.  (The previous "x2 to
        // estimate the missing ghost side" made the boundary-shell
        // Hodge O(1)-inconsistent — measured as a 30x spurious-curl
        // excess on the exact static dipole, see roadmap A2 notes.)
        dist = r_mid_layer[0] - radii[0];
      } else if (k == m_N_r) {
        // Outer boundary: only one prism below.
        dist = radii[m_N_r] - r_mid_layer[m_N_r - 1];
      } else {
        dist = r_mid_layer[k] - r_mid_layer[k - 1];
      }
      hodge2[fi] = (face_area[fi] > 0) ? dist / face_area[fi] : 0;
    }
  }

  // Rectangular faces between shells: the two adjacent prisms at the same
  // layer k share r_mid but have different û_circ.  The dual edge is the
  // great-circle arc on the sphere of radius r_mid between them.
  for (int k = 0; k < m_N_r; k++) {
    double r_mid = r_mid_layer[k];
    for (int e = 0; e < m_N_edge_s; e++) {
      int fi = rect_face_idx(k, e);
      int t0 = edge_tris[e][0];
      int t1 = edge_tris[e][1];
      if (t0 == -1 || t1 == -1) {
        hodge2[fi] = 0;
        continue;
      }
      double ang = arc_angle(circ_ux[t0], circ_uy[t0], circ_uz[t0],
                             circ_ux[t1], circ_uy[t1], circ_uz[t1]);
      double dist = r_mid * ang;
      hodge2[fi] = (face_area[fi] > 0) ? dist / face_area[fi] : 0;
    }
  }

  // --- Step 3: hodge1_inv[e] = |e| / |e*| ---
  hodge1_inv.resize(m_N_edges);

  // Horizontal edges on shell k: the dual face is a ruled trapezoid with
  // corners (r_mid_below, û_0), (r_mid_below, û_1), (r_mid_above, û_1),
  // (r_mid_above, û_0) — same geometry as a primal rectangular face between
  // the two arcs at r_mid_below and r_mid_above along the great-circle arc
  // between û_0 and û_1 (the circumcenter directions of the two triangles
  // sharing the sphere edge).  Its area is
  //     Δr * 0.5 * (r_mid_below + r_mid_above) * arc_angle(û_0, û_1).
  // On boundary shells (k = 0 or k = N_r) the primal edge is shared by only
  // two prisms in one layer; the dual face degenerates to half of that
  // trapezoid, with Δr = |r_shell − r_mid|.
  for (int k = 0; k <= m_N_r; k++) {
    for (int e = 0; e < m_N_edge_s; e++) {
      int ei = h_edge_idx(k, e);
      int t0 = edge_tris[e][0];
      int t1 = edge_tris[e][1];
      double area = 0.0;
      if (t0 >= 0 && t1 >= 0) {
        double ang = arc_angle(circ_ux[t0], circ_uy[t0], circ_uz[t0],
                               circ_ux[t1], circ_uy[t1], circ_uz[t1]);
        double r_below, r_above, r_avg, dr_span;
        if (k == 0) {
          // Only layer 0 above; dual is half trapezoid between r_min and r_mid[0]
          r_below = radii[0];
          r_above = r_mid_layer[0];
          dr_span = r_above - r_below;
          r_avg = 0.5 * (r_below + r_above);
        } else if (k == m_N_r) {
          r_below = r_mid_layer[m_N_r - 1];
          r_above = radii[m_N_r];
          dr_span = r_above - r_below;
          r_avg = 0.5 * (r_below + r_above);
        } else {
          r_below = r_mid_layer[k - 1];
          r_above = r_mid_layer[k];
          dr_span = r_above - r_below;
          r_avg = 0.5 * (r_below + r_above);
        }
        area = dr_span * r_avg * ang;
      }
      hodge1_inv[ei] = (area > 0) ? edge_length[ei] / area : 0;
    }
  }

  // Vertical edges at sphere vertex s, layer k: the dual face is the
  // spherical polygon on the sphere of radius r_mid[k] whose corners are the
  // circumcenter directions of the 5–6 triangles meeting at vertex s.
  //     Area = r_mid² · Ω_polygon
  //     Ω_polygon = Σ Ω(û_s, û_i, û_{i+1})  (fan triangulation from û_s,
  //                                          which is inside every spherical
  //                                          Voronoi cell around vertex s).
  std::vector<std::vector<int>> vert_tris(m_N_vert_s);
  for (int t = 0; t < m_N_tri; t++) {
    for (int j = 0; j < 3; j++) {
      vert_tris[sm.triangles[t][j]].push_back(t);
    }
  }

  for (int k = 0; k < m_N_r; k++) {
    double r_mid = r_mid_layer[k];
    for (int s = 0; s < m_N_vert_s; s++) {
      int ei = v_edge_idx(k, s);
      auto& tris = vert_tris[s];
      int np = tris.size();  // 5 or 6

      // Sort the triangle circumcenters by azimuth around û_s in the local
      // tangent plane of the unit sphere.  This guarantees a simple polygon.
      double anchor_x = sm.vx[s], anchor_y = sm.vy[s], anchor_z = sm.vz[s];
      // Tangent basis at û_s.
      double tx, ty, tz;
      if (std::fabs(anchor_x) < 0.9) {
        tx = 0.0; ty = -anchor_z; tz = anchor_y;
      } else {
        tx = anchor_z; ty = 0.0; tz = -anchor_x;
      }
      double tn = std::sqrt(tx * tx + ty * ty + tz * tz);
      tx /= tn; ty /= tn; tz /= tn;
      double bx = anchor_y * tz - anchor_z * ty;
      double by = anchor_z * tx - anchor_x * tz;
      double bz = anchor_x * ty - anchor_y * tx;

      std::vector<double> angles(np);
      for (int i = 0; i < np; i++) {
        int t = tris[i];
        double dx = circ_ux[t] - anchor_x;
        double dy = circ_uy[t] - anchor_y;
        double dz = circ_uz[t] - anchor_z;
        double u_coord = dx * tx + dy * ty + dz * tz;
        double v_coord = dx * bx + dy * by + dz * bz;
        angles[i] = std::atan2(v_coord, u_coord);
      }
      std::vector<int> order(np);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&](int ia, int ib) { return angles[ia] < angles[ib]; });

      // Spherical polygon area by fan triangulation from û_s.
      double omega = 0.0;
      for (int i = 0; i < np; i++) {
        int j = (i + 1) % np;
        int t0 = tris[order[i]];
        int t1 = tris[order[j]];
        omega += sph_triangle_area(
            anchor_x, anchor_y, anchor_z,
            circ_ux[t0], circ_uy[t0], circ_uz[t0],
            circ_ux[t1], circ_uy[t1], circ_uz[t1]);
      }
      double area = r_mid * r_mid * omega;
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
  // Store unit sphere vertex positions (Cartesian + spherical-angular).
  sphere_vx.resize(m_N_vert_s);
  sphere_vy.resize(m_N_vert_s);
  sphere_vz.resize(m_N_vert_s);
  sphere_theta.resize(m_N_vert_s);
  sphere_phi.resize(m_N_vert_s);
  for (int s = 0; s < m_N_vert_s; s++) {
    sphere_vx[s] = sm.vx[s];
    sphere_vy[s] = sm.vy[s];
    sphere_vz[s] = sm.vz[s];
    double cth = sm.vz[s];
    if (cth > 1.0) cth = 1.0;
    if (cth < -1.0) cth = -1.0;
    sphere_theta[s] = std::acos(cth);
    sphere_phi[s] = std::atan2(sm.vy[s], sm.vx[s]);
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

prismatic_mesh_ptrs prismatic_mesh::host_ptrs() const {
  prismatic_mesh_ptrs p{};
  p.N_r = m_N_r;
  p.N_tri = m_N_tri;
  p.N_vert_s = m_N_vert_s;
  p.N_edge_s = m_N_edge_s;
  // Explicit narrow: the POD ptrs bundle serves the SINGLE-RANK global
  // path only (the full-3D arrays it points at cap well below 2^31).
  p.N_verts = int(m_N_verts);
  p.N_edges = int(m_N_edges);
  p.N_faces = int(m_N_faces);

  p.radii = radii.host_ptr();

  p.d1_row_ptr = d1_row_ptr.host_ptr();
  p.d1_col_idx = d1_col_idx.host_ptr();
  p.d1_val = d1_val.host_ptr();
  p.d1t_row_ptr = d1t_row_ptr.host_ptr();
  p.d1t_col_idx = d1t_col_idx.host_ptr();
  p.d1t_val = d1t_val.host_ptr();

  p.hodge1_inv = hodge1_inv.host_ptr();
  p.hodge2 = hodge2.host_ptr();

  p.edge_boundary = edge_boundary.host_ptr();
  p.face_boundary = face_boundary.host_ptr();
  p.edge_radial_layer = edge_radial_layer.host_ptr();
  p.face_radial_layer = face_radial_layer.host_ptr();

  p.vert_r = vert_r.host_ptr();
  p.vert_theta = vert_theta.host_ptr();
  p.vert_phi = vert_phi.host_ptr();
  p.face_area = face_area.host_ptr();
  p.vert_dual_vol = vert_dual_vol.host_ptr();
  p.edge_length = edge_length.host_ptr();
  p.edge_v0 = edge_v0.host_ptr();
  p.edge_v1 = edge_v1.host_ptr();
  p.tri_face_v0 = tri_face_v0.host_ptr();
  p.tri_face_v1 = tri_face_v1.host_ptr();
  p.tri_face_v2 = tri_face_v2.host_ptr();
  p.rect_face_v0 = rect_face_v0.host_ptr();
  p.rect_face_v1 = rect_face_v1.host_ptr();
  p.rect_face_v2 = rect_face_v2.host_ptr();
  p.rect_face_v3 = rect_face_v3.host_ptr();

  p.sphere_vx = sphere_vx.host_ptr();
  p.sphere_vy = sphere_vy.host_ptr();
  p.sphere_vz = sphere_vz.host_ptr();
  p.sphere_theta = sphere_theta.host_ptr();
  p.sphere_phi = sphere_phi.host_ptr();
  p.tri_verts = tri_verts.host_ptr();
  p.tri_edges_s = tri_edges_s.host_ptr();
  p.tri_edge_signs = tri_edge_signs.host_ptr();
  p.tri_neighbor = tri_neighbor.host_ptr();
  p.sphere_edge_v0 = sphere_edge_v0.host_ptr();
  p.sphere_edge_v1 = sphere_edge_v1.host_ptr();

  return p;
}

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
prismatic_mesh_ptrs prismatic_mesh::dev_ptrs() const {
  prismatic_mesh_ptrs p{};
  p.N_r = m_N_r;
  p.N_tri = m_N_tri;
  p.N_vert_s = m_N_vert_s;
  p.N_edge_s = m_N_edge_s;
  // Explicit narrow: the POD ptrs bundle serves the SINGLE-RANK global
  // path only (the full-3D arrays it points at cap well below 2^31).
  p.N_verts = int(m_N_verts);
  p.N_edges = int(m_N_edges);
  p.N_faces = int(m_N_faces);

  p.radii = radii.dev_ptr();

  p.d1_row_ptr = d1_row_ptr.dev_ptr();
  p.d1_col_idx = d1_col_idx.dev_ptr();
  p.d1_val = d1_val.dev_ptr();
  p.d1t_row_ptr = d1t_row_ptr.dev_ptr();
  p.d1t_col_idx = d1t_col_idx.dev_ptr();
  p.d1t_val = d1t_val.dev_ptr();

  p.hodge1_inv = hodge1_inv.dev_ptr();
  p.hodge2 = hodge2.dev_ptr();

  p.edge_boundary = edge_boundary.dev_ptr();
  p.face_boundary = face_boundary.dev_ptr();
  p.edge_radial_layer = edge_radial_layer.dev_ptr();
  p.face_radial_layer = face_radial_layer.dev_ptr();

  p.vert_r = vert_r.dev_ptr();
  p.vert_theta = vert_theta.dev_ptr();
  p.vert_phi = vert_phi.dev_ptr();
  p.face_area = face_area.dev_ptr();
  p.vert_dual_vol = vert_dual_vol.dev_ptr();
  p.edge_length = edge_length.dev_ptr();
  p.edge_v0 = edge_v0.dev_ptr();
  p.edge_v1 = edge_v1.dev_ptr();
  p.tri_face_v0 = tri_face_v0.dev_ptr();
  p.tri_face_v1 = tri_face_v1.dev_ptr();
  p.tri_face_v2 = tri_face_v2.dev_ptr();
  p.rect_face_v0 = rect_face_v0.dev_ptr();
  p.rect_face_v1 = rect_face_v1.dev_ptr();
  p.rect_face_v2 = rect_face_v2.dev_ptr();
  p.rect_face_v3 = rect_face_v3.dev_ptr();

  p.sphere_vx = sphere_vx.dev_ptr();
  p.sphere_vy = sphere_vy.dev_ptr();
  p.sphere_vz = sphere_vz.dev_ptr();
  p.sphere_theta = sphere_theta.dev_ptr();
  p.sphere_phi = sphere_phi.dev_ptr();
  p.tri_verts = tri_verts.dev_ptr();
  p.tri_edges_s = tri_edges_s.dev_ptr();
  p.tri_edge_signs = tri_edge_signs.dev_ptr();
  p.tri_neighbor = tri_neighbor.dev_ptr();
  p.sphere_edge_v0 = sphere_edge_v0.dev_ptr();
  p.sphere_edge_v1 = sphere_edge_v1.dev_ptr();

  return p;
}

void prismatic_mesh::copy_to_device() {
  // Sphere-only builds (7D) leave the 3D per-cochain buffers empty —
  // skip them.
  auto copy = [](auto& buf) {
    if (buf.size() > 0) buf.copy_to_device();
  };
  copy(radii);
  copy(d1_row_ptr); copy(d1_col_idx); copy(d1_val);
  copy(d1t_row_ptr); copy(d1t_col_idx); copy(d1t_val);
  copy(hodge1_inv); copy(hodge2);
  copy(edge_boundary); copy(face_boundary);
  copy(edge_radial_layer); copy(face_radial_layer);
  copy(vert_r); copy(vert_theta); copy(vert_phi);
  copy(face_area); copy(edge_length); copy(vert_dual_vol);
  copy(edge_v0); copy(edge_v1);
  copy(tri_face_v0); copy(tri_face_v1); copy(tri_face_v2);
  copy(rect_face_v0); copy(rect_face_v1); copy(rect_face_v2); copy(rect_face_v3);
  copy(sphere_vx); copy(sphere_vy); copy(sphere_vz);
  copy(sphere_theta); copy(sphere_phi);
  copy(tri_verts); copy(tri_edges_s); copy(tri_edge_signs); copy(tri_neighbor);
  copy(sphere_edge_v0); copy(sphere_edge_v1);
}
#endif

}  // namespace Aperture
