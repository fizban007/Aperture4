#pragma once

#include <vector>

namespace Aperture {

class prismatic_mesh;

// =========================================================================
// Icosphere topology summary.
//
// For a subdivided icosphere at level L, precomputes:
//   - sphere-vertex → incident ico-face set (valence 1, 2, or 5)
//   - sphere-edge   → incident ico-face set (valence 1 or 2)
//
// Consumers:
//   1. prismatic_partition — angular ownership queries under ico-face
//      decomposition (lowest-index incident ico-face owns the element).
//   2. Phase 3 angular halo plan builder — identify the sphere-edges
//      along each ico-edge boundary and the valence-5 corner vertices,
//      which determine the halo index sets.
//
// Built once per (L, mesh) pair.  Shared by all ranks running the same
// subdivision level — no MPI required.
//
// Subdivision invariant used in construction:
//   triangle t at level L belongs to ico-face (t / 4^L)
// because the subdivide() algorithm keeps the 4 children of triangle i
// at indices [4i, 4i+4) in the new list.
// =========================================================================
class icosphere_topology {
 public:
  icosphere_topology() = default;

  // Build from a prismatic_mesh that has been `build()`-ed.  Uses the
  // mesh's host-side `tri_verts` and `tri_edges_s` buffers.
  static icosphere_topology build_from_mesh(const prismatic_mesh& mesh);

  // Build from topology tables directly.  `tri_verts` is int[N_tri * 3]
  // giving the 3 sphere-vertex indices per triangle; `tri_edges_s` is
  // int[N_tri * 3] giving the 3 sphere-edge indices per triangle.
  // Triangles are assumed to be in subdivide-order: triangle t's ico-face
  // index is t / 4^L.
  static icosphere_topology build_from_tables(int L, int N_tri, int N_edge_s,
                                              int N_vert_s,
                                              const int* tri_verts,
                                              const int* tri_edges_s);

  int L() const { return m_L; }
  int N_tri() const { return m_N_tri; }
  int N_edge_s() const { return m_N_edge_s; }
  int N_vert_s() const { return m_N_vert_s; }

  // ---- Sphere-vertex queries ----
  int vertex_valence(int v) const {
    return m_vertex_offset[v + 1] - m_vertex_offset[v];
  }
  // Pointer to the valence incident ico-faces, sorted ascending.
  const int* vertex_ico_faces(int v) const {
    return m_vertex_ico_faces.data() + m_vertex_offset[v];
  }
  // Lowest-index incident ico-face — the owner under the partition's
  // angular ownership rule.
  int vertex_owner_ico_face(int v) const {
    return m_vertex_ico_faces[m_vertex_offset[v]];
  }

  // ---- Sphere-edge queries ----
  int edge_valence(int e) const {
    return m_edge_offset[e + 1] - m_edge_offset[e];
  }
  const int* edge_ico_faces(int e) const {
    return m_edge_ico_faces.data() + m_edge_offset[e];
  }
  int edge_owner_ico_face(int e) const {
    return m_edge_ico_faces[m_edge_offset[e]];
  }

  // ---- Adjacency: sphere-edge -> adjacent triangles ----
  // Every sphere-edge on a closed icosphere has exactly 2 adjacent
  // triangles.  Returned in arbitrary order (the topology struct does
  // not carry an orientation).
  int edge_tri_a(int e) const { return m_edge_tris[2 * e + 0]; }
  int edge_tri_b(int e) const { return m_edge_tris[2 * e + 1]; }

  // ---- Adjacency: sphere-vertex -> fan of adjacent triangles ----
  // Valence is 5 at the 12 icosahedron corners, 6 everywhere else.
  int vertex_tri_count(int v) const {
    return m_vertex_tri_offset[v + 1] - m_vertex_tri_offset[v];
  }
  const int* vertex_tris(int v) const {
    return m_vertex_tris.data() + m_vertex_tri_offset[v];
  }

  // ---- Sphere-edge endpoints ----
  // Each sphere-edge connects two sphere-vertices.  Endpoints are
  // stored in sorted order (v0 < v1) for convenience.
  int edge_v0(int e) const { return m_sphere_edge_v0[e]; }
  int edge_v1(int e) const { return m_sphere_edge_v1[e]; }

  // ---- Adjacency: sphere-vertex -> sphere-edges incident to it ----
  // Matches vertex_tri_count in valence (same # of incident edges as
  // triangles at a vertex on the sphere).
  int vertex_edge_count(int v) const {
    return m_vertex_edge_offset[v + 1] - m_vertex_edge_offset[v];
  }
  const int* vertex_edges(int v) const {
    return m_vertex_edges.data() + m_vertex_edge_offset[v];
  }

  // ---- Incident-unit queries (Phase 7A) ----
  // A level-m patch unit is a contiguous block of 4^(L−m) triangles
  // (see prismatic_partition): unit_of_tri(t) = t >> 2(L−m).  These
  // map an element's incident triangles through that arithmetic and
  // deduplicate.  Results are written to `out` sorted ascending; the
  // return value is the count of distinct units.  At m = 0 the results
  // equal the incident-ico-face lists (edge_ico_faces / vertex_ico_-
  // faces); the m = 0 CSR tables are retained — these wrappers do not
  // replace them.
  //
  // Capacity: an edge has ≤ 2 incident units (its 2 adjacent tris), a
  // vertex ≤ 6 (its fan).
  int edge_incident_units(int e, int patch_level, int out[2]) const {
    const int shift = 2 * (m_L - patch_level);
    int a = m_edge_tris[2 * e + 0] >> shift;
    int b = m_edge_tris[2 * e + 1] >> shift;
    if (a == b) {
      out[0] = a;
      return 1;
    }
    out[0] = a < b ? a : b;
    out[1] = a < b ? b : a;
    return 2;
  }

  int vertex_incident_units(int v, int patch_level, int out[6]) const {
    const int shift = 2 * (m_L - patch_level);
    const int* tris = vertex_tris(v);
    const int n = vertex_tri_count(v);
    int cnt = 0;
    for (int j = 0; j < n; ++j) {
      const int u = tris[j] >> shift;
      // Insertion into the short sorted output, skipping duplicates.
      int k = 0;
      while (k < cnt && out[k] < u) ++k;
      if (k < cnt && out[k] == u) continue;
      for (int w = cnt; w > k; --w) out[w] = out[w - 1];
      out[k] = u;
      ++cnt;
    }
    return cnt;
  }

 private:
  int m_L = 0;
  int m_N_tri = 0;
  int m_N_edge_s = 0;
  int m_N_vert_s = 0;

  // CSR-style incidence storage.  incidence[offset[v] .. offset[v+1])
  // contains the sorted ico-face indices incident to sphere-vertex v.
  std::vector<int> m_vertex_ico_faces;
  std::vector<int> m_vertex_offset;  // size N_vert_s + 1

  std::vector<int> m_edge_ico_faces;
  std::vector<int> m_edge_offset;    // size N_edge_s + 1

  // Edge -> {tri_a, tri_b}, one entry per sphere-edge (always 2 on a
  // closed icosphere).
  std::vector<int> m_edge_tris;           // size 2 * N_edge_s

  // Vertex -> list of adjacent triangles (valence 5 or 6).
  std::vector<int> m_vertex_tris;
  std::vector<int> m_vertex_tri_offset;   // size N_vert_s + 1

  // Sphere-edge endpoints (sphere-vertex indices), sorted so v0 < v1.
  std::vector<int> m_sphere_edge_v0;      // size N_edge_s
  std::vector<int> m_sphere_edge_v1;      // size N_edge_s

  // Vertex -> list of incident sphere-edges (same valence as vertex_tris).
  std::vector<int> m_vertex_edges;
  std::vector<int> m_vertex_edge_offset;  // size N_vert_s + 1
};

}  // namespace Aperture
