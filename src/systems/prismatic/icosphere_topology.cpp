#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_mesh.h"
#include <set>

namespace Aperture {

namespace {

int pow4(int L) {
  int p = 1;
  for (int i = 0; i < L; ++i) p *= 4;
  return p;
}

}  // namespace

icosphere_topology icosphere_topology::build_from_tables(
    int L, int N_tri, int N_edge_s, int N_vert_s,
    const int* tri_verts, const int* tri_edges_s) {
  icosphere_topology out;
  out.m_L = L;
  out.m_N_tri = N_tri;
  out.m_N_edge_s = N_edge_s;
  out.m_N_vert_s = N_vert_s;

  const int sub_tris_per_ico_face = pow4(L);

  // Gather incidence using sets to auto-dedupe and auto-sort.
  std::vector<std::set<int>> vert_inc(N_vert_s);
  std::vector<std::set<int>> edge_inc(N_edge_s);

  for (int t = 0; t < N_tri; ++t) {
    const int ico_face = t / sub_tris_per_ico_face;
    for (int j = 0; j < 3; ++j) {
      int v = tri_verts[t * 3 + j];
      int e = tri_edges_s[t * 3 + j];
      vert_inc[v].insert(ico_face);
      edge_inc[e].insert(ico_face);
    }
  }

  // Pack CSR for vertices.
  out.m_vertex_offset.resize(N_vert_s + 1);
  int voff = 0;
  for (int v = 0; v < N_vert_s; ++v) {
    out.m_vertex_offset[v] = voff;
    voff += int(vert_inc[v].size());
  }
  out.m_vertex_offset[N_vert_s] = voff;
  out.m_vertex_ico_faces.reserve(voff);
  for (int v = 0; v < N_vert_s; ++v) {
    for (int f : vert_inc[v]) {  // std::set iterates ascending
      out.m_vertex_ico_faces.push_back(f);
    }
  }

  // Pack CSR for edges.
  out.m_edge_offset.resize(N_edge_s + 1);
  int eoff = 0;
  for (int e = 0; e < N_edge_s; ++e) {
    out.m_edge_offset[e] = eoff;
    eoff += int(edge_inc[e].size());
  }
  out.m_edge_offset[N_edge_s] = eoff;
  out.m_edge_ico_faces.reserve(eoff);
  for (int e = 0; e < N_edge_s; ++e) {
    for (int f : edge_inc[e]) {
      out.m_edge_ico_faces.push_back(f);
    }
  }

  // ---- Edge -> adjacent triangles (fixed 2 per edge) ----
  out.m_edge_tris.assign(2 * N_edge_s, -1);
  for (int t = 0; t < N_tri; ++t) {
    for (int j = 0; j < 3; ++j) {
      int e = tri_edges_s[t * 3 + j];
      int slot = (out.m_edge_tris[2 * e + 0] == -1) ? 0 : 1;
      out.m_edge_tris[2 * e + slot] = t;
    }
  }

  // ---- Vertex -> adjacent triangles (valence 5 or 6) ----
  std::vector<std::vector<int>> vert_tri_inc(N_vert_s);
  for (int t = 0; t < N_tri; ++t) {
    for (int j = 0; j < 3; ++j) {
      int v = tri_verts[t * 3 + j];
      vert_tri_inc[v].push_back(t);
    }
  }
  out.m_vertex_tri_offset.resize(N_vert_s + 1);
  int voff2 = 0;
  for (int v = 0; v < N_vert_s; ++v) {
    out.m_vertex_tri_offset[v] = voff2;
    voff2 += int(vert_tri_inc[v].size());
  }
  out.m_vertex_tri_offset[N_vert_s] = voff2;
  out.m_vertex_tris.reserve(voff2);
  for (int v = 0; v < N_vert_s; ++v) {
    for (int t : vert_tri_inc[v]) {
      out.m_vertex_tris.push_back(t);
    }
  }

  return out;
}

icosphere_topology icosphere_topology::build_from_mesh(
    const prismatic_mesh& mesh) {
  return build_from_tables(mesh.m_L, mesh.m_N_tri, mesh.m_N_edge_s,
                           mesh.m_N_vert_s, mesh.tri_verts.host_ptr(),
                           mesh.tri_edges_s.host_ptr());
}

}  // namespace Aperture
