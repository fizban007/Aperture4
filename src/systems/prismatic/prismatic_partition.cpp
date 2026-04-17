#include "systems/prismatic/prismatic_partition.h"
#include "systems/prismatic/icosphere_topology.h"
#include <array>

namespace Aperture {

bool prismatic_partition::owns_sphere_edge(int sphere_edge_idx) const {
  if (m_topology != nullptr) {
    return owns_ico_face(m_topology->edge_owner_ico_face(sphere_edge_idx));
  }
  return owns_all_angular();
}

bool prismatic_partition::owns_sphere_vertex(int sphere_vertex_idx) const {
  if (m_topology != nullptr) {
    return owns_ico_face(m_topology->vertex_owner_ico_face(sphere_vertex_idx));
  }
  return owns_all_angular();
}

namespace {

// The 20 triangles of the canonical icosahedron, vertex-indexed into the
// 12-vertex set.  Duplicated from prismatic_mesh::sphere_mesh::build_-
// icosahedron() so this file has no dependency on the private mesh impl.
// Must stay in sync with that function — if you change the face ordering
// there, update here too (a unit test verifies consistency).
constexpr std::array<std::array<int, 3>, 20> kIcoFaces = {{
    {0, 11, 5},  {0, 5, 1},   {0, 1, 7},    {0, 7, 10},   {0, 10, 11},
    {1, 5, 9},   {5, 11, 4},  {11, 10, 2},  {10, 7, 6},   {7, 1, 8},
    {3, 9, 4},   {3, 4, 2},   {3, 2, 6},    {3, 6, 8},    {3, 8, 9},
    {4, 9, 5},   {2, 4, 11},  {6, 2, 10},   {8, 6, 7},    {9, 8, 1},
}};

// Shared-vertex count between two ico-faces.
int shared_vertex_count(int f1, int f2) {
  int n = 0;
  for (int a : kIcoFaces[f1])
    for (int b : kIcoFaces[f2])
      if (a == b) ++n;
  return n;
}

// Build the edge-neighbor table: for each ico-face f, the 3 ico-faces that
// share an ico-edge (= 2 common vertices) with f.
std::array<std::array<int, 3>, 20> build_edge_neighbors() {
  std::array<std::array<int, 3>, 20> out{};
  for (int f = 0; f < 20; ++f) {
    int w = 0;
    for (int g = 0; g < 20; ++g) {
      if (g == f) continue;
      if (shared_vertex_count(f, g) == 2) {
        out[f][w++] = g;
      }
    }
    // Every triangle of the icosahedron has exactly 3 edge-neighbors; if
    // this fires, kIcoFaces is inconsistent.
    if (w != 3) {
      out[f][0] = out[f][1] = out[f][2] = -1;
    }
  }
  return out;
}

// Build the diagonal-neighbor table: for each ico-face f, the 6 ico-faces
// that share a single ico-vertex with f (vertex-only adjacency).
std::array<std::array<int, 6>, 20> build_diagonal_neighbors() {
  std::array<std::array<int, 6>, 20> out{};
  for (int f = 0; f < 20; ++f) {
    int w = 0;
    for (int g = 0; g < 20; ++g) {
      if (g == f) continue;
      if (shared_vertex_count(f, g) == 1) {
        if (w < 6) out[f][w] = g;
        ++w;
      }
    }
    // Pad with -1 if fewer than 6 (shouldn't happen on a valid icosahedron).
    while (w < 6) out[f][w++] = -1;
  }
  return out;
}

}  // namespace

const std::array<std::array<int, 3>, 20>&
prismatic_partition::edge_neighbors() {
  static const auto table = build_edge_neighbors();
  return table;
}

const std::array<std::array<int, 6>, 20>&
prismatic_partition::diagonal_neighbors() {
  static const auto table = build_diagonal_neighbors();
  return table;
}

}  // namespace Aperture
