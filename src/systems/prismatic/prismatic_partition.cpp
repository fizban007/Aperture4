#include "systems/prismatic/prismatic_partition.h"
#include "systems/prismatic/icosphere_topology.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace Aperture {

int prismatic_partition::owner_unit_of_sphere_edge(int sphere_edge_idx) const {
  if (m_topology == nullptr) return -1;
  return std::min(unit_of_tri(m_topology->edge_tri_a(sphere_edge_idx)),
                  unit_of_tri(m_topology->edge_tri_b(sphere_edge_idx)));
}

int prismatic_partition::owner_unit_of_sphere_vertex(
    int sphere_vertex_idx) const {
  if (m_topology == nullptr) return -1;
  const int* tris = m_topology->vertex_tris(sphere_vertex_idx);
  const int n = m_topology->vertex_tri_count(sphere_vertex_idx);
  int u = unit_of_tri(tris[0]);
  for (int j = 1; j < n; ++j) u = std::min(u, unit_of_tri(tris[j]));
  return u;
}

bool prismatic_partition::owns_sphere_edge(int sphere_edge_idx) const {
  if (m_topology != nullptr) {
    return owns_unit(owner_unit_of_sphere_edge(sphere_edge_idx));
  }
  return owns_all_angular();
}

bool prismatic_partition::owns_sphere_vertex(int sphere_vertex_idx) const {
  if (m_topology != nullptr) {
    return owns_unit(owner_unit_of_sphere_vertex(sphere_vertex_idx));
  }
  return owns_all_angular();
}

int prismatic_partition::min_patch_level_for(int A) {
  if (A <= 0) return -1;
  int a = A;
  if (a % 5 == 0) a /= 5;
  if ((a & (a - 1)) != 0) return -1;  // not of the form 2^j or 5·2^j
  for (int m = 0; m <= 15; ++m) {
    const long U = 20L << (2 * m);
    if (U % A == 0) return m;
  }
  return -1;
}

int prismatic_partition::suggest_angular_ranks(int world_size, int L,
                                               int N_r, bool pic) {
  if (world_size < 1) return 0;
  int best = 0;
  for (int A = 1; A <= world_size; ++A) {
    if (world_size % A != 0) continue;
    const int m = min_patch_level_for(A);
    if (m < 0 || m > L) continue;
    const int K = world_size / A;
    if (K > N_r) continue;
    if (pic && K > 1 && N_r / K < 2) continue;
    if (A > best) best = A;
  }
  return best;
}

void prismatic_partition::set_angular_units(int A, int angular_rank_in,
                                            int m) {
  if (m < 0) m = min_patch_level_for(A);
  if (m < 0 || m > L) {
    throw std::invalid_argument(
        "prismatic_partition: no valid patch level for A (need A = 2^j or "
        "5*2^j with A | 20*4^m, m <= L)");
  }
  const long U = 20L << (2 * m);
  if (U % A != 0) {
    throw std::invalid_argument(
        "prismatic_partition: A does not divide 20*4^m at the given patch "
        "level");
  }
  if (angular_rank_in < 0 || angular_rank_in >= A) {
    throw std::invalid_argument("prismatic_partition: angular rank out of "
                                "range");
  }
  patch_level = m;
  const int per_rank = static_cast<int>(U / A);
  unit_lo = angular_rank_in * per_rank;
  unit_hi = unit_lo + per_rank;
  angular_rank = angular_rank_in;
  n_angular_ranks = A;
  canonical_rank_order = true;

  // Sync the legacy whole-face view.
  if (owns_all_angular()) {
    ico_face_lo = 0;
    ico_face_hi = 20;
  } else if (per_rank == units_per_face() &&
             unit_lo % units_per_face() == 0) {
    // Exactly one whole ico-face: recover it from the path position.
    const int f = face_path()[unit_lo / units_per_face()];
    ico_face_lo = f;
    ico_face_hi = f + 1;
  } else {
    ico_face_lo = ico_face_hi = -1;
  }
}

prismatic_partition prismatic_partition::angular_units(int L, int N_r_global,
                                                       int A,
                                                       int angular_rank,
                                                       int m) {
  auto p = single_rank(L, N_r_global);
  p.set_angular_units(A, angular_rank, m);
  return p;
}

prismatic_partition prismatic_partition::combined(int L, int N_r_global,
                                                  int A, int K,
                                                  int world_rank, int m) {
  auto p = radial_slab(L, N_r_global, K, world_rank / A);
  p.set_angular_units(A, world_rank % A, m);
  return p;
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

// Hamiltonian cycle on the icosahedron face-adjacency graph (the
// dodecahedral graph, which is Hamiltonian).  kFacePath[p] = ico-face at
// path position p; consecutive entries share an ico-edge CYCLICALLY
// (entry 19 is adjacent to entry 0), so any contiguous run of whole
// faces in path order is a connected band.  Chosen as the
// lexicographically smallest cycle from face 0 under the kIcoFaces
// ordering above; the first five entries are the fan around base
// vertex 0.  A unit test verifies the cycle against edge_neighbors()
// and the inverse table.
constexpr std::array<int, 20> kFacePath = {
    0, 1, 2, 3, 4, 7, 16, 11, 10, 14, 13, 12, 17, 8, 18, 9, 19, 5, 15, 6};

// Inverse: kFacePathPos[f] = path position of ico-face f.
constexpr std::array<int, 20> kFacePathPos = {
    0, 1, 2, 3, 4, 17, 19, 5, 13, 15, 8, 7, 11, 10, 9, 18, 6, 12, 14, 16};

}  // namespace

const std::array<int, 20>& prismatic_partition::face_path() {
  static const auto table = kFacePath;
  return table;
}

const std::array<int, 20>& prismatic_partition::face_path_pos() {
  static const auto table = kFacePathPos;
  return table;
}

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
