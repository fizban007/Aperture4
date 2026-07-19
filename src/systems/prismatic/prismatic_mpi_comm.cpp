#include "systems/prismatic/prismatic_mpi_comm.h"
#include "systems/prismatic/prismatic_partition.h"
#include <cstdio>
#include <stdexcept>

namespace Aperture {

prismatic_mpi_comm::~prismatic_mpi_comm() { release(); }

prismatic_mpi_comm::prismatic_mpi_comm(prismatic_mpi_comm&& other) noexcept
    : m_comm_angular(other.m_comm_angular),
      m_comm_radial(other.m_comm_radial),
      m_comm_world_l(other.m_comm_world_l),
      m_angular_rank(other.m_angular_rank),
      m_radial_rank(other.m_radial_rank),
      m_n_angular(other.m_n_angular),
      m_n_radial(other.m_n_radial),
      m_canonical(other.m_canonical),
      m_owns_comms(other.m_owns_comms) {
  other.m_comm_angular = MPI_COMM_NULL;
  other.m_comm_radial = MPI_COMM_NULL;
  other.m_comm_world_l = MPI_COMM_NULL;
  other.m_owns_comms = false;
}

prismatic_mpi_comm&
prismatic_mpi_comm::operator=(prismatic_mpi_comm&& other) noexcept {
  if (this != &other) {
    release();
    m_comm_angular = other.m_comm_angular;
    m_comm_radial = other.m_comm_radial;
    m_comm_world_l = other.m_comm_world_l;
    m_angular_rank = other.m_angular_rank;
    m_radial_rank = other.m_radial_rank;
    m_n_angular = other.m_n_angular;
    m_n_radial = other.m_n_radial;
    m_canonical = other.m_canonical;
    m_owns_comms = other.m_owns_comms;
    other.m_comm_angular = MPI_COMM_NULL;
    other.m_comm_radial = MPI_COMM_NULL;
    other.m_comm_world_l = MPI_COMM_NULL;
    other.m_owns_comms = false;
  }
  return *this;
}

void prismatic_mpi_comm::release() {
  if (!m_owns_comms) return;
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (finalized) {
    // Can't free after Finalize; leak the handles, they're freed by MPI.
    m_comm_angular = MPI_COMM_NULL;
    m_comm_radial = MPI_COMM_NULL;
    m_comm_world_l = MPI_COMM_NULL;
    m_owns_comms = false;
    return;
  }
  if (m_comm_angular != MPI_COMM_NULL) MPI_Comm_free(&m_comm_angular);
  if (m_comm_radial != MPI_COMM_NULL) MPI_Comm_free(&m_comm_radial);
  if (m_comm_world_l != MPI_COMM_NULL) MPI_Comm_free(&m_comm_world_l);
  m_owns_comms = false;
}

void prismatic_mpi_comm::ensure_mpi_initialized() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(nullptr, nullptr);
  }
}

prismatic_mpi_comm prismatic_mpi_comm::single_rank() {
  prismatic_mpi_comm out;
  // Keep MPI_COMM_NULL for sub-comms in this mode; is_single_rank()
  // branches in the backend handle this.
  return out;
}

prismatic_mpi_comm prismatic_mpi_comm::create(MPI_Comm world, int n_radial) {
  int world_size = 0, world_rank = 0;
  MPI_Comm_size(world, &world_size);
  MPI_Comm_rank(world, &world_rank);

  if (n_radial < 1) {
    throw std::runtime_error("prismatic_mpi_comm::create: n_radial < 1");
  }
  if (world_size != 20 * n_radial) {
    throw std::runtime_error(
        "prismatic_mpi_comm::create: world_size must equal 20 * n_radial");
  }

  prismatic_mpi_comm out;
  out.m_n_angular = 20;
  out.m_n_radial = n_radial;
  out.m_radial_rank = world_rank / 20;
  out.m_angular_rank = world_rank % 20;

  // ---- comm_angular: split on radial_rank, attach Dist_graph topology ----
  MPI_Comm flat_angular = MPI_COMM_NULL;
  MPI_Comm_split(world, out.m_radial_rank, out.m_angular_rank, &flat_angular);

  // 9 neighbors: 3 edge (ico-edge-sharing) + 6 diagonal (valence-5 corner).
  const auto& en = prismatic_partition::edge_neighbors();
  const auto& dn = prismatic_partition::diagonal_neighbors();
  int neighbors[9];
  int w = 0;
  for (int g : en[out.m_angular_rank]) neighbors[w++] = g;
  for (int g : dn[out.m_angular_rank]) neighbors[w++] = g;

  // Undirected: sources == destinations.  reorder=0 so rank numbering
  // matches the key from the Comm_split above.
  int reorder = 0;
  MPI_Dist_graph_create_adjacent(flat_angular, 9, neighbors, MPI_UNWEIGHTED,
                                 9, neighbors, MPI_UNWEIGHTED, MPI_INFO_NULL,
                                 reorder, &out.m_comm_angular);
  MPI_Comm_free(&flat_angular);

  // ---- comm_radial: split on angular_rank, attach Cart topology ----
  MPI_Comm flat_radial = MPI_COMM_NULL;
  MPI_Comm_split(world, out.m_angular_rank, out.m_radial_rank, &flat_radial);

  int dims[1] = {n_radial};
  int periods[1] = {0};
  MPI_Cart_create(flat_radial, 1, dims, periods, 0, &out.m_comm_radial);
  MPI_Comm_free(&flat_radial);

  MPI_Comm_dup(world, &out.m_comm_world_l);
  out.m_owns_comms = true;
  return out;
}

bool prismatic_mpi_comm::node_tile_shape(int A, int K, int ranks_per_node,
                                         int& a_t, int& k_t) {
  a_t = k_t = 1;
  if (ranks_per_node <= 1) return false;
  if ((long(A) * K) % ranks_per_node != 0) return false;
  int best_a = 0, best_k = 0;
  for (int a = 1; a <= ranks_per_node; ++a) {
    if (ranks_per_node % a != 0) continue;
    const int k = ranks_per_node / a;
    if (A % a != 0 || K % k != 0) continue;
    // Squarest tile (minimal halo perimeter per node); angular-major
    // tie-break.
    const int cur = best_a < best_k ? best_a : best_k;
    const int cand = a < k ? a : k;
    if (best_a == 0 || cand > cur || (cand == cur && a > best_a)) {
      best_a = a;
      best_k = k;
    }
  }
  if (best_a == 0) return false;
  a_t = best_a;
  k_t = best_k;
  return true;
}

void prismatic_mpi_comm::node_tile_coords(int A, int a_t, int k_t, int w,
                                          int& ang, int& rad) {
  const int rpn = a_t * k_t;
  const int node = w / rpn;
  const int i = w - node * rpn;
  const int tiles_per_row = A / a_t;
  const int tcol = node % tiles_per_row;
  const int trow = node / tiles_per_row;
  ang = tcol * a_t + i % a_t;
  rad = trow * k_t + i / a_t;
}

prismatic_mpi_comm prismatic_mpi_comm::create(MPI_Comm world, int n_angular,
                                              int n_radial,
                                              int ranks_per_node) {
  int world_size = 0, world_rank = 0;
  MPI_Comm_size(world, &world_size);
  MPI_Comm_rank(world, &world_rank);

  if (n_radial < 1) {
    throw std::runtime_error("prismatic_mpi_comm::create: n_radial < 1");
  }
  if (prismatic_partition::min_patch_level_for(n_angular) < 0) {
    throw std::runtime_error(
        "prismatic_mpi_comm::create: invalid angular rank count (need "
        "A = 2^j or 5*2^j with A | 20*4^m)");
  }
  if (world_size != n_angular * n_radial) {
    throw std::runtime_error(
        "prismatic_mpi_comm::create: world_size must equal A * K");
  }

  prismatic_mpi_comm out;
  out.m_n_angular = n_angular;
  out.m_n_radial = n_radial;
  out.m_canonical = true;

  // Logical (ang, rad) of THIS actual rank: the identity assignment
  // unless node tiling applies (7E — see header).
  int a_t = 1, k_t = 1;
  if (node_tile_shape(n_angular, n_radial, ranks_per_node, a_t, k_t)) {
    node_tile_coords(n_angular, a_t, k_t, world_rank, out.m_angular_rank,
                     out.m_radial_rank);
    if (world_rank == 0) {
      // Loud once: co-noded ranks form a_t x k_t patches of the grid.
      std::fprintf(stderr,
                   "prismatic_mpi_comm: node tiling %dx%d (angular x "
                   "radial) per %d-rank block\n",
                   a_t, k_t, ranks_per_node);
    }
  } else {
    if (ranks_per_node > 1 && world_rank == 0) {
      std::fprintf(stderr,
                   "prismatic_mpi_comm: no valid %d-rank node tile for "
                   "%dx%d — using the identity rank assignment\n",
                   ranks_per_node, n_angular, n_radial);
    }
    out.m_radial_rank = world_rank / n_angular;
    out.m_angular_rank = world_rank % n_angular;
  }

  // Plain split for the angular sub-comm (no Dist_graph decoration —
  // see header).  Rank numbering follows the split key = angular rank.
  MPI_Comm_split(world, out.m_radial_rank, out.m_angular_rank,
                 &out.m_comm_angular);

  // Radial sub-comm with 1D Cartesian topology, as in the legacy path.
  MPI_Comm flat_radial = MPI_COMM_NULL;
  MPI_Comm_split(world, out.m_angular_rank, out.m_radial_rank, &flat_radial);
  int dims[1] = {n_radial};
  int periods[1] = {0};
  MPI_Cart_create(flat_radial, 1, dims, periods, 0, &out.m_comm_radial);
  MPI_Comm_free(&flat_radial);

  // Logical-order world comm: rank r in it == logical world rank
  // rad*A + ang (== the parent world rank without tiling).  Migration
  // and any other logical-rank-addressed collectives use this.
  MPI_Comm_split(world, 0,
                 out.m_radial_rank * n_angular + out.m_angular_rank,
                 &out.m_comm_world_l);

  out.m_owns_comms = true;
  return out;
}

}  // namespace Aperture
