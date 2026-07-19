#pragma once

#include <mpi.h>

namespace Aperture {

// =========================================================================
// Prismatic-mesh MPI communicator set.
//
// For a 20·K-rank decomposition (20 angular ico-faces × K radial slabs),
// holds the two sub-communicators used by the halo exchange:
//
//   comm_angular — one per radial slab, groups the 20 angular ranks on
//     that slab.  Has Dist_graph topology with 9 neighbors per rank
//     (3 ico-edge + 6 vertex-diagonal), matching the ico-face adjacency.
//
//   comm_radial — one per angular ico-face, groups the K radial ranks
//     owning that ico-face.  Has 1D Cartesian topology (non-periodic).
//
// Rank mapping in world:
//   world_rank = radial_rank * 20 + angular_rank
//   (equivalently: radial_rank = world_rank / 20,
//                  angular_rank = world_rank % 20)
//
// In each sub-communicator, rank numbering respects the key passed to
// MPI_Comm_split: comm_angular ranks go 0..19 in angular order, and
// comm_radial ranks go 0..K-1 in radial order.  So peer_rank values
// produced by build_angular_halo_plan and build_radial_halo_plan
// (which use ico_face index and radial slab index respectively) are
// directly valid as MPI ranks in the corresponding sub-comm.
//
// The single_rank() factory returns an empty descriptor — no sub-comms,
// no topology, is_single_rank() == true.  Used for single-process
// runs; halo exchanges become no-ops in this mode.
// =========================================================================
class prismatic_mpi_comm {
 public:
  prismatic_mpi_comm() = default;
  ~prismatic_mpi_comm();

  prismatic_mpi_comm(const prismatic_mpi_comm&) = delete;
  prismatic_mpi_comm& operator=(const prismatic_mpi_comm&) = delete;

  prismatic_mpi_comm(prismatic_mpi_comm&& other) noexcept;
  prismatic_mpi_comm& operator=(prismatic_mpi_comm&& other) noexcept;

  // Construct from a world communicator (typically MPI_COMM_WORLD) and
  // the number of radial slabs K.  Requires world_size == 20 * K.
  // LEGACY identity rank→face wiring (angular rank == ico-face index,
  // matching build_angular_halo_plan and the Phase-6 particle stack);
  // removed in 7C along with that stack.
  static prismatic_mpi_comm create(MPI_Comm world, int n_radial_ranks);

  // Generalized A·K decomposition (Phase 7A.4): A angular ranks
  // (A | 20·4^m for some m ≤ L, validated) × K radial slabs, with
  //   LOGICAL world rank = radial_rank * A + angular_rank.
  // Angular ranks follow the canonical path-ordered unit assignment
  // (prismatic_partition::angular_units), and canonical_rank_order()
  // reports true so consumers pick the generic plan builder.  The
  // angular sub-comm carries no Dist_graph decoration: the halo
  // backend posts plain Isend/Irecv to plan peer ranks and never
  // queries the graph topology.
  //
  // Phase 7E — node tiling (cluster-agnostic rank placement):
  // `ranks_per_node` > 1 permutes the ACTUAL-world-rank → (ang, rad)
  // assignment so every contiguous block of ranks_per_node actual
  // ranks forms a compact a_t × k_t tile of the A × K grid (the
  // universal launcher default places consecutive ranks on a node, so
  // intra-node peers become geometric halo neighbors — no launcher
  // placement files, no cluster names in code).  The tile shape is
  // chosen automatically (a_t | A, k_t | K, a_t·k_t = ranks_per_node,
  // squarest tile wins, angular-major tie-break); an unsatisfiable
  // shape falls back to the identity assignment with a log note.
  // All LOGICAL-rank machinery (partitions, plans, world_rank(),
  // world()) is unaffected — only which physical process plays which
  // logical rank changes, so global outputs are unchanged.
  static prismatic_mpi_comm create(MPI_Comm world, int n_angular_ranks,
                                   int n_radial_ranks,
                                   int ranks_per_node = 0);

  // Tile-shape and coordinate helpers (pure functions; unit-tested).
  // node_tile_shape: picks (a_t, k_t); false when no valid shape.
  static bool node_tile_shape(int A, int K, int ranks_per_node, int& a_t,
                              int& k_t);
  // node_tile_coords: (ang, rad) of ACTUAL world rank w under the tile
  // layout.
  static void node_tile_coords(int A, int a_t, int k_t, int w, int& ang,
                               int& rad);

  // Degenerate single-process mode.
  static prismatic_mpi_comm single_rank();

  // Helper: call MPI_Init if it hasn't been called yet.  Useful in unit
  // tests that want to exercise MPI infrastructure under a single
  // process without requiring a test-binary-level main().
  static void ensure_mpi_initialized();

  MPI_Comm angular() const { return m_comm_angular; }
  MPI_Comm radial() const { return m_comm_radial; }

  // World-spanning communicator whose rank order is the LOGICAL order
  // (rad·A + ang) — identical to the parent world without node tiling,
  // a reordered dup under it.  Collectives that address ranks by the
  // logical world rank (particle migration's Alltoallv) MUST use this,
  // not MPI_COMM_WORLD.
  MPI_Comm world() const { return m_comm_world_l; }

  int angular_rank() const { return m_angular_rank; }
  int radial_rank() const { return m_radial_rank; }
  int n_angular_ranks() const { return m_n_angular; }
  int n_radial_ranks() const { return m_n_radial; }

  bool is_single_rank() const {
    return m_n_angular == 1 && m_n_radial == 1;
  }

  // True when built by the generalized create(world, A, K): angular
  // ranks are canonical path-ordered unit ranges and partitions must be
  // built with prismatic_partition::combined().  False for the legacy
  // 20-face identity factory (partitions via combined_ico_face()).
  bool canonical_rank_order() const { return m_canonical; }

  // This rank's index in world = radial_rank * A + angular_rank.
  int world_rank() const { return m_radial_rank * m_n_angular + m_angular_rank; }
  int world_size() const { return m_n_radial * m_n_angular; }

 private:
  MPI_Comm m_comm_angular = MPI_COMM_NULL;
  MPI_Comm m_comm_radial = MPI_COMM_NULL;
  MPI_Comm m_comm_world_l = MPI_COMM_NULL;
  int m_angular_rank = 0;
  int m_radial_rank = 0;
  int m_n_angular = 1;
  int m_n_radial = 1;
  bool m_canonical = false;
  bool m_owns_comms = false;

  void release();
};

}  // namespace Aperture
