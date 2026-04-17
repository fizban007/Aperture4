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
  static prismatic_mpi_comm create(MPI_Comm world, int n_radial_ranks);

  // Degenerate single-process mode.
  static prismatic_mpi_comm single_rank();

  // Helper: call MPI_Init if it hasn't been called yet.  Useful in unit
  // tests that want to exercise MPI infrastructure under a single
  // process without requiring a test-binary-level main().
  static void ensure_mpi_initialized();

  MPI_Comm angular() const { return m_comm_angular; }
  MPI_Comm radial() const { return m_comm_radial; }

  int angular_rank() const { return m_angular_rank; }
  int radial_rank() const { return m_radial_rank; }
  int n_angular_ranks() const { return m_n_angular; }
  int n_radial_ranks() const { return m_n_radial; }

  bool is_single_rank() const {
    return m_n_angular == 1 && m_n_radial == 1;
  }

 private:
  MPI_Comm m_comm_angular = MPI_COMM_NULL;
  MPI_Comm m_comm_radial = MPI_COMM_NULL;
  int m_angular_rank = 0;
  int m_radial_rank = 0;
  int m_n_angular = 1;
  int m_n_radial = 1;
  bool m_owns_comms = false;

  void release();
};

}  // namespace Aperture
