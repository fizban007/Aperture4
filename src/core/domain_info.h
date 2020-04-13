#ifndef __DOMAIN_INFO_H_
#define __DOMAIN_INFO_H_

#include <mpi.h>

namespace Aperture {

template <int Dim>
struct domain_info_t {
  int mpi_dims[Dim];  ///< Size of the domain decomposition in 3 directions
  int mpi_coord[Dim];  ///< The 3D MPI coordinate of this rank
  bool is_boundary[Dim * 2];  ///< Is this rank at boundary in each direction
  int is_periodic[Dim];  ///< Whether to use periodic boundary
                                 ///< conditions in each direction
  int neighbor_left[Dim];
  int neighbor_right[Dim];

  domain_info_t() {
    for (int i = 0; i < Dim; i++) {
      mpi_dims[i] = 1;
      mpi_coord[i] = 0;
      is_boundary[i * 2] = false;
      is_boundary[i * 2 + 1] = false;
      is_periodic[i] = 0;
      neighbor_left[i] = MPI_PROC_NULL;
      neighbor_right[i] = MPI_PROC_NULL;
    }
  }
};

}  // namespace Aperture

#endif
