#ifndef __DOMAIN_INFO_H_
#define __DOMAIN_INFO_H_

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
};

}  // namespace Aperture

#endif
