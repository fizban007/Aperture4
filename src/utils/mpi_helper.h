#ifndef _MPI_HELPER_H_
#define _MPI_HELPER_H_

#include <mpi.h>

namespace Aperture {

namespace MPI_Helper {

template <typename T>
MPI_Datatype get_mpi_datatype(const T& x);

void handle_mpi_error(int error_code, int rank);

}  // namespace MPI_Helper

}  // namespace Aperture

#endif  // _MPI_HELPER_H_
