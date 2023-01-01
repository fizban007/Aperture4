/*
 * Copyright (c) 2020 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

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
