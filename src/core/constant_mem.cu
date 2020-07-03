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

#include "constant_mem.h"
#include "constant_mem_func.h"
#include "cuda_control.h"
#include "utils/morton.h"

namespace Aperture {

__constant__ uint32_t morton2dLUT_dev[256];
__constant__ uint32_t morton3dLUT_dev[256];
__constant__ Grid<1> dev_grid_1d;
__constant__ Grid<2> dev_grid_2d;
__constant__ Grid<3> dev_grid_3d;
__constant__ float dev_charges[max_ptc_types];
__constant__ float dev_masses[max_ptc_types];

__constant__ uint64_t dev_rank = 0;
__device__ uint32_t dev_ptc_id = 0;
__device__ uint32_t dev_ph_id = 0;

void
init_morton(const uint32_t m2dLUT[256], const uint32_t m3dLUT[256]) {
  const uint32_t* p_2d = &m2dLUT[0];
  CudaSafeCall(cudaMemcpyToSymbol(morton2dLUT_dev, (void*)p_2d,
                                  sizeof(morton2dLUT_dev)));
  const uint32_t* p_3d = &m3dLUT[0];
  CudaSafeCall(cudaMemcpyToSymbol(morton3dLUT_dev, (void*)p_3d,
                                  sizeof(morton3dLUT_dev)));
}

void
init_dev_rank(int rank) {
  uint64_t r = rank;
  r <<= 32;
  CudaSafeCall(cudaMemcpyToSymbol(dev_rank, (void*)&r, sizeof(uint64_t)));
}

template <>
void
init_dev_grid(const Grid<1>& grid) {
  CudaSafeCall(cudaMemcpyToSymbol(dev_grid_1d, &grid, sizeof(Grid<1>)));
}

template <>
void
init_dev_grid(const Grid<2>& grid) {
  CudaSafeCall(cudaMemcpyToSymbol(dev_grid_2d, &grid, sizeof(Grid<2>)));
}

template <>
void
init_dev_grid(const Grid<3>& grid) {
  CudaSafeCall(cudaMemcpyToSymbol(dev_grid_3d, &grid, sizeof(Grid<3>)));
}

void
init_dev_charge_mass(const float charge[max_ptc_types],
                     const float mass[max_ptc_types]) {
  CudaSafeCall(
      cudaMemcpyToSymbol(dev_charges, (void*)charge, sizeof(dev_charges)));
  CudaSafeCall(cudaMemcpyToSymbol(dev_masses, (void*)mass, sizeof(dev_masses)));
}

}  // namespace Aperture
