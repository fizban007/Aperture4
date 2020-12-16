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

#ifndef __CONSANT_MEM_H_
#define __CONSANT_MEM_H_

#include "core/enum_types.h"
#include "core/typedefs_and_constants.h"
#include <cstdint>

namespace Aperture {

template <int Dim, typename value_t>
struct Grid;

struct sim_params;

#if defined(__CUDACC__)

extern __constant__ uint32_t morton2dLUT_dev[256];
extern __constant__ uint32_t morton3dLUT_dev[256];
extern __constant__ Grid<1, float> dev_grid_1d_float;
extern __constant__ Grid<1, double> dev_grid_1d_double;
extern __constant__ Grid<2, float> dev_grid_2d_float;
extern __constant__ Grid<2, double> dev_grid_2d_double;
extern __constant__ Grid<3, float> dev_grid_3d_float;
extern __constant__ Grid<3, double> dev_grid_3d_double;
// extern __constant__ Grid<1> dev_grid_1d;
// extern __constant__ Grid<2> dev_grid_2d;
// extern __constant__ Grid<3> dev_grid_3d;
extern __constant__ float dev_charges[max_ptc_types];
extern __constant__ float dev_masses[max_ptc_types];

extern __constant__ uint64_t dev_rank;
extern __device__ uint32_t dev_ptc_id;
extern __device__ uint32_t dev_ph_id;

template <int Rank, typename value_t>
__device__ __forceinline__ const Grid<Rank, value_t>& dev_grid();

template <>
__device__ __forceinline__ const Grid<1, float>&
dev_grid<1, float>() {
  return dev_grid_1d_float;
}
template <>
__device__ __forceinline__ const Grid<2, float>&
dev_grid<2, float>() {
  return dev_grid_2d_float;
}
template <>
__device__ __forceinline__ const Grid<3, float>&
dev_grid<3, float>() {
  return dev_grid_3d_float;
}

template <>
__device__ __forceinline__ const Grid<1, double>&
dev_grid<1, double>() {
  return dev_grid_1d_double;
}
template <>
__device__ __forceinline__ const Grid<2, double>&
dev_grid<2, double>() {
  return dev_grid_2d_double;
}
template <>
__device__ __forceinline__ const Grid<3, double>&
dev_grid<3, double>() {
  return dev_grid_3d_double;
}

#endif

}  // namespace Aperture

#endif
