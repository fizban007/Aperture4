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
#include "core/gpu_translation_layer.h"
#include "core/gpu_error_check.h"
#include "utils/morton.h"

namespace Aperture {

__device__ __constant__ uint32_t morton2dLUT_dev[256];
__device__ __constant__ uint32_t morton3dLUT_dev[256];
__device__ __constant__ Grid<1, float> dev_grid_1d_float;
__device__ __constant__ Grid<1, double> dev_grid_1d_double;
__device__ __constant__ Grid<2, float> dev_grid_2d_float;
__device__ __constant__ Grid<2, double> dev_grid_2d_double;
__device__ __constant__ Grid<3, float> dev_grid_3d_float;
__device__ __constant__ Grid<3, double> dev_grid_3d_double;
__device__ __constant__ float dev_charges[max_ptc_types];
__device__ __constant__ float dev_masses[max_ptc_types];
__device__ __constant__ double dev_gauss_xs[5] = {
    0.1488743389816312, 0.4333953941292472, 0.6794095682990244,
    0.8650633666889845, 0.9739065285171717};

__device__ __constant__ double dev_gauss_ws[5] = {
    0.2955242247147529, 0.2692667193099963, 0.2190863625159821,
    0.1494513491505806, 0.0666713443086881};

__constant__ uint64_t dev_rank = 0;
__device__ uint32_t dev_ptc_id = 0;
__device__ uint32_t dev_ph_id = 0;

void
init_morton(const uint32_t m2dLUT[256], const uint32_t m3dLUT[256]) {
  const uint32_t* p_2d = &m2dLUT[0];
  GpuSafeCall(gpuMemcpyToSymbol(morton2dLUT_dev, (void*)p_2d,
                                sizeof(morton2dLUT_dev)));
  const uint32_t* p_3d = &m3dLUT[0];
  GpuSafeCall(gpuMemcpyToSymbol(morton3dLUT_dev, (void*)p_3d,
                                sizeof(morton3dLUT_dev)));
}

void
init_dev_rank(int rank) {
  uint64_t r = rank;
  r <<= 32;
  GpuSafeCall(gpuMemcpyToSymbol(dev_rank, (void*)&r, sizeof(uint64_t)));
}

template <>
void
init_dev_grid(const Grid<1, float>& grid) {
  GpuSafeCall(gpuMemcpyToSymbol(dev_grid_1d_float, &grid, sizeof(Grid<1, float>)));
}

template <>
void
init_dev_grid(const Grid<1, double>& grid) {
  GpuSafeCall(gpuMemcpyToSymbol(dev_grid_1d_double, &grid, sizeof(Grid<1, double>)));
}

template <>
void
init_dev_grid(const Grid<2, float>& grid) {
  GpuSafeCall(gpuMemcpyToSymbol(dev_grid_2d_float, &grid, sizeof(Grid<2, float>)));
}

template <>
void
init_dev_grid(const Grid<2, double>& grid) {
  GpuSafeCall(gpuMemcpyToSymbol(dev_grid_2d_double, &grid, sizeof(Grid<2, double>)));
}

template <>
void
init_dev_grid(const Grid<3, float>& grid) {
  GpuSafeCall(gpuMemcpyToSymbol(dev_grid_3d_float, &grid, sizeof(Grid<3, float>)));
}

template <>
void
init_dev_grid(const Grid<3, double>& grid) {
  GpuSafeCall(gpuMemcpyToSymbol(dev_grid_3d_double, &grid, sizeof(Grid<3, double>)));
}

// template <>
// void
// init_dev_grid(const Grid<2>& grid) {
//   GpuSafeCall(gpuMemcpyToSymbol(dev_grid_2d, &grid, sizeof(Grid<2>)));
// }

// template <>
// void
// init_dev_grid(const Grid<3>& grid) {
//   GpuSafeCall(gpuMemcpyToSymbol(dev_grid_3d, &grid, sizeof(Grid<3>)));
// }

void
init_dev_charge_mass(const float charge[max_ptc_types],
                     const float mass[max_ptc_types]) {
  GpuSafeCall(
      gpuMemcpyToSymbol(dev_charges, (void*)charge, sizeof(dev_charges)));
  GpuSafeCall(gpuMemcpyToSymbol(dev_masses, (void*)mass, sizeof(dev_masses)));
}

// namespace detail {

// template <int Rank, typename value_t>
// struct dev_grid_helper;

// template <>
// struct dev_grid_helper<1, float> {
//   using grid_type = Grid<1, float>;
//   static constexpr grid_type* dev_grid = &dev_grid_1d_float;
// };

// template <>
// struct dev_grid_helper<2, float> {
//   using grid_type = Grid<2, float>;
//   static constexpr grid_type* dev_grid = &dev_grid_2d_float;
// };

// template <>
// struct dev_grid_helper<3, float> {
//   using grid_type = Grid<3, float>;
//   static constexpr grid_type* dev_grid = &dev_grid_3d_float;
// };

// template <>
// struct dev_grid_helper<1, double> {
//   using grid_type = Grid<1, double>;
//   static constexpr grid_type* dev_grid = &dev_grid_1d_double;
// };

// template <>
// struct dev_grid_helper<2, double> {
//   using grid_type = Grid<2, double>;
//   static constexpr grid_type* dev_grid = &dev_grid_2d_double;
// };

// template <>
// struct dev_grid_helper<3, double> {
//   using grid_type = Grid<3, double>;
//   static constexpr grid_type* dev_grid = &dev_grid_3d_double;
// };

// }  // namespace detail

// template <int Rank, typename value_t>
// FORCE_INLINE __device__ const Grid<Rank, value_t>& dev_grid();
// __device__ __forceinline__ const Grid<Rank, value_t>& dev_grid() {
//   return *detail::dev_grid_helper<Rank, value_t>::dev_grid;
// }

template <>
__device__ const Grid<1, float>& dev_grid() {
  return dev_grid_1d_float;
}

template <>
__device__ const Grid<1, double>& dev_grid() {
  return dev_grid_1d_double;
}

template <>
__device__ const Grid<2, float>& dev_grid() {
  return dev_grid_2d_float;
}

template <>
__device__ const Grid<2, double>& dev_grid() {
  return dev_grid_2d_double;
}

template <>
__device__ const Grid<3, float>& dev_grid() {
  return dev_grid_3d_float;
}

template <>
__device__ const Grid<3, double>& dev_grid() {
  return dev_grid_3d_double;
}

}  // namespace Aperture
