#include "core/params.h"
#include "utils/morton.h"
#include "constant_mem.h"
#include "constant_mem_func.h"
#include "cuda_control.h"

namespace Aperture {

__constant__ uint32_t morton2dLUT_dev[256];
__constant__ uint32_t morton3dLUT_dev[256];
__constant__ sim_params dev_params;
__constant__ Grid<1> dev_grid_1d;
__constant__ Grid<2> dev_grid_2d;
__constant__ Grid<3> dev_grid_3d;

void
init_morton(const uint32_t m2dLUT[256], const uint32_t m3dLUT[256]) {
  const uint32_t* p_2d = &m2dLUT[0];
  CudaSafeCall(cudaMemcpyToSymbol(morton2dLUT_dev, (void*)p_2d, sizeof(morton2dLUT_dev)));
  const uint32_t* p_3d = &m3dLUT[0];
  CudaSafeCall(cudaMemcpyToSymbol(morton3dLUT_dev, (void*)p_3d, sizeof(morton3dLUT_dev)));
}

void
init_dev_params(const sim_params& params) {
  const sim_params* p = &params;
  CudaSafeCall(cudaMemcpyToSymbol(dev_params, p, sizeof(sim_params)));
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

}
