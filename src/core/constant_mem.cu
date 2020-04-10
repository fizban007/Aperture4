#include "core/params.h"
#include "utils/morton.h"
#include "constant_mem.h"
#include "constant_mem_func.h"
#include "cuda_control.h"

namespace Aperture {

__constant__ uint32_t morton2dLUT_dev[256];
__constant__ uint32_t morton3dLUT_dev[256];
__constant__ sim_params_base dev_params;

void
init_morton(const uint32_t m2dLUT[256], const uint32_t m3dLUT[256]) {
  const uint32_t* p_2d = &m2dLUT[0];
  CudaSafeCall(cudaMemcpyToSymbol(morton2dLUT_dev, (void*)p_2d, sizeof(morton2dLUT_dev)));
  const uint32_t* p_3d = &m3dLUT[0];
  CudaSafeCall(cudaMemcpyToSymbol(morton3dLUT_dev, (void*)p_3d, sizeof(morton3dLUT_dev)));
}

void
init_dev_params(const sim_params& params) {
  const sim_params_base* p = &params;
  CudaSafeCall(cudaMemcpyToSymbol(dev_params, p, sizeof(sim_params_base)));
}

}
