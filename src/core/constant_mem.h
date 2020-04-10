#ifndef __CONSANT_MEM_H_
#define __CONSANT_MEM_H_

#include "core/params.h"
#include <cstdint>

namespace Aperture {

#if defined(__CUDACC__)

extern __constant__ uint32_t morton2dLUT_dev[256];
extern __constant__ uint32_t morton3dLUT_dev[256];
extern __constant__ sim_params_base dev_params;

#endif

}  // namespace Aperture

#endif
