#ifndef __CONSANT_MEM_H_
#define __CONSANT_MEM_H_

#include <cstdint>

namespace Aperture {

// #if defined(__CUDACC__)

extern __constant__ uint32_t morton2dLUT_dev[256];
extern __constant__ uint32_t morton3dLUT_dev[256];

// #endif

}  // namespace Aperture

#endif
