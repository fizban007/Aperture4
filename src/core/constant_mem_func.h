#ifndef __CONSTANT_MEM_FUNC_H_
#define __CONSTANT_MEM_FUNC_H_

#include <cstdint>

namespace Aperture {

struct sim_params;

void init_morton(const uint32_t m2dLUT[256], const uint32_t m3dLUT[256]);
void init_dev_params(const sim_params& params);

}

#endif
