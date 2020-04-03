#ifndef __ZORDER_H_
#define __ZORDER_H_

#include <cstdint>

namespace Aperture {

uint64_t coord_to_linear(uint32_t x);
uint64_t coord_to_linear(uint32_t x, uint32_t y);
uint64_t coord_to_linear(uint32_t x, uint32_t y, uint32_t z);

}

#endif // __ZORDER_H_
