#include "zorder.h"

namespace Aperture {

uint64_t coord_to_linear(uint32_t x) { return x; }

uint64_t coord_to_linear(uint32_t xPos, uint32_t yPos) {
  static const uint64_t MASKS[] = {0x5555555555555555, 0x3333333333333333,
                                   0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF};
  static const uint64_t SHIFTS[] = {1, 2, 4, 8, 16};

  uint64_t x = xPos; // Interleave lower 16 bits of x and y, so the bits of x
  uint64_t y = yPos; // are in the even positions and bits from y in the odd;

  x = (x | (x << SHIFTS[3])) & MASKS[3];
  x = (x | (x << SHIFTS[2])) & MASKS[2];
  x = (x | (x << SHIFTS[1])) & MASKS[1];
  x = (x | (x << SHIFTS[0])) & MASKS[0];

  y = (y | (y << SHIFTS[3])) & MASKS[3];
  y = (y | (y << SHIFTS[2])) & MASKS[2];
  y = (y | (y << SHIFTS[1])) & MASKS[1];
  y = (y | (y << SHIFTS[0])) & MASKS[0];

  const uint32_t result = x | (y << 1);
  return result;
}
} // namespace Aperture
