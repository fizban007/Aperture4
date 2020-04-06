#ifndef __MULTI_ARRAY_IMPL_H_
#define __MULTI_ARRAY_IMPL_H_

#include "multi_array.h"
#include <exception>
#include <type_traits>

namespace Aperture {

template <typename T, MemoryModel Model, typename Index_t>
multi_array<T, Model, Index_t>::multi_array() : m_ext(0, 0, 0) {}

template <typename T, MemoryModel Model, typename Index_t>
multi_array<T, Model, Index_t>::multi_array(uint32_t width,
                                            uint32_t height,
                                            uint32_t depth)
    : base_type(width * height * depth), m_ext(width, height, depth) {
  if constexpr (std::is_same_v<Index_t, idx_zorder<>>) {
    if (((width != 1) && (width & (width - 1))) ||
        ((height != 1) && (height & (height - 1))) ||
        ((depth != 1) && (depth & (depth - 1)))) {
      throw std::range_error(
          "One of the dimensions is not a power of 2, can't use zorder "
          "indexing!");
    }
  }
}

template <typename T, MemoryModel Model, typename Index_t>
multi_array<T, Model, Index_t>::multi_array(const Extent& ext)
    : multi_array(ext.x, ext.y, ext.z) {}

template <typename T, MemoryModel Model, typename Index_t>
multi_array<T, Model, Index_t>::~multi_array() {}

}  // namespace Aperture

#endif  // __MULTI_ARRAY_IMPL_H_
