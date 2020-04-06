#ifndef __NDPTR_H_
#define __NDPTR_H_

#include "cuda_control.h"
#include "index.h"

namespace Aperture {

/// An thin wrapper around a naked pointer, purely to facilitate device access
/// to the underlying memory. Since one can't pass a multi_array directly to a
/// cuda kernel, this is the next best thing.
template <class T, class Index_t = idx_col_major_t<>>
struct ndptr {
  typedef Index_t idx_type;

  T* p = nullptr;

  HOST_DEVICE ndptr(T* p_) : p(p_) {}

  HD_INLINE T& operator[](const Index_t& idx) { return p[idx.key]; }

  HD_INLINE idx_type idx_at(uint32_t idx, const Extent& ext) const {
    return Index_t(idx, ext);
  }
};

template <class T, class Index_t = idx_col_major_t<>>
struct ndptr_const {
  typedef Index_t idx_type;

  const T* p = nullptr;

  HOST_DEVICE ndptr_const(const T* p_) : p(p_) {}

  // Cannot use this operator to change the underlying data
  HD_INLINE T operator[](const Index_t& idx) const {
    return p[idx.key];
  }

  HD_INLINE idx_type idx_at(uint32_t idx, const Extent& ext) const {
    return Index_t(idx, ext);
  }
};

}  // namespace Aperture

#endif
