#ifndef __NDPTR_H_
#define __NDPTR_H_

#include "core/cuda_control.h"
#include "utils/range.hpp"
#include "utils/index.hpp"

namespace Aperture {

template <int Rank>
using default_index_t = idx_col_major_t<Rank>;

/// An thin wrapper around a naked pointer, purely to facilitate device access
/// to the underlying memory. Since one can't pass a multi_array directly to a
/// cuda kernel, this is the next best thing.
template <class T, int Rank, class Index_t = default_index_t<Rank>>
struct ndptr {
  typedef Index_t idx_type;

  T* p = nullptr;

  HOST_DEVICE ndptr(T* p_) : p(p_) {}

  HD_INLINE T& operator[](size_t idx) { return p[idx]; }
  HD_INLINE T& operator[](const Index_t& idx) { return p[idx.key]; }

  HD_INLINE idx_type idx_at(uint32_t idx, const extent_t<Rank>& ext) const {
    return Index_t(idx, ext);
  }

  HD_INLINE range_proxy<Index_t> indices(const extent_t<Rank>& ext) const {
    return range(Index_t(0, ext), Index_t(ext.size(), ext));
  }
};

template <class T, int Rank, class Index_t = default_index_t<Rank>>
struct ndptr_const {
  typedef Index_t idx_type;

  const T* p = nullptr;

  HOST_DEVICE ndptr_const(const T* p_) : p(p_) {}

  // Cannot use this operator to change the underlying data
  HD_INLINE T operator[](size_t idx) const { return p[idx]; }
  HD_INLINE T operator[](const Index_t& idx) const {
    return p[idx.key];
  }

  HD_INLINE idx_type idx_at(uint32_t idx, const extent_t<Rank>& ext) const {
    return Index_t(idx, ext);
  }

  HD_INLINE range_proxy<Index_t> indices(const extent_t<Rank>& ext) const {
    return range(Index_t(0, ext), Index_t(ext.size(), ext));
  }
};

}  // namespace Aperture


#endif // __NDPTR_H_
