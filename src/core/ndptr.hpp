#ifndef __NDPTR_H_
#define __NDPTR_H_

#include "core/cuda_control.h"
#include "utils/range.hpp"
#include "utils/index.hpp"

namespace Aperture {

/// An thin wrapper around a naked pointer, purely to facilitate device access
/// to the underlying memory. Since one can't pass a multi_array directly to a
/// cuda kernel, this is the next best thing.
template <class T, int Rank, class Idx_t = default_idx_t<Rank>>
struct ndptr {
  typedef Idx_t idx_t;
  typedef T value_t;

  T* p = nullptr;

  HOST_DEVICE ndptr() {}
  HOST_DEVICE ndptr(T* p_) : p(p_) {}
  HOST_DEVICE ndptr(const ndptr<T, Rank, Idx_t>& other) = default;

  HD_INLINE T& operator[](size_t idx) { return p[idx]; }
  HD_INLINE const T& operator[](size_t idx) const { return p[idx]; }
  HD_INLINE T& operator[](const Idx_t& idx) { return p[idx.linear]; }
  HD_INLINE const T& operator[](const Idx_t& idx) const { return p[idx.linear]; }

  HD_INLINE idx_t idx_at(uint32_t lin, const extent_t<Rank>& ext) const {
    return Idx_t(lin, ext);
  }

  HD_INLINE idx_t get_idx(const index_t<Rank>& pos, const extent_t<Rank>& ext) const {
    return Idx_t(pos, ext);
  }

  HD_INLINE range_proxy<Idx_t> indices(const extent_t<Rank>& ext) const {
    return range(Idx_t(0, ext), Idx_t(ext.size(), ext));
  }
};

template <class T, int Rank, class Idx_t = default_idx_t<Rank>>
struct ndptr_const {
  typedef Idx_t idx_t;
  typedef T value_t;

  const T* p = nullptr;

  HOST_DEVICE ndptr_const() {}
  HOST_DEVICE ndptr_const(const T* p_) : p(p_) {}
  HOST_DEVICE ndptr_const(const ndptr_const<T, Rank, Idx_t>& other) = default;

  // Cannot use this operator to change the underlying data
  HD_INLINE T operator[](size_t idx) const { return p[idx]; }
  HD_INLINE T operator[](const Idx_t& idx) const {
    return p[idx.linear];
  }

  HD_INLINE idx_t idx_at(uint32_t lin, const extent_t<Rank>& ext) const {
    return Idx_t(lin, ext);
  }

  HD_INLINE idx_t get_idx(const index_t<Rank>& pos, const extent_t<Rank>& ext) const {
    return Idx_t(pos, ext);
  }

  HD_INLINE range_proxy<Idx_t> indices(const extent_t<Rank>& ext) const {
    return range(Idx_t(0, ext), Idx_t(ext.size(), ext));
  }
};

template <class T, int Rank, class Idx_t>
HD_INLINE bool operator==(const ndptr<T, Rank, Idx_t>& ptr, std::nullptr_t) {
  return ptr.p == nullptr;
}

template <class T, int Rank, class Idx_t>
HD_INLINE bool operator==(const ndptr_const<T, Rank, Idx_t>& ptr, std::nullptr_t) {
  return ptr.p == nullptr;
}

template <class T, int Rank, class Idx_t>
HD_INLINE bool operator==(std::nullptr_t, const ndptr<T, Rank, Idx_t>& ptr) {
  return ptr.p == nullptr;
}

template <class T, int Rank, class Idx_t>
HD_INLINE bool operator==(std::nullptr_t, const ndptr_const<T, Rank, Idx_t>& ptr) {
  return ptr.p == nullptr;
}

template <class T, int Rank, class Idx_t>
HD_INLINE bool operator!=(const ndptr<T, Rank, Idx_t>& ptr, std::nullptr_t) {
  return ptr.p != nullptr;
}

template <class T, int Rank, class Idx_t>
HD_INLINE bool operator!=(const ndptr_const<T, Rank, Idx_t>& ptr, std::nullptr_t) {
  return ptr.p != nullptr;
}

template <class T, int Rank, class Idx_t>
HD_INLINE bool operator!=(std::nullptr_t, const ndptr<T, Rank, Idx_t>& ptr) {
  return ptr.p != nullptr;
}

template <class T, int Rank, class Idx_t>
HD_INLINE bool operator!=(std::nullptr_t, const ndptr_const<T, Rank, Idx_t>& ptr) {
  return ptr.p != nullptr;
}


}  // namespace Aperture


#endif // __NDPTR_H_
