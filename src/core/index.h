#ifndef __INDEX_H_
#define __INDEX_H_

#include <cstdint>
#include "cuda_control.h"
#include "morton2d.h"
#include "morton3d.h"

namespace Aperture {

// Here we define some index types that help us navigate a multi-dimensional
// array. Since all storage is linear, the indexing will basically determine the
// memory order.

template <class T = uint32_t>
struct idx_flat_t {
  T key = 0;
  T inc_y = 0;
  T inc_z = 0;

  typedef idx_flat_t<T> self_type;

  HD_INLINE idx_flat_t(T k) : key(k) {}
  HD_INLINE idx_flat_t(T k, T dy) : key(k), inc_y(dy) {}
  HD_INLINE idx_flat_t(T k, T dy, T dz) : key(k), inc_y(dy), inc_z(dz) {}

  HD_INLINE idx_flat_t(const self_type& idx) {
    key = idx.key;
    inc_y = idx.inc_y;
    inc_z = idx.inc_z;
  }

  HD_INLINE self_type incX(int n = 1) const {
    self_type result(*this);
    result.key += n;
    return result;
  }

  HD_INLINE self_type incY(int n = 1) const {
    self_type result(*this);
    result.key += n * inc_y;
    return result;
  }

  HD_INLINE self_type incZ(int n = 1) const {
    self_type result(*this);
    result.key += n * inc_z;
    return result;
  }

  HD_INLINE self_type decX(int n = 1) const {
    self_type result(*this);
    result.key -= n;
    return result;
  }

  HD_INLINE self_type decY(int n = 1) const {
    self_type result(*this);
    result.key -= n * inc_y;
    return result;
  }

  HD_INLINE self_type decZ(int n = 1) const {
    self_type result(*this);
    result.key -= n * inc_z;
    return result;
  }
};

template <class T = uint32_t>
struct idx_col_major_t : public idx_flat_t<T> {
  typedef idx_flat_t<T> base_type;
  typedef idx_col_major_t<T> self_type;

  HD_INLINE explicit idx_col_major_t(T k) : base_type(k) {}

  HD_INLINE idx_col_major_t(T x, T y, T dim0)
      : base_type(x + y * dim0, dim0) {}

  HD_INLINE idx_col_major_t(T x, T y, T z, T dim0, T dim1)
      : base_type(x + (y + z * dim1) * dim0, dim0, dim0 * dim1) {}

  HD_INLINE idx_col_major_t(const self_type& idx) : base_type(idx) {}

  HD_INLINE void decode(T& x) const { x = this->key; }

  HD_INLINE void decode(T& x, T& y) const {
    x = this->key % this->inc_y;
    y = this->key / this->inc_y;
  }

  HD_INLINE void decode(T& x, T& y, T& z) const {
    x = this->key % this->inc_y;
    y = (this->key % this->inc_z) / this->inc_y;
    z = this->key / this->inc_z;
  }
};

template <class T = uint32_t>
struct idx_row_major_t : public idx_flat_t<T> {
  typedef idx_flat_t<T> base_type;
  typedef idx_row_major_t<T> self_type;

  HD_INLINE explicit idx_row_major_t(T k) : base_type(k) {}

  HD_INLINE idx_row_major_t(T y, T x, T dim0)
      : base_type(x + y * dim0, dim0) {}

  HD_INLINE idx_row_major_t(T z, T y, T x, T dim0, T dim1)
      : base_type(x + (y + z * dim1) * dim0, dim0, dim0 * dim1) {}

  HD_INLINE idx_row_major_t(const self_type& idx) : base_type(idx) {}

  HD_INLINE void decode(T& x) const { x = this->key; }

  HD_INLINE void decode(T& y, T& x) const {
    x = this->key % this->inc_y;
    y = this->key / this->inc_y;
  }

  HD_INLINE void decode(T& z, T& y, T& x) const {
    x = this->key % this->inc_y;
    y = (this->key % this->inc_z) / this->inc_y;
    z = this->key / this->inc_z;
  }
};

}  // namespace Aperture

#endif  // __INDEX_H_
