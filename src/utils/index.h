#ifndef __INDEX_H_
#define __INDEX_H_

#include "core/cuda_control.h"
#include "morton2d.h"
#include "morton3d.h"
#include "vec3.h"
#include <cstdint>

namespace Aperture {

// Here we define some index types that help us navigate a
// multi-dimensional array. Since all storage is linear, the indexing
// will basically determine the memory order.

template <class Derived, class T = uint32_t>
struct idx_flat_t {
  T key = 0;
  T inc_y = 0;
  T inc_z = 0;

  typedef idx_flat_t<T> self_type;

  HD_INLINE idx_flat_t() {}
  HD_INLINE idx_flat_t(T k) : key(k) {}
  HD_INLINE idx_flat_t(T k, T dy) : key(k), inc_y(dy) {}
  HD_INLINE idx_flat_t(T k, T dy, T dz)
      : key(k), inc_y(dy), inc_z(dz) {}

  HD_INLINE idx_flat_t(const self_type& idx) {
    key = idx.key;
    inc_y = idx.inc_y;
    inc_z = idx.inc_z;
  }

  HD_INLINE Derived incX(int n = 1) const {
    Derived result(*this);
    result.key += n;
    return result;
  }

  HD_INLINE Derived incY(int n = 1) const {
    Derived result(*this);
    result.key += n * inc_y;
    return result;
  }

  HD_INLINE Derived incZ(int n = 1) const {
    Derived result(*this);
    result.key += n * inc_z;
    return result;
  }

  HD_INLINE Derived decX(int n = 1) const {
    Derived result(*this);
    result.key -= n;
    return result;
  }

  HD_INLINE Derived decY(int n = 1) const {
    Derived result(*this);
    result.key -= n * inc_y;
    return result;
  }

  HD_INLINE Derived decZ(int n = 1) const {
    Derived result(*this);
    result.key -= n * inc_z;
    return result;
  }
};

template <class T = uint32_t>
struct idx_col_major_t : public idx_flat_t<idx_col_major_t<T>, T> {
  typedef idx_flat_t<idx_col_major_t<T>, T> base_type;
  typedef idx_col_major_t<T> self_type;

  HD_INLINE idx_col_major_t(const base_type& other)
      : base_type(other) {}

  HD_INLINE idx_col_major_t(T x, const Extent& size)
      : base_type(x, size.x, size.y * size.x) {}

  HD_INLINE idx_col_major_t(T x, T y, const Extent& size)
      : base_type() {
    this->key = x + y * size.x;
    this->inc_y = size.x;
    this->inc_z = size.x * size.y;
  }

  HD_INLINE idx_col_major_t(T x, T y, T z, const Extent& size)
      : base_type() {
    this->key = x + (y + z * size.y) * size.x;
    this->inc_y = size.x;
    this->inc_z = size.x * size.y;
  }

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
struct idx_row_major_t : public idx_flat_t<idx_row_major_t<T>, T> {
  typedef idx_flat_t<idx_row_major_t<T>, T> base_type;
  typedef idx_row_major_t<T> self_type;

  HD_INLINE idx_row_major_t(const base_type& other)
      : base_type(other) {}

  HD_INLINE idx_row_major_t(T x, const Extent& size)
      : base_type(x, size.z, size.y * size.z) {}

  HD_INLINE idx_row_major_t(T y, T x, const Extent& size)
      : base_type() {
    this->key = x + y * size.z;
    this->inc_y = size.z;
    this->inc_z = size.z * size.y;
  }

  HD_INLINE idx_row_major_t(T z, T y, T x, const Extent& size)
      : base_type() {
    this->key = x + (y + z * size.y) * size.z;
    this->inc_y = size.z;
    this->inc_z = size.z * size.y;
  }

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

template <class T = uint32_t>
struct idx_zorder {
  uint64_t key = 0;
  int dim;

  typedef idx_zorder<T> self_type;

  HD_INLINE idx_zorder(uint64_t k, int d) : key(k), dim(d) {}

  HD_INLINE idx_zorder(T x, const Extent& ext) : key(x), dim(1) {}

  HD_INLINE idx_zorder(T x, T y, const Extent& ext) {
    key = morton2d<T>(x, y).key;
    dim = 2;
  }

  HD_INLINE idx_zorder(T x, T y, T z, const Extent& ext) {
    key = morton3d<T>(x, y, z).key;
    dim = 3;
  }

  HD_INLINE void decode(T& x) const { x = key; }

  HD_INLINE void decode(T& x, T& y) const {
    uint64_t a, b;
    assert(dim == 2);
    morton2d<T>(key).decode(a, b);
    x = a;
    y = b;
  }

  HD_INLINE void decode(T& x, T& y, T& z) const {
    uint64_t a, b, c;
    assert(dim == 3);
    morton3d<T>(key).decode(a, b, c);
    x = a;
    y = b;
    z = c;
  }

  HD_INLINE self_type incX() const {
    if (dim == 1) {
      return self_type(key + 1, dim);
    } else if (dim == 2) {
      return self_type(morton2d<T>(key).incX().key, dim);
    } else {
      return self_type(morton3d<T>(key).incX().key, dim);
    }
  }

  HD_INLINE self_type incY() const {
    if (dim == 1) {
      return *this;
    } else if (dim == 2) {
      return self_type(morton2d<T>(key).incY().key, dim);
    } else {
      return self_type(morton3d<T>(key).incY().key, dim);
    }
  }

  HD_INLINE self_type incZ() const {
    if (dim == 3) {
      return self_type(morton3d<T>(key).incZ().key, dim);
    } else {
      return *this;
    }
  }

  HD_INLINE self_type decX() const {
    if (dim == 1) {
      return self_type(key + 1, dim);
    } else if (dim == 2) {
      return self_type(morton2d<T>(key).decX().key, dim);
    } else {
      return self_type(morton3d<T>(key).decX().key, dim);
    }
  }

  HD_INLINE self_type decY() const {
    if (dim == 1) {
      return *this;
    } else if (dim == 2) {
      return self_type(morton2d<T>(key).decY().key, dim);
    } else {
      return self_type(morton3d<T>(key).decY().key, dim);
    }
  }

  HD_INLINE self_type decZ() const {
    if (dim == 3) {
      return self_type(morton3d<T>(key).decZ().key, dim);
    } else {
      return *this;
    }
  }
};

}  // namespace Aperture

#endif  // __INDEX_H_
