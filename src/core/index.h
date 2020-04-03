#ifndef __INDEX_H_
#define __INDEX_H_

#include <cstdint>

namespace Aperture {

template <class T = uint32_t>
struct index_column_major_t {
  T key = 0;
  T inc_y = 0;
  T inc_z = 0;

  typedef index_column_major_t<T> self_type;

  inline explicit index_column_major_t(T k) : key(k) {}

  inline index_column_major_t(T x, T y, T dim0) {
    key = x + y * dim0;
    inc_y = dim0;
  }

  inline index_column_major_t(T x, T y, T z, T dim0, T dim1) {
    key = x + (y + z * dim1) * dim0;
    inc_y = dim0;
    inc_z = dim0 * dim1;
  }

  inline index_column_major_t(const self_type& idx) {
    key = idx.key;
    inc_y = idx.inc_y;
    inc_z = idx.inc_z;
  }

  inline void decode(T& x) const {
    x = key;
  }

  inline void decode(T& x, T& y) const {
    x = key % inc_y;
    y = key / inc_y;
  }

  inline void decode(T& x, T& y, T& z) const {
    x = key % inc_y;
    y = (key % inc_z) / inc_y;
    z = key / inc_z;
  }

  inline self_type incX() const {
    self_type result(*this);
    result.key += 1;
    return result;
  }

  inline self_type incY() const {
    self_type result(*this);
    result.key += inc_y;
    return result;
  }

  inline self_type incZ() const {
    self_type result(*this);
    result.key += inc_z;
    return result;
  }

  inline self_type decX() const {
    self_type result(*this);
    result.key += 1;
    return result;
  }

  inline self_type decY() const {
    self_type result(*this);
    result.key -= inc_y;
    return result;
  }

  inline self_type decZ() const {
    self_type result(*this);
    result.key -= inc_z;
    return result;
  }
};

template <class T = uint32_t>
struct index_row_major_t {
  T key;
  T inc_y = 0;
  T inc_z = 0;

  typedef index_row_major_t<T> self_type;

  inline explicit index_row_major_t(T k) : key(k) {}

  inline index_row_major_t(T y, T x, T dim0) {
    key = x + y * dim0;
    inc_y = dim0;
  }

  inline index_row_major_t(T z, T y, T x, T dim0, T dim1) {
    key = x + (y + z * dim1) * dim0;
    inc_y = dim0;
    inc_z = dim0 * dim1;
  }

  inline void decode(T& x) const {
    x = key;
  }

  inline void decode(T& y, T& x) const {
    x = key % inc_y;
    y = key / inc_y;
  }

  inline void decode(T& z, T& y, T& x) const {
    x = key % inc_y;
    y = (key % inc_z) / inc_y;
    z = key / inc_z;
  }

  inline self_type incX() const {
    self_type result(*this);
    result.key += 1;
    return result;
  }

  inline self_type incY() const {
    self_type result(*this);
    result.key += inc_y;
    return result;
  }

  inline self_type incZ() const {
    self_type result(*this);
    result.key += inc_z;
    return result;
  }

  inline self_type decX() const {
    self_type result(*this);
    result.key -= 1;
    return result;
  }

  inline self_type decY() const {
    self_type result(*this);
    result.key -= inc_y;
    return result;
  }

  inline self_type decZ() const {
    self_type result(*this);
    result.key -= inc_z;
    return result;
  }
};

}  // namespace Aperture

#endif  // __INDEX_H_
