#ifndef __INDEX_H_
#define __INDEX_H_

#include <cstdint>

namespace Aperture {

template <class T = uint32_t>
struct idx_flat_t {
  T key = 0;
  T inc_y = 0;
  T inc_z = 0;

  typedef idx_flat_t<T> self_type;

  inline idx_flat_t(T k) : key(k) {}
  inline idx_flat_t(T k, T dy) : key(k), inc_y(dy) {}
  inline idx_flat_t(T k, T dy, T dz) : key(k), inc_y(dy), inc_z(dz) {}

  inline idx_flat_t(const self_type& idx) {
    key = idx.key;
    inc_y = idx.inc_y;
    inc_z = idx.inc_z;
  }

  inline self_type incX(int n = 1) const {
    self_type result(*this);
    result.key += n;
    return result;
  }

  inline self_type incY(int n = 1) const {
    self_type result(*this);
    result.key += n * inc_y;
    return result;
  }

  inline self_type incZ(int n = 1) const {
    self_type result(*this);
    result.key += n * inc_z;
    return result;
  }

  inline self_type decX(int n = 1) const {
    self_type result(*this);
    result.key -= n;
    return result;
  }

  inline self_type decY(int n = 1) const {
    self_type result(*this);
    result.key -= n * inc_y;
    return result;
  }

  inline self_type decZ(int n = 1) const {
    self_type result(*this);
    result.key -= n * inc_z;
    return result;
  }
};

template <class T = uint32_t>
struct idx_col_major_t : public idx_flat_t<T> {
  typedef idx_flat_t<T> base_type;
  typedef idx_col_major_t<T> self_type;

  inline explicit idx_col_major_t(T k) : base_type(k) {}

  inline idx_col_major_t(T x, T y, T dim0)
      : base_type(x + y * dim0, dim0) {}

  inline idx_col_major_t(T x, T y, T z, T dim0, T dim1)
      : base_type(x + (y + z * dim1) * dim0, dim0, dim0 * dim1) {}

  inline idx_col_major_t(const self_type& idx) : base_type(idx) {}

  inline void decode(T& x) const { x = this->key; }

  inline void decode(T& x, T& y) const {
    x = this->key % this->inc_y;
    y = this->key / this->inc_y;
  }

  inline void decode(T& x, T& y, T& z) const {
    x = this->key % this->inc_y;
    y = (this->key % this->inc_z) / this->inc_y;
    z = this->key / this->inc_z;
  }
};

template <class T = uint32_t>
struct idx_row_major_t : public idx_flat_t<T> {
  typedef idx_flat_t<T> base_type;
  typedef idx_row_major_t<T> self_type;

  inline explicit idx_row_major_t(T k) : base_type(k) {}

  inline idx_row_major_t(T y, T x, T dim0)
      : base_type(x + y * dim0, dim0) {}

  inline idx_row_major_t(T z, T y, T x, T dim0, T dim1)
      : base_type(x + (y + z * dim1) * dim0, dim0, dim0 * dim1) {}

  inline idx_row_major_t(const self_type& idx) : base_type(idx) {}

  inline void decode(T& x) const { x = this->key; }

  inline void decode(T& y, T& x) const {
    x = this->key % this->inc_y;
    y = this->key / this->inc_y;
  }

  inline void decode(T& z, T& y, T& x) const {
    x = this->key % this->inc_y;
    y = (this->key % this->inc_z) / this->inc_y;
    z = this->key / this->inc_z;
  }
};

}  // namespace Aperture

#endif  // __INDEX_H_
