#ifndef __INDEX_H_
#define __INDEX_H_

#include "core/cuda_control.h"
// #include "morton2d.h"
#include "utils/morton.h"
#include "vec.hpp"
#include <type_traits>

namespace Aperture {

template <class Derived, int Rank>
struct idx_base_t {
  uint64_t linear;
  // vec_t<uint32_t, Rank> pos;

  typedef idx_base_t<Derived, Rank> self_type;

  HD_INLINE Derived operator++(int) {
    Derived result = (Derived&)*this;
    ++linear;
    return result;
  }

  HD_INLINE Derived& operator++() {
    ++linear;
    return (Derived&)*this;
  }

  HD_INLINE Derived& operator+=(uint32_t x) {
    linear += x;
    return (Derived&)*this;
  }

  HD_INLINE bool operator==(const Derived& other) const {
    return linear == other.linear;
  }

  HD_INLINE bool operator>=(const Derived& other) const {
    return linear >= other.linear;
  }

  HD_INLINE bool operator<(const Derived& other) const {
    return linear < other.linear;
  }
};

template <int Rank>
struct idx_col_major_t
    : public idx_base_t<idx_col_major_t<Rank>, Rank> {
  vec_t<uint32_t, Rank> strides;

  typedef idx_base_t<idx_col_major_t<Rank>, Rank> base_type;
  typedef idx_col_major_t<Rank> self_type;

  HD_INLINE idx_col_major_t(const vec_t<uint32_t, Rank>& pos,
                            const vec_t<uint32_t, Rank>& extent) {
    strides = get_strides(extent);
    // this->pos = pos;
    this->linear = pos.dot(strides);
  }

  HD_INLINE idx_col_major_t(const uint64_t& linear,
                            const vec_t<uint32_t, Rank>& extent) {
    strides = get_strides(extent);
    // this->pos = get_pos(linear);
    this->linear = linear;
  }

  HD_INLINE vec_t<uint32_t, Rank> get_pos() const {
    return get_pos(this->linear);
  }

  HD_INLINE vec_t<uint32_t, Rank> get_pos(uint64_t linear) const {
    // Assume strides is already in place
    auto result = vec_t<uint32_t, Rank>{};
    auto n = linear;
#pragma unroll
    for (int i = Rank - 1; i >= 0; i--) {
      result[i] = n / strides[i];
      n = n % strides[i];
    }
    return result;
  }

  HD_INLINE vec_t<uint32_t, Rank> get_strides(
      const vec_t<uint32_t, Rank>& extent) {
    auto result = vec_t<uint32_t, Rank>{};

    result[0] = 1;
    if (Rank > 1)
      for (int n = 1; n < Rank; ++n)
        result[n] = result[n - 1] * extent[n - 1];

    return result;
  }

  template <int Dir>
      HD_INLINE std::enable_if_t <
      Dir<Rank, self_type> inc(int n = 1) const {
    // HD_INLINE self_type inc(int n = 1) const {
    auto result = *this;
    // result.pos[Dir] += n;
    result.linear += strides[Dir];
    return result;
  }

  template <int Dir>
      HD_INLINE std::enable_if_t <
      Dir<Rank, self_type> dec(int n = 1) const {
    // HD_INLINE self_type inc(int n = 1) const {
    auto result = *this;
    // result.pos[Dir] += n;
    result.linear -= strides[Dir];
    return result;
  }
};

template <int Rank>
struct idx_row_major_t
    : public idx_base_t<idx_row_major_t<Rank>, Rank> {
  vec_t<uint32_t, Rank> strides;

  typedef idx_base_t<idx_row_major_t<Rank>, Rank> base_type;
  typedef idx_row_major_t<Rank> self_type;

  HD_INLINE idx_row_major_t(const vec_t<uint32_t, Rank>& pos,
                            const vec_t<uint32_t, Rank>& extent) {
    strides = get_strides(extent);
    // this->pos = pos;
    this->linear = pos.dot(strides);
  }

  HD_INLINE idx_row_major_t(const uint64_t& linear,
                            const vec_t<uint32_t, Rank>& extent) {
    strides = get_strides(extent);
    // this->pos = get_pos(linear);
    this->linear = linear;
  }

  HD_INLINE vec_t<uint32_t, Rank> get_strides(
      const vec_t<uint32_t, Rank>& extent) {
    auto result = vec_t<uint32_t, Rank>{};

    if (Rank > 0) result[Rank - 1] = 1;
    if (Rank > 1)
      for (int n = Rank - 2; n >= 0; --n)
        result[n] = result[n + 1] * extent[n + 1];

    return result;
  }

  HD_INLINE vec_t<uint32_t, Rank> get_pos(uint64_t linear) const {
    // Assume strides is already in place
    auto result = vec_t<uint32_t, Rank>{};
    auto n = linear;
#pragma unroll
    for (int i = 0; i < Rank; i++) {
      result[i] = n / strides[i];
      n = n % strides[i];
    }
    return result;
  }

  HD_INLINE vec_t<uint32_t, Rank> get_pos() const {
    return get_pos(this->linear);
  }

  template <int Dir>
      HD_INLINE std::enable_if_t <
      Dir<Rank, self_type> inc(int n = 1) const {
    // HD_INLINE self_type inc(int n = 1) const {
    auto result = *this;
    // result.pos[Dir] += n;
    result.linear += strides[Dir];
    return result;
  }

  template <int Dir>
      HD_INLINE std::enable_if_t <
      Dir<Rank, self_type> dec(int n = 1) const {
    // HD_INLINE self_type inc(int n = 1) const {
    auto result = *this;
    // result.pos[Dir] += n;
    result.linear -= strides[Dir];
    return result;
  }
};

template <int Rank>
struct idx_zorder_t : public idx_base_t<idx_zorder_t<Rank>, Rank> {};

template <>
struct idx_zorder_t<2> : public idx_base_t<idx_zorder_t<2>, 2> {
  typedef idx_base_t<idx_zorder_t<2>, 2> base_type;
  typedef idx_zorder_t<2> self_type;

  HD_INLINE idx_zorder_t(const index_t<2>& pos,
                         const extent_t<2>& extent) {
    // this->pos = pos;
    this->linear = morton2d<uint32_t>(pos[0], pos[1]).key;
  }

  HD_INLINE idx_zorder_t(const uint64_t& linear,
                         const extent_t<2>& extent) {
    this->linear = linear;
    // this->pos = get_pos(linear);
  }

  HD_INLINE vec_t<uint32_t, 2> get_pos(uint64_t linear) const {
    auto result = vec_t<uint32_t, 2>{};
    uint64_t x, y;
    morton2(linear).decode(x, y);
    result[0] = x;
    result[1] = y;
    return result;
  }

  HD_INLINE vec_t<uint32_t, 2> get_pos() const {
    return get_pos(this->linear);
  }

  template <int Dir>
  HD_INLINE self_type inc(int n = 1) const;

  template <int Dir>
  HD_INLINE self_type dec(int n = 1) const;
};

template <>
HD_INLINE idx_zorder_t<2>
idx_zorder_t<2>::inc<0>(int n) const {
  auto result = *this;
  // result.pos[0] += n;
  auto m = morton2(result.linear).incX(n);
  result.linear = m.key;
  return result;
}

template <>
HD_INLINE idx_zorder_t<2>
idx_zorder_t<2>::inc<1>(int n) const {
  auto result = *this;
  // result.pos[1] += n;
  auto m = morton2(result.linear).incY(n);
  result.linear = m.key;
  return result;
}

template <>
struct idx_zorder_t<3> : public idx_base_t<idx_zorder_t<3>, 3> {
  typedef idx_base_t<idx_zorder_t<3>, 3> base_type;
  typedef idx_zorder_t<3> self_type;

  HD_INLINE idx_zorder_t(const index_t<3>& pos,
                         const extent_t<3>& extent) {
    // this->pos = pos;
    this->linear = morton3d<uint32_t>(pos[0], pos[1], pos[2]).key;
  }

  HD_INLINE idx_zorder_t(const uint64_t& linear,
                         const extent_t<3>& extent) {
    this->linear = linear;
    // this->pos = get_pos(linear);
  }

  HD_INLINE vec_t<uint32_t, 3> get_pos(uint64_t linear) const {
    auto result = vec_t<uint32_t, 3>{};
    uint64_t x, y, z;
    morton3(linear).decode(x, y, z);
    result[0] = x;
    result[1] = y;
    result[2] = z;
    return result;
  }

  HD_INLINE vec_t<uint32_t, 3> get_pos() const {
    return get_pos(this->linear);
  }

  template <int Dir>
  HD_INLINE self_type inc(int n = 1) const;

  template <int Dir>
  HD_INLINE self_type dec(int n = 1) const;
};

template <>
HD_INLINE idx_zorder_t<3>
idx_zorder_t<3>::inc<0>(int n) const {
  auto result = *this;
  // result.pos[0] += n;
  auto m = morton3(result.linear).incX(n);
  result.linear = m.key;
  return result;
}

template <>
HD_INLINE idx_zorder_t<3>
idx_zorder_t<3>::inc<1>(int n) const {
  auto result = *this;
  // result.pos[1] += n;
  auto m = morton3(result.linear).incY(n);
  result.linear = m.key;
  return result;
}

template <>
HD_INLINE idx_zorder_t<3>
idx_zorder_t<3>::inc<2>(int n) const {
  auto result = *this;
  // result.pos[2] += n;
  auto m = morton3(result.linear).incZ(n);
  result.linear = m.key;
  return result;
}

}  // namespace Aperture

#endif  // __INDEX_H_
