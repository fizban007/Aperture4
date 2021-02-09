/*
 * Copyright (c) 2020 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __INDEX_H_
#define __INDEX_H_

#include "core/cuda_control.h"
// #include "morton2d.h"
#include "utils/morton.h"
#include "vec.hpp"
#include <type_traits>

namespace Aperture {

template <int Rank>
struct idx_col_major_t;
template <int Rank>
struct idx_row_major_t;
template <int Rank>
struct idx_zorder_t;

template <int Rank>
using default_idx_t = idx_col_major_t<Rank>;

template <class Derived, int Rank>
struct idx_base_t {
  uint64_t linear;

  typedef idx_base_t<Derived, Rank> self_type;
  static constexpr int dim = Rank;

  HD_INLINE Derived operator++(int) {
    Derived result = (Derived&)*this;
    ++linear;
    return result;
  }

  HD_INLINE Derived& operator++() {
    ++linear;
    return (Derived&)*this;
  }

  HD_INLINE Derived operator+(int x) const {
    Derived result((Derived&)*this);
    result.linear += x;
    return result;
  }

  HD_INLINE Derived operator-(int x) const {
    Derived result((Derived&)*this);
    result.linear -= x;
    return result;
  }

  HD_INLINE int operator-(const Derived& x) const { return linear - x.linear; }

  HD_INLINE Derived operator+(uint32_t x) const {
    Derived result((Derived&)*this);
    result.linear += x;
    return result;
  }

  HD_INLINE Derived operator+(uint64_t x) const {
    Derived result((Derived&)*this);
    result.linear += x;
    return result;
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
struct idx_col_major_t : public idx_base_t<idx_col_major_t<Rank>, Rank> {
  // index_t<Rank> strides;
  const extent_t<Rank>& ext;

  typedef idx_base_t<idx_col_major_t<Rank>, Rank> base_type;
  typedef idx_col_major_t<Rank> self_type;

  using base_type::operator-;

  HOST_DEVICE idx_col_major_t(uint64_t n, const extent_t<Rank>& extent)
      : ext(extent) {
    this->linear = n;
  }

  HOST_DEVICE idx_col_major_t(const index_t<Rank>& pos,
                              const extent_t<Rank>& extent)
      : ext(extent) {
    this->linear = to_linear(pos);
  }

  HD_INLINE idx_col_major_t(const self_type& idx) = default;
  HD_INLINE self_type& operator=(const self_type& idx) = default;

  // HD_INLINE index_t<Rank> get_pos() const {
  inline index_t<Rank> get_pos() const { return pos(this->linear); }

  inline index_t<Rank> pos(uint64_t linear) const {
    auto result = index_t<Rank>{};
    auto n = linear;
    result[0] = n % this->ext[0];
#pragma unroll
    for (int i = 1; i < Rank; i++) {
      n /= this->ext[i - 1];
      result[i] = n % this->ext[i];
      // n /= this->ext[i];
    }
    return result;
  }

  HD_INLINE uint64_t to_linear(const index_t<Rank>& pos) const {
    //     uint64_t result = pos[0];
    //     int64_t stride = 1;
    // #pragma unroll
    //     for (int i = 1; i < Rank; i++) {
    //       stride *= this->ext[i - 1];
    //       result += pos[i] * stride;
    //     }
    // return result;
    return pos.dot(ext.strides());
  }

  template <int Dir>
  // HD_INLINE std::enable_if_t <
  // Dir<Rank, self_type> inc(int n = 1) const {
  HD_INLINE self_type inc(int n = 1) const {
    auto result = *this;
    // result.pos[Dir] += n;
    //     int64_t stride = 1;
    // #pragma unroll
    //     for (int i = 1; i < Dir + 1; i++) {
    //       stride *= this->ext[i - 1];
    //     }
    result.linear = (long)result.linear + n * ext.strides()[Dir];
    return result;
  }

  template <int Dir>
  // HD_INLINE std::enable_if_t <
  // Dir<Rank, self_type> dec(int n = 1) const {
  HD_INLINE self_type dec(int n = 1) const {
    auto result = *this;
    // result.pos[Dir] -= n;
    //     int64_t stride = 1;
    // #pragma unroll
    //     for (int i = 1; i < Dir + 1; i++) {
    //       stride *= this->ext[i - 1];
    //     }
    result.linear = (long)result.linear - n * ext.strides()[Dir];
    return result;
  }

  using base_type::operator+;

  HD_INLINE uint64_t operator+(const index_t<Rank>& pos) {
    return this->linear + pos.dot(ext.strides());
  }

  HD_INLINE uint64_t operator-(const index_t<Rank>& pos) {
    return this->linear - pos.dot(ext.strides());
  }

  HD_INLINE self_type inc_x(int n = 1) const { return inc<0>(n); }

  HD_INLINE self_type inc_y(int n = 1) const { return inc<1>(n); }

  HD_INLINE self_type inc_z(int n = 1) const { return inc<2>(n); }

  HD_INLINE self_type dec_x(int n = 1) const { return dec<0>(n); }

  HD_INLINE self_type dec_y(int n = 1) const { return dec<1>(n); }

  HD_INLINE self_type dec_z(int n = 1) const { return dec<2>(n); }
};

template <int Rank, int N>
HD_INLINE idx_col_major_t<Rank>
inc_x(const idx_col_major_t<Rank>& idx, const vec_t<uint32_t, N>& ext,
      int n = 1) {
  return idx + n;
}

template <int Rank, int N>
HD_INLINE idx_col_major_t<Rank>
dec_x(const idx_col_major_t<Rank>& idx, const vec_t<uint32_t, N>& ext,
      int n = 1) {
  return idx - n;
}

template <int Rank, int N>
HD_INLINE idx_col_major_t<Rank>
inc_y(const idx_col_major_t<Rank>& idx, const vec_t<uint32_t, N>& ext,
      int n = 1) {
  return idx + n * ext[0];
}

template <int Rank, int N>
HD_INLINE idx_col_major_t<Rank>
dec_y(const idx_col_major_t<Rank>& idx, const vec_t<uint32_t, N>& ext,
      int n = 1) {
  return idx - n * ext[0];
}

template <int Rank, int N>
HD_INLINE idx_col_major_t<Rank>
inc_z(const idx_col_major_t<Rank>& idx, const vec_t<uint32_t, N>& ext,
      int n = 1) {
  return idx + n * ext[0] * ext[1];
}

template <int Rank, int N>
HD_INLINE idx_col_major_t<Rank>
dec_z(const idx_col_major_t<Rank>& idx, const vec_t<uint32_t, N>& ext,
      int n = 1) {
  return idx - n * ext[0] * ext[1];
}

template <int Rank>
HD_INLINE index_t<Rank>
get_pos(const idx_col_major_t<Rank>& idx, const extent_t<Rank>& ext) {
  return idx.get_pos();
}

template <>
HD_INLINE index_t<1>
get_pos(const idx_col_major_t<1>& idx, const extent_t<1>& ext) {
  return index_t<1>(idx.linear);
}

template <>
HD_INLINE index_t<2>
get_pos(const idx_col_major_t<2>& idx, const extent_t<2>& ext) {
  return index_t<2>(idx.linear % ext[0], idx.linear / ext[0]);
}

template <>
HD_INLINE index_t<3>
get_pos(const idx_col_major_t<3>& idx, const extent_t<3>& ext) {
  return index_t<3>(idx.linear % ext[0], (idx.linear / ext[0]) % ext[1],
                    idx.linear / (ext[0] * ext[1]));
}

template <int Rank>
struct idx_row_major_t : public idx_base_t<idx_row_major_t<Rank>, Rank> {
  // index_t<Rank> strides;
  const extent_t<Rank>& ext;

  typedef idx_base_t<idx_row_major_t<Rank>, Rank> base_type;
  typedef idx_row_major_t<Rank> self_type;

  HOST_DEVICE idx_row_major_t(uint64_t n, const extent_t<Rank>& extent)
      : ext(extent) {
    this->linear = n;
  }

  HOST_DEVICE idx_row_major_t(const index_t<Rank>& pos,
                              const extent_t<Rank>& extent)
      : ext(extent) {
    this->linear = to_linear(pos);
  }

  inline index_t<Rank> get_pos() const { return pos(this->linear); }

  HD_INLINE uint64_t to_linear(const index_t<Rank>& pos) const {
    uint64_t result = pos[Rank - 1];
    int64_t stride = this->ext[Rank - 1];
    if (Rank > 1) {
#pragma unroll
      for (int i = Rank - 2; i >= 0; i--) {
        result += pos[i] * stride;
        stride *= this->ext[i];
      }
    }
    return result;
  }

  HD_INLINE index_t<Rank> pos(uint64_t linear) const {
    auto result = index_t<Rank>{};
    auto n = linear;
#pragma unroll
    for (int i = Rank - 1; i >= 0; i--) {
      result[i] = n % this->ext[i];
      n /= this->ext[i];
    }
    return result;
  }

  template <int Dir>
  // HD_INLINE std::enable_if_t <
  // Dir<Rank, self_type> inc(int n = 1) const {
  HD_INLINE self_type inc(int n = 1) const {
    auto result = *this;
    // result.pos[Dir] += n;
    int64_t stride = 1;
#pragma unroll
    for (int i = Rank - 2; i >= Dir; i--) {
      stride *= this->ext[i + 1];
    }
    result.linear = (long)result.linear + n * stride;
    return result;
  }

  template <int Dir>
  // HD_INLINE std::enable_if_t <
  // Dir<Rank, self_type> dec(int n = 1) const {
  HD_INLINE self_type dec(int n = 1) const {
    auto result = *this;
    // result.pos[Dir] += n;
    int64_t stride = 1;
#pragma unroll
    for (int i = Rank - 2; i >= Dir; i--) {
      stride *= this->ext[i + 1];
    }
    result.linear = (long)result.linear - n * stride;
    return result;
  }

  HD_INLINE self_type inc_x(int n = 1) const { return inc<Rank - 1>(n); }

  HD_INLINE self_type inc_y(int n = 1) const { return inc<Rank - 2>(n); }

  HD_INLINE self_type inc_z(int n = 1) const { return inc<Rank - 3>(n); }

  HD_INLINE self_type dec_x(int n = 1) const { return dec<Rank - 1>(n); }

  HD_INLINE self_type dec_y(int n = 1) const { return dec<Rank - 2>(n); }

  HD_INLINE self_type dec_z(int n = 1) const { return dec<Rank - 3>(n); }
};

template <int Rank>
HD_INLINE index_t<Rank>
get_pos(const idx_row_major_t<Rank>& idx, const extent_t<Rank>& ext) {
  return idx.get_pos();
}

template <>
HD_INLINE index_t<1>
get_pos(const idx_row_major_t<1>& idx, const extent_t<1>& ext) {
  return index_t<1>(idx.linear);
}

template <>
HD_INLINE index_t<2>
get_pos(const idx_row_major_t<2>& idx, const extent_t<2>& ext) {
  return index_t<2>(idx.linear / ext[0], idx.linear % ext[0]);
}

template <>
HD_INLINE index_t<3>
get_pos(const idx_row_major_t<3>& idx, const extent_t<3>& ext) {
  return index_t<3>(idx.linear / (ext[0] * ext[1]),
                    (idx.linear / ext[0]) % ext[1], idx.linear % ext[0]);
}

template <int Rank>
struct idx_zorder_t : public idx_base_t<idx_zorder_t<Rank>, Rank> {};

template <>
struct idx_zorder_t<2> : public idx_base_t<idx_zorder_t<2>, 2> {
  typedef idx_base_t<idx_zorder_t<2>, 2> base_type;
  typedef idx_zorder_t<2> self_type;

  HOST_DEVICE idx_zorder_t(uint64_t n, const extent_t<2>& extent) {
    this->linear = n;
  }

  HOST_DEVICE idx_zorder_t(const index_t<2>& pos, const extent_t<2>& extent) {
    this->linear = to_linear(pos);
  }

  HD_INLINE index_t<2> get_pos() const { return pos(this->linear); }

  HD_INLINE uint64_t to_linear(const index_t<2>& pos) const {
    return morton2d<uint32_t>(pos[0], pos[1]).key;
  }

  HD_INLINE index_t<2> pos(uint64_t linear) const {
    // auto result = index_t<2>{};
    uint64_t x, y;
    morton2(linear).decode(x, y);
    // result[0] = x;
    // result[1] = y;
    return index_t<2>(x, y);
  }

  template <int Dir>
  HD_INLINE self_type inc(int n = 1) const;

  template <int Dir>
  HD_INLINE self_type dec(int n = 1) const;

  HD_INLINE self_type inc_x(int n = 1) const {
    auto result = *this;
    // result.pos[0] += n;
    auto m = morton2(result.linear).incX(n);
    result.linear = m.key;
    return result;
  }

  HD_INLINE self_type inc_y(int n = 1) const {
    auto result = *this;
    // result.pos[1] += n;
    auto m = morton2(result.linear).incY(n);
    result.linear = m.key;
    return result;
  }

  HD_INLINE self_type dec_x(int n = 1) const {
    auto result = *this;
    // result.pos[0] += n;
    auto m = morton2(result.linear).decX(n);
    result.linear = m.key;
    return result;
  }

  HD_INLINE self_type dec_y(int n = 1) const {
    auto result = *this;
    // result.pos[1] -= n;
    auto m = morton2(result.linear).decY(n);
    result.linear = m.key;
    return result;
  }
};

template <>
HD_INLINE idx_zorder_t<2>
idx_zorder_t<2>::inc<0>(int n) const {
  return inc_x(n);
}

template <>
HD_INLINE idx_zorder_t<2>
idx_zorder_t<2>::inc<1>(int n) const {
  return inc_y(n);
}

template <>
HD_INLINE idx_zorder_t<2>
idx_zorder_t<2>::dec<0>(int n) const {
  return dec_x(n);
}

template <>
HD_INLINE idx_zorder_t<2>
idx_zorder_t<2>::dec<1>(int n) const {
  return dec_y(n);
}

template <>
struct idx_zorder_t<3> : public idx_base_t<idx_zorder_t<3>, 3> {
  typedef idx_base_t<idx_zorder_t<3>, 3> base_type;
  typedef idx_zorder_t<3> self_type;

  HOST_DEVICE idx_zorder_t(uint64_t n, const extent_t<3>& extent) {
    this->linear = n;
  }

  HOST_DEVICE idx_zorder_t(const index_t<3>& pos, const extent_t<3>& extent) {
    this->linear = to_linear(pos);
  }

  HD_INLINE index_t<3> get_pos() const { return pos(this->linear); }

  HD_INLINE uint64_t to_linear(const index_t<3>& pos) const {
    return morton3d<uint32_t>(pos[0], pos[1], pos[2]).key;
  }

  HD_INLINE index_t<3> pos(uint64_t linear) const {
    // auto result = index_t<3>{};
    uint64_t x, y, z;
    morton3(linear).decode(x, y, z);
    return index_t<3>(x, y, z);
  }

  template <int Dir>
  HD_INLINE self_type inc(int n = 1) const;

  template <int Dir>
  HD_INLINE self_type dec(int n = 1) const;

  HD_INLINE self_type inc_x(int n = 1) const {
    auto result = *this;
    // result.pos[0] += n;
    auto m = morton3(result.linear).incX(n);
    result.linear = m.key;
    return result;
  }

  HD_INLINE self_type inc_y(int n = 1) const {
    auto result = *this;
    // result.pos[0] += n;
    auto m = morton3(result.linear).incY(n);
    result.linear = m.key;
    return result;
  }

  HD_INLINE self_type inc_z(int n = 1) const {
    auto result = *this;
    // result.pos[0] += n;
    auto m = morton3(result.linear).incZ(n);
    result.linear = m.key;
    return result;
  }

  HD_INLINE self_type dec_x(int n = 1) const {
    auto result = *this;
    // result.pos[0] += n;
    auto m = morton3(result.linear).decX(n);
    result.linear = m.key;
    return result;
  }

  HD_INLINE self_type dec_y(int n = 1) const {
    auto result = *this;
    // result.pos[0] += n;
    auto m = morton3(result.linear).decY(n);
    result.linear = m.key;
    return result;
  }

  HD_INLINE self_type dec_z(int n = 1) const {
    auto result = *this;
    // result.pos[0] += n;
    auto m = morton3(result.linear).decZ(n);
    result.linear = m.key;
    return result;
  }
};

template <>
HD_INLINE idx_zorder_t<3>
idx_zorder_t<3>::inc<0>(int n) const {
  return inc_x(n);
}

template <>
HD_INLINE idx_zorder_t<3>
idx_zorder_t<3>::inc<1>(int n) const {
  return inc_y(n);
}

template <>
HD_INLINE idx_zorder_t<3>
idx_zorder_t<3>::inc<2>(int n) const {
  return inc_z(n);
}

template <>
HD_INLINE idx_zorder_t<3>
idx_zorder_t<3>::dec<0>(int n) const {
  return dec_x(n);
}

template <>
HD_INLINE idx_zorder_t<3>
idx_zorder_t<3>::dec<1>(int n) const {
  return dec_y(n);
}

template <>
HD_INLINE idx_zorder_t<3>
idx_zorder_t<3>::dec<2>(int n) const {
  return dec_z(n);
}

template <int Rank>
HD_INLINE index_t<Rank>
get_pos(const idx_zorder_t<Rank>& idx, const extent_t<Rank>& ext) {
  return idx.get_pos();
}

}  // namespace Aperture

#endif  // __INDEX_H_
