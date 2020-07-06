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

#ifndef __NDSUBSET_DEV_H_
#define __NDSUBSET_DEV_H_

#include "core/cuda_control.h"
#include "core/multi_array.hpp"
#include "utils/range.hpp"
#include "utils/type_traits.hpp"

namespace Aperture {

#ifdef CUDA_ENABLED

template <typename Indexable, typename Idx_t>
class ndsubset_dev_const_t {
 public:
  ndsubset_dev_const_t(const Indexable& array, const index_t<Idx_t::dim>& begin,
                       const extent_t<Idx_t::dim>& ext,
                       const extent_t<Idx_t::dim>& parent_ext)
      : m_array(array), m_begin(begin), m_ext(ext), m_parent_ext(parent_ext) {}
  ~ndsubset_dev_const_t() {}

  inline auto at_dev(const Idx_t& idx) const { return m_array.at_dev(idx); }

  // private:
  Indexable m_array;
  index_t<Idx_t::dim> m_begin;
  extent_t<Idx_t::dim> m_ext;
  extent_t<Idx_t::dim> m_parent_ext;
};

template <typename Indexable, typename Idx_t>
class ndsubset_dev_t {
 public:
  typedef ndsubset_dev_t<Indexable, Idx_t> self_type;

  ndsubset_dev_t(const Indexable& array, const index_t<Idx_t::dim>& begin,
                 const extent_t<Idx_t::dim>& ext,
                 const extent_t<Idx_t::dim>& parent_ext)
      : m_array(array), m_begin(begin), m_ext(ext), m_parent_ext(parent_ext) {}
  ~ndsubset_dev_t() {}

  inline auto& at_dev(const Idx_t& idx) { return m_array.at_dev(idx); }

  template <typename Other>
  void check_ext(const ndsubset_dev_const_t<Other, Idx_t>& subset) {
    if (m_ext != subset.m_ext) {
      throw std::runtime_error("subset sizes do not match!");
    }
  }

  template <typename Other, typename Op>
  void loop_subset(const ndsubset_dev_const_t<Other, Idx_t>& subset,
                   const Op& op) {
    check_ext(subset);
    kernel_launch(
        [op] __device__(auto dst, auto src, auto dst_pos, auto src_pos,
                        auto dst_ext, auto src_ext, auto ext) {
          for (auto n : grid_stride_range(0, ext.size())) {
            // Always use column major inside loop to simplify conversion
            // between different indexing schemes
            idx_col_major_t<Idx_t::dim> idx(n, ext);
            auto idx_dst = Idx_t(dst_pos + idx.get_pos(), dst_ext);
            auto idx_src = Idx_t(src_pos + idx.get_pos(), src_ext);
            op(dst.at_dev(idx_dst), src.at_dev(idx_src));
          }
        },
        m_array, subset.m_array, m_begin, subset.m_begin, m_parent_ext,
        subset.m_parent_ext, m_ext);
  }

  template <typename Other>
  self_type& operator=(const ndsubset_dev_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [] __device__(auto& x, const auto& y) { x = y; });
    return *this;
  }

  template <typename Other>
  self_type& operator+=(const ndsubset_dev_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [] __device__(auto& x, const auto& y) { x += y; });
    return *this;
  }

  template <typename Other>
  self_type& operator-=(const ndsubset_dev_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [] __device__(auto& x, const auto& y) { x -= y; });
    return *this;
  }

  template <typename Other>
  self_type& operator*=(const ndsubset_dev_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [] __device__(auto& x, const auto& y) { x *= y; });
    return *this;
  }

  template <typename Other>
  self_type& operator/=(const ndsubset_dev_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [] __device__(auto& x, const auto& y) { x /= y; });
    return *this;
  }

  // private:
  Indexable m_array;
  index_t<Idx_t::dim> m_begin;
  extent_t<Idx_t::dim> m_ext;
  extent_t<Idx_t::dim> m_parent_ext;
};

template <typename Indexable, typename = typename std::enable_if_t<
                                  is_dev_const_indexable<Indexable>::value>>
ndsubset_dev_const_t<Indexable, typename Indexable::idx_t>
select_dev(const Indexable& array, const index_t<Indexable::idx_t::dim>& begin,
           const extent_t<Indexable::idx_t::dim>& ext,
           const extent_t<Indexable::idx_t::dim>& parent_ext) {
  return ndsubset_dev_const_t<Indexable, typename Indexable::idx_t>(
      array, begin, ext, parent_ext);
}

template <typename T, int Rank, typename Idx_t>
auto
select_dev(multi_array<T, Rank, Idx_t>& array, const index_t<Rank>& begin,
           const extent_t<Rank>& ext) {
  return ndsubset_dev_t<typename multi_array<T, Rank, Idx_t>::ref_t, Idx_t>(
      array.ref(), begin, ext, array.extent());
}

template <typename T, int Rank, typename Idx_t>
auto
select_dev(multi_array<T, Rank, Idx_t>& array) {
  return ndsubset_dev_t<typename multi_array<T, Rank, Idx_t>::ref_t, Idx_t>(
      array.ref(), array.begin().get_pos(), array.extent(), array.extent());
}

#endif

}  // namespace Aperture

#endif  // __NDSUBSET_DEV_H_
