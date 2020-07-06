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

#ifndef __NDSUBSET_H_
#define __NDSUBSET_H_

#include "core/cuda_control.h"
#include "core/multi_array.hpp"
#include "utils/range.hpp"
#include "utils/type_traits.hpp"

namespace Aperture {

template <typename Indexable, typename Idx_t>
class ndsubset_const_t {
 public:
  ndsubset_const_t(const Indexable& array, const index_t<Idx_t::dim>& begin,
                   const extent_t<Idx_t::dim>& ext,
                   const extent_t<Idx_t::dim>& parent_ext)
      : m_array(array), m_begin(begin), m_ext(ext), m_parent_ext(parent_ext) {}
  ~ndsubset_const_t() {}

  inline auto at(const Idx_t& idx) const { return m_array.at(idx); }

  // private:
  Indexable m_array;
  index_t<Idx_t::dim> m_begin;
  extent_t<Idx_t::dim> m_ext;
  extent_t<Idx_t::dim> m_parent_ext;
};

template <typename Indexable, typename Idx_t>
class ndsubset_t {
 public:
  typedef ndsubset_t<Indexable, Idx_t> self_type;

  ndsubset_t(const Indexable& array, const index_t<Idx_t::dim>& begin,
             const extent_t<Idx_t::dim>& ext,
             const extent_t<Idx_t::dim>& parent_ext)
      : m_array(array), m_begin(begin), m_ext(ext), m_parent_ext(parent_ext) {}
  ~ndsubset_t() {}

  inline auto& at(const Idx_t& idx) { return m_array.at(idx); }

  template <typename Other>
  void check_ext(const ndsubset_const_t<Other, Idx_t>& subset) {
    if (m_ext != subset.m_ext) {
      throw std::runtime_error("subset sizes do not match!");
    }
  }

  template <typename Other, typename Op>
  void loop_subset(const ndsubset_const_t<Other, Idx_t>& subset, const Op& op) {
    check_ext(subset);
    for (auto n : range(0, m_ext.size())) {
      idx_col_major_t<Idx_t::dim> idx(n, m_ext);
      auto idx_dst = Idx_t(m_begin + idx.get_pos(), m_parent_ext);
      auto idx_src = Idx_t(subset.m_begin + idx.get_pos(), subset.m_parent_ext);
      op(m_array.at(idx_dst), subset.at(idx_src));
    }
  }

  template <typename Other>
  self_type& operator=(const ndsubset_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [](auto& x, const auto& y) { x = y; });
    return *this;
  }

  template <typename Other>
  self_type& operator+=(const ndsubset_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [](auto& x, const auto& y) { x += y; });
    return *this;
  }

  template <typename Other>
  self_type& operator-=(const ndsubset_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [](auto& x, const auto& y) { x -= y; });
    return *this;
  }

  template <typename Other>
  self_type& operator*=(const ndsubset_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [](auto& x, const auto& y) { x *= y; });
    return *this;
  }

  template <typename Other>
  self_type& operator/=(const ndsubset_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [](auto& x, const auto& y) { x /= y; });
    return *this;
  }

  // private:
  Indexable m_array;
  index_t<Idx_t::dim> m_begin;
  extent_t<Idx_t::dim> m_ext;
  extent_t<Idx_t::dim> m_parent_ext;
};

template <typename Indexable, typename = typename std::enable_if_t<
                                  is_host_const_indexable<Indexable>::value>>
ndsubset_const_t<Indexable, typename Indexable::idx_t>
select(const Indexable& array, const index_t<Indexable::idx_t::dim>& begin,
       const extent_t<Indexable::idx_t::dim>& ext,
       const extent_t<Indexable::idx_t::dim>& parent_ext) {
  return ndsubset_const_t<Indexable, typename Indexable::idx_t>(
      array, begin, ext, parent_ext);
}

template <typename T, int Rank, typename Idx_t>
auto
select(multi_array<T, Rank, Idx_t>& array, const index_t<Rank>& begin,
       const extent_t<Rank>& ext) {
  return ndsubset_t<typename multi_array<T, Rank, Idx_t>::ref_t, Idx_t>(
      array.ref(), begin, ext, array.extent());
}

template <typename T, int Rank, typename Idx_t>
auto
select(multi_array<T, Rank, Idx_t>& array) {
  return ndsubset_t<typename multi_array<T, Rank, Idx_t>::ref_t, Idx_t>(
      array.ref(), array.begin().get_pos(), array.extent(), array.extent());
}

}  // namespace Aperture

#endif  // __NDSUBSET_H_
