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
#include "utils/indexable.hpp"

namespace Aperture {

template <typename Indexable, typename Idx_t>
class ndsubset_const_t {
 public:
  ndsubset_const_t(const Indexable& array, const index_t<Idx_t::dim>& begin,
                   const extent_t<Idx_t::dim>& ext)
                   // const extent_t<Idx_t::dim>& parent_ext)
      : m_array(array), m_begin(begin), m_ext(ext) {}
  ~ndsubset_const_t() {}

  // inline auto at(const Idx_t& idx) const { return m_array.at(idx); }
  inline auto at(const index_t<Idx_t::dim>& pos) const { return m_array.at(m_begin + pos); }

  // private:
  Indexable m_array;
  index_t<Idx_t::dim> m_begin;
  extent_t<Idx_t::dim> m_ext;
  // extent_t<Idx_t::dim> m_parent_ext;
};

template <typename Indexable, typename Idx_t>
class ndsubset_t {
 public:
  typedef ndsubset_t<Indexable, Idx_t> self_type;

  ndsubset_t(const Indexable& array, const index_t<Idx_t::dim>& begin,
             const extent_t<Idx_t::dim>& ext)
             // const extent_t<Idx_t::dim>& parent_ext)
      : m_array(array), m_begin(begin), m_ext(ext) {}
  ~ndsubset_t() {}

  // inline auto& at(const Idx_t& idx) { return m_array.at(idx); }
  inline auto& at(const index_t<Idx_t::dim>& pos) { return m_array.at(m_begin + pos); }

  template <typename Other>
  void check_ext(const ndsubset_const_t<Other, Idx_t>& subset) {
    if (m_ext != subset.m_ext) {
      throw std::runtime_error("subset sizes do not match!");
    }
  }

  template <typename Other, typename Op>
  void loop_subset(const ndsubset_const_t<Other, Idx_t>& subset, const Op& op) {
    check_ext(subset);
    using col_idx_t = idx_col_major_t<Idx_t::dim>;
    for (auto idx : range(col_idx_t(0, m_ext),
                          col_idx_t(m_ext.size(), m_ext))) {
      op(at(idx.get_pos()), subset.at(idx.get_pos()));
    }
  }

  template <typename OtherIndexable, typename Op,
            typename = is_host_const_indexable<OtherIndexable>>
  void loop_indexable(const OtherIndexable& other, const Op& op) {
    using col_idx_t = idx_col_major_t<Idx_t::dim>;
    for (auto idx : range(col_idx_t(0, m_ext),
                          col_idx_t(m_ext.size(), m_ext))) {
      op(m_array.at(m_begin + idx.get_pos()),
         other.at(m_begin + idx.get_pos()));
    }
  }

  template <typename Op>
  void loop_self(const Op& op) {
    using col_idx_t = idx_col_major_t<Idx_t::dim>;
    for (auto idx : range(col_idx_t(0, m_ext),
                          col_idx_t(m_ext.size(), m_ext))) {
      op(m_array.at(m_begin + idx.get_pos()));
    }
  }

  self_type& operator=(const self_type& other) = delete;

  template <typename Other>
  self_type& operator=(const ndsubset_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [](auto& x, const auto& y) { x = y; });
    return *this;
  }

  template <typename OtherIndexable>
  self_type& operator=(const OtherIndexable& other) {
    loop_indexable(other, [](auto& x, const auto& y) { x = y; });
    return *this;
  }

  self_type& operator=(const typename Indexable::value_t& value) {
    loop_self([value](auto& x) { x = value; });
    return *this;
  }

  template <typename Other>
  self_type& operator+=(const ndsubset_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [](auto& x, const auto& y) { x += y; });
    return *this;
  }

  template <typename OtherIndexable>
  self_type& operator+=(const OtherIndexable& other) {
    loop_indexable(other, [](auto& x, const auto& y) { x += y; });
    return *this;
  }

  self_type& operator+=(const typename Indexable::value_t& value) {
    loop_self([value](auto& x) { x += value; });
    return *this;
  }

  template <typename Other>
  self_type& operator-=(const ndsubset_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [](auto& x, const auto& y) { x -= y; });
    return *this;
  }

  template <typename OtherIndexable>
  self_type& operator-=(const OtherIndexable& other) {
    loop_indexable(other, [](auto& x, const auto& y) { x -= y; });
    return *this;
  }

  self_type& operator-=(const typename Indexable::value_t& value) {
    loop_self([value](auto& x) { x -= value; });
    return *this;
  }

  template <typename Other>
  self_type& operator*=(const ndsubset_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [](auto& x, const auto& y) { x *= y; });
    return *this;
  }

  template <typename OtherIndexable>
  self_type& operator*=(const OtherIndexable& other) {
    loop_indexable(other, [](auto& x, const auto& y) { x *= y; });
    return *this;
  }

  self_type& operator*=(const typename Indexable::value_t& value) {
    loop_self([value](auto& x) { x *= value; });
    return *this;
  }

  template <typename Other>
  self_type& operator/=(const ndsubset_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [](auto& x, const auto& y) { x /= y; });
    return *this;
  }

  template <typename OtherIndexable>
  self_type& operator/=(const OtherIndexable& other) {
    loop_indexable(other, [](auto& x, const auto& y) { x /= y; });
    return *this;
  }

  self_type& operator/=(const typename Indexable::value_t& value) {
    loop_self([value](auto& x) { x /= value; });
    return *this;
  }

  // private:
  Indexable m_array;
  index_t<Idx_t::dim> m_begin;
  extent_t<Idx_t::dim> m_ext;
  // extent_t<Idx_t::dim> m_parent_ext;
};

template <typename Indexable, typename = typename std::enable_if_t<
                                  is_host_const_indexable<Indexable>::value>>
ndsubset_const_t<Indexable, typename Indexable::idx_t>
select(const Indexable& array, const index_t<Indexable::idx_t::dim>& begin,
       const extent_t<Indexable::idx_t::dim>& ext) {
       // const extent_t<Indexable::idx_t::dim>& parent_ext) {
  return ndsubset_const_t<Indexable, typename Indexable::idx_t>(
      array, begin, ext);
}

template <typename T, int Rank, typename Idx_t>
auto
select(multi_array<T, Rank, Idx_t>& array, const index_t<Rank>& begin,
       const extent_t<Rank>& ext) {
  return ndsubset_t<typename multi_array<T, Rank, Idx_t>::ref_t, Idx_t>(
      array.ref(), begin, ext);
}

template <typename T, int Rank, typename Idx_t>
auto
select(multi_array<T, Rank, Idx_t>& array) {
  return ndsubset_t<typename multi_array<T, Rank, Idx_t>::ref_t, Idx_t>(
      array.ref(), array.begin().get_pos(), array.extent());
}

}  // namespace Aperture

#endif  // __NDSUBSET_H_
