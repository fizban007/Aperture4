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
#include "utils/indexable.hpp"
#include "utils/range.hpp"
#include "utils/type_traits.hpp"
#include "utils/kernel_helper.hpp"

namespace Aperture {

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)

template <typename Indexable, typename Idx_t>
class ndsubset_dev_const_t {
 public:
  ndsubset_dev_const_t(const Indexable& array, const index_t<Idx_t::dim>& begin,
                       const extent_t<Idx_t::dim>& ext)
      // const extent_t<Idx_t::dim>& parent_ext)
      : m_array(array), m_begin(begin), m_ext(ext) {}
  ~ndsubset_dev_const_t() {}

  // inline auto at_dev(const Idx_t& idx) const { return m_array.at_dev(idx); }
  // HD_INLINE auto at_dev(const index_t<Idx_t::dim>& pos) const { return
  // m_array.at(pos); }

  // private:
  Indexable m_array;
  index_t<Idx_t::dim> m_begin;
  extent_t<Idx_t::dim> m_ext;
  // extent_t<Idx_t::dim> m_parent_ext;
};

template <typename Indexable, typename Idx_t>
class ndsubset_dev_t {
 public:
  typedef ndsubset_dev_t<Indexable, Idx_t> self_type;

  ndsubset_dev_t(const Indexable& array, const index_t<Idx_t::dim>& begin,
                 const extent_t<Idx_t::dim>& ext)
      // const extent_t<Idx_t::dim>& parent_ext)
      : m_array(array), m_begin(begin), m_ext(ext) {}
  ~ndsubset_dev_t() {}

  // inline auto& at_dev(const Idx_t& idx) { return m_array.at_dev(idx); }
  // HD_INLINE auto& at_dev(const index_t<Idx_t::dim>& pos) { return
  // m_array.at(pos); }

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
    auto kernel = [op] __device__(auto dst, auto src, auto dst_pos, auto src_pos,
                                  auto ext) {
      using col_idx_t = idx_col_major_t<Idx_t::dim>;
      for (auto idx : grid_stride_range(col_idx_t(0, ext),
                                        col_idx_t(ext.size(), ext))) {
        // Always use column major inside loop to simplify conversion
        // between different indexing schemes
        // op(dst.at_dev(dst_pos + idx.get_pos()),
        //    src.at_dev(src_pos + idx.get_pos()));
        op(dst.at_dev(dst_pos + get_pos(idx, ext)),
           src.at_dev(src_pos + get_pos(idx, ext)));
      }
    };
    kernel_exec_policy p;
    configure_grid(p, kernel, m_array, subset.m_array, m_begin, subset.m_begin, m_ext);
    if (m_stream != 0) { p.set_stream(m_stream); }
    kernel_launch(p, kernel,
        m_array, subset.m_array, m_begin, subset.m_begin, m_ext);
    GpuSafeCall(gpuDeviceSynchronize());
    GpuCheckError();
  }

  template <typename OtherIndexable, typename Op>
  void loop_indexable(const OtherIndexable& other, const Op& op) {
    auto kernel = [] __device__(Indexable dst, OtherIndexable src,
                                index_t<Idx_t::dim> pos,
                                extent_t<Idx_t::dim> ext, Op op) {
      using col_idx_t = idx_col_major_t<Idx_t::dim>;
      for (auto idx :
           grid_stride_range(col_idx_t(0, ext), col_idx_t(ext.size(), ext))) {
        // Always use column major inside loop to simplify conversion
        // between different indexing schemes
        // op(dst.at_dev(pos + idx.get_pos()), src.at_dev(pos + idx.get_pos()));
        op(dst.at_dev(pos + get_pos(idx, ext)), src.at_dev(pos + get_pos(idx, ext)));
      }
    };
    kernel_exec_policy p;
    configure_grid(p, kernel, m_array, other, m_begin, m_ext, op);
    if (m_stream != 0) { p.set_stream(m_stream); }
    kernel_launch(p, kernel, m_array, other, m_begin, m_ext, op);
    GpuSafeCall(gpuDeviceSynchronize());
    GpuCheckError();
  }

  template <typename Op>
  void loop_self(const Op& op) {
    auto kernel =
        [op] __device__(auto dst, auto pos, auto ext) {
          using col_idx_t = idx_col_major_t<Idx_t::dim>;
          for (auto idx : grid_stride_range(col_idx_t(0, ext),
                                            col_idx_t(ext.size(), ext))) {
            // Always use column major inside loop to simplify conversion
            // between different indexing schemes
            // op(dst.at_dev(pos + idx.get_pos()));
            op(dst.at_dev(pos + get_pos(idx, ext)));
          }
        };
    kernel_exec_policy p;
    configure_grid(p, kernel, m_array, m_begin, m_ext);
    if (m_stream != 0) { p.set_stream(m_stream); }
    kernel_launch(p, kernel, m_array, m_begin, m_ext);
    GpuSafeCall(gpuDeviceSynchronize());
    GpuCheckError();
  }

  self_type& operator=(const self_type& other) = delete;

  template <typename Other>
  self_type& operator=(const ndsubset_dev_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [] __device__(auto& x, const auto& y) { x = y; });
    return *this;
  }

  template <typename OtherIndexable>
  self_type& operator=(const OtherIndexable& other) {
    loop_indexable(other, [] __device__(auto& x, const auto& y) { x = y; });
    return *this;
  }

  self_type& operator=(const typename Indexable::value_t& value) {
    loop_self([value] __device__(auto& x) { x = value; });
    return *this;
  }

  template <typename Other>
  self_type& operator+=(const ndsubset_dev_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [] __device__(auto& x, const auto& y) { x += y; });
    return *this;
  }

  template <typename OtherIndexable>
  self_type& operator+=(const OtherIndexable& other) {
    loop_indexable(other, [] __device__(auto& x, const auto& y) { x += y; });
    return *this;
  }

  self_type& operator+=(const typename Indexable::value_t& value) {
    loop_self([value] __device__(auto& x) { x += value; });
    return *this;
  }

  template <typename Other>
  self_type& operator-=(const ndsubset_dev_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [] __device__(auto& x, const auto& y) { x -= y; });
    return *this;
  }

  template <typename OtherIndexable>
  self_type& operator-=(const OtherIndexable& other) {
    loop_indexable(other, [] __device__(auto& x, const auto& y) { x -= y; });
    return *this;
  }

  self_type& operator-=(const typename Indexable::value_t& value) {
    loop_self([value] __device__(auto& x) { x -= value; });
    return *this;
  }

  template <typename Other>
  self_type& operator*=(const ndsubset_dev_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [] __device__(auto& x, const auto& y) { x *= y; });
    return *this;
  }

  template <typename OtherIndexable>
  self_type& operator*=(const OtherIndexable& other) {
    loop_indexable(other, [] __device__(auto& x, const auto& y) { x *= y; });
    return *this;
  }

  self_type& operator*=(const typename Indexable::value_t& value) {
    loop_self([value] __device__(auto& x) { x *= value; });
    return *this;
  }

  template <typename Other>
  self_type& operator/=(const ndsubset_dev_const_t<Other, Idx_t>& subset) {
    loop_subset(subset, [] __device__(auto& x, const auto& y) { x /= y; });
    return *this;
  }

  template <typename OtherIndexable>
  self_type& operator/=(const OtherIndexable& other) {
    loop_indexable(other, [] __device__(auto& x, const auto& y) { x /= y; });
    return *this;
  }

  self_type& operator/=(const typename Indexable::value_t& value) {
    loop_self([value] __device__(auto& x) { x /= value; });
    return *this;
  }

  self_type& with_stream(gpuStream_t stream) {
    m_stream = stream;
    return *this;
  }

  // private:
  Indexable m_array;
  index_t<Idx_t::dim> m_begin;
  extent_t<Idx_t::dim> m_ext;
  gpuStream_t m_stream = 0;
  // extent_t<Idx_t::dim> m_parent_ext;
};

template <typename Indexable, typename = typename std::enable_if_t<
                                  is_dev_const_indexable<Indexable>::value>>
ndsubset_dev_const_t<Indexable, typename Indexable::idx_t>
select_dev(const Indexable& array, const index_t<Indexable::idx_t::dim>& begin,
           const extent_t<Indexable::idx_t::dim>& ext) {
  // const extent_t<Indexable::idx_t::dim>& parent_ext) {
  return ndsubset_dev_const_t<Indexable, typename Indexable::idx_t>(array,
                                                                    begin, ext);
}

template <typename T, int Rank, typename Idx_t>
auto
select_dev(const multi_array<T, Rank, Idx_t>& array, const index_t<Rank>& begin,
           const extent_t<Rank>& ext) {
  return ndsubset_dev_const_t<typename multi_array<T, Rank, Idx_t>::cref_t, Idx_t>(
      array.cref(), begin, ext);
}

template <typename T, int Rank, typename Idx_t>
auto
select_dev(const multi_array<T, Rank, Idx_t>& array) {
  return ndsubset_dev_const_t<typename multi_array<T, Rank, Idx_t>::cref_t, Idx_t>(
      array.cref(), array.begin().get_pos(), array.extent());
}

template <typename T, int Rank, typename Idx_t>
auto
select_dev_const(multi_array<T, Rank, Idx_t>& array, const index_t<Rank>& begin,
                 const extent_t<Rank>& ext) {
  return ndsubset_dev_const_t<typename multi_array<T, Rank, Idx_t>::cref_t, Idx_t>(
      array.cref(), begin, ext);
}

template <typename T, int Rank, typename Idx_t>
auto
select_dev_const(multi_array<T, Rank, Idx_t>& array) {
  return ndsubset_dev_const_t<typename multi_array<T, Rank, Idx_t>::cref_t, Idx_t>(
      array.cref(), array.begin().get_pos(), array.extent());
}

template <typename T, int Rank, typename Idx_t>
auto
select_dev(multi_array<T, Rank, Idx_t>& array, const index_t<Rank>& begin,
           const extent_t<Rank>& ext) {
  return ndsubset_dev_t<typename multi_array<T, Rank, Idx_t>::ref_t, Idx_t>(
      array.ref(), begin, ext);
}

template <typename T, int Rank, typename Idx_t>
auto
select_dev(multi_array<T, Rank, Idx_t>& array) {
  return ndsubset_dev_t<typename multi_array<T, Rank, Idx_t>::ref_t, Idx_t>(
      array.ref(), array.begin().get_pos(), array.extent());
}

#endif

}  // namespace Aperture

#endif  // __NDSUBSET_DEV_H_
