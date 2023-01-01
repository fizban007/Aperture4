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

#pragma once

#include "core/data_adapter.h"
#include "core/gpu_translation_layer.h"
#include "utils/indexable.hpp"
#include "utils/type_traits.hpp"
#include <functional>

namespace Aperture {

// Expressions

template <typename Op, typename Left, typename Right>
struct binary_exp_t {
  typedef binary_exp_t<Op, Left, Right> self_type;
  typedef typename Left::idx_t idx_t;
  typedef typename Left::value_t value_t;

  Left left;
  Right right;
  Op op;

  HOST_DEVICE binary_exp_t(const Left& t1, const Right& t2, const Op& o)
      : left(t1), right(t2), op(o) {}

  template <typename L = Left, typename R = Right,
            typename = typename std::enable_if<
                conjunction<is_plain_const_indexable<L>,
                            is_plain_const_indexable<R>>::value>::type>
  HD_INLINE auto operator[](const idx_t& idx) const {
    return op(left[idx], right[idx]);
  }

  template <
      typename L = Left, typename R = Right,
      typename = typename std::enable_if<conjunction<
          is_host_const_indexable<L>, is_host_const_indexable<R>>::value>::type>
  // inline auto at(const idx_t& idx) const {
  inline auto at(const index_t<Left::idx_t::dim>& pos) const {
    return op(left.at(pos), right.at(pos));
  }

  template <
      typename L = Left, typename R = Right,
      typename = typename std::enable_if<conjunction<
          is_dev_const_indexable<L>, is_dev_const_indexable<R>>::value>::type>
  HD_INLINE auto at_dev(const index_t<Left::idx_t::dim>& pos) const {
    return op(left.at_dev(pos), right.at_dev(pos));
  }
};

template <typename Op, typename Left, typename Right>
HD_INLINE auto
make_binary_exp(const Left& l, const Right& r, const Op& op) {
  return binary_exp_t<Op, Left, Right>(l, r, op);
}

template <typename Op, typename Arg>
struct unary_exp_t {
  typedef unary_exp_t<Op, Arg> self_type;
  typedef typename Arg::idx_t idx_t;
  typedef typename Arg::value_t value_t;

  Arg arg;
  Op op;

  HOST_DEVICE unary_exp_t(const Arg& t, const Op& o) : arg(t), op(o) {}

  template <typename A = Arg, typename = typename std::enable_if_t<
                                  is_plain_const_indexable<A>::value>>
  HD_INLINE auto operator[](const idx_t& idx) const {
    return op(arg[idx]);
  }

  template <typename A = Arg, typename = typename std::enable_if_t<
                                  is_host_const_indexable<A>::value>>
  inline auto at(const index_t<Arg::idx_t::dim>& pos) const {
    return op(arg.at(pos));
  }

  template <typename A = Arg, typename = typename std::enable_if_t<
                                  is_dev_const_indexable<A>::value>>
  HD_INLINE auto at_dev(const index_t<Arg::idx_t::dim>& pos) const {
    return op(arg.at_dev(pos));
  }
};

template <typename Op, typename Arg>
HD_INLINE auto
make_unary_exp(const Arg& arg, const Op& op) {
  return unary_exp_t<Op, Arg>(arg, op);
}

template <typename Value_t, typename Idx_t>
struct const_exp_t {
  Value_t v;

  typedef const_exp_t<Value_t, Idx_t> self_type;
  typedef Idx_t idx_t;
  typedef Value_t value_t;

  HOST_DEVICE const_exp_t(const Value_t& t) : v(t) {}

  HD_INLINE auto operator[](const idx_t& idx) const { return v; }

  inline auto at(const index_t<idx_t::dim>& pos) const { return v; }

  HD_INLINE auto at_dev(const index_t<idx_t::dim>& pos) const { return v; }
};

// Functions

#define INSTANTIATE_BINARY_OPS(name, op)                                   \
  template <typename Left, typename Right,                                 \
            typename = all_const_indexable<Left, Right>>                   \
  HD_INLINE auto operator op(const Left& l, const Right& r) {              \
    return make_binary_exp(l, r, name<typename Left::value_t>{});          \
  }                                                                        \
                                                                           \
  template <typename Left, typename = is_const_indexable<Left>>            \
  HD_INLINE auto operator op(const Left& l, typename Left::value_t r) {    \
    return make_binary_exp(                                                \
        l, const_exp_t<typename Left::value_t, typename Left::idx_t>(r),   \
        name<typename Left::value_t>{});                                   \
  }                                                                        \
                                                                           \
  template <typename Right, typename = is_const_indexable<Right>>          \
  HD_INLINE auto operator op(typename Right::value_t l, const Right& r) {  \
    return make_binary_exp(                                                \
        const_exp_t<typename Right::value_t, typename Right::idx_t>(l), r, \
        name<typename Right::value_t>{});                                  \
  }

INSTANTIATE_BINARY_OPS(std::plus, +)
INSTANTIATE_BINARY_OPS(std::minus, -)
INSTANTIATE_BINARY_OPS(std::multiplies, *)
INSTANTIATE_BINARY_OPS(std::divides, /)

template <typename Arg, typename = is_const_indexable<Arg>>
HD_INLINE auto
operator-(const Arg& arg) {
  return make_unary_exp(arg, std::negate<typename Arg::value_t>());
}

}  // namespace Aperture

#include "core/multi_array.hpp"

namespace Aperture {

#define INSTANTIATE_MULTIARRAY_BINARY_OPS(op)                                  \
  template <typename T, int Rank, typename Idx_t, typename Right>              \
  auto operator op(const multi_array<T, Rank, Idx_t>& array, const Right& r) { \
    return operator op(array.cref(), r);                                       \
  }                                                                            \
                                                                               \
  template <typename T, int Rank, typename Idx_t, typename Left>               \
  auto operator op(const Left& l, const multi_array<T, Rank, Idx_t>& array) {  \
    return operator op(l, array.cref());                                       \
  }                                                                            \
                                                                               \
  template <typename T, int Rank, typename Idx_t>                              \
  auto operator op(const multi_array<T, Rank, Idx_t>& v1,                      \
                   const multi_array<T, Rank, Idx_t>& v2) {                    \
    return operator op(v1.cref(), v2.cref());                                  \
  }

INSTANTIATE_MULTIARRAY_BINARY_OPS(+);
INSTANTIATE_MULTIARRAY_BINARY_OPS(-);
INSTANTIATE_MULTIARRAY_BINARY_OPS(*);
INSTANTIATE_MULTIARRAY_BINARY_OPS(/);

#define INSTANTIATE_MULTIARRAY_UNARY_OPS(name, op)                     \
  template <typename T, int Rank, typename Idx_t>                      \
  HD_INLINE auto operator op(const multi_array<T, Rank, Idx_t>& arg) { \
    return make_unary_exp(cref(arg), name<T>());                       \
  }

INSTANTIATE_MULTIARRAY_UNARY_OPS(std::negate, -);

template <typename Op, typename Left, typename Right>
struct host_adapter<binary_exp_t<Op, Left, Right>> {
  typedef binary_exp_t<Op, Left, Right> type;
  typedef binary_exp_t<Op, Left, Right> const_type;

  static inline type apply(binary_exp_t<Op, Left, Right>& array) {
    return array;
  }
  static inline type apply(const binary_exp_t<Op, Left, Right>& array) {
    return array;
  }
};

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)

template <typename Op, typename Left, typename Right>
struct gpu_adapter<binary_exp_t<Op, Left, Right>> {
  typedef binary_exp_t<Op, Left, Right> type;
  typedef binary_exp_t<Op, Left, Right> const_type;

  static inline type apply(binary_exp_t<Op, Left, Right>& array) {
    return array;
  }
  static inline type apply(const binary_exp_t<Op, Left, Right>& array) {
    return array;
  }
};

#endif

}  // namespace Aperture
