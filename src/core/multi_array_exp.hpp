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

#ifndef __MULTI_ARRAY_EXP_H_
#define __MULTI_ARRAY_EXP_H_

#include "core/cuda_control.h"
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

  HD_INLINE auto operator[](const idx_t& idx) const {
    return op(left[idx], right[idx]);
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

  template <typename Idx_t>
  HD_INLINE auto operator[](const Idx_t& idx) const {
    return op(arg[idx]);
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

  HD_INLINE auto operator[](const Idx_t& idx) const { return v; }
};

// Functions

#define INSTANTIATE_BINARY_OPS(name, op)                                   \
  template <typename Left, typename Right,                                 \
            typename = all_const_indexable<Left, Right>>                   \
  HD_INLINE auto operator op(const Left& l, const Right& r) {              \
    return make_binary_exp(l, r, std::name<typename Left::value_t>());     \
  }                                                                        \
                                                                           \
  template <typename Left, typename = is_const_indexable<Left>>            \
  HD_INLINE auto operator op(const Left& l, typename Left::value_t r) {    \
    return make_binary_exp(                                                \
        l, const_exp_t<typename Left::value_t, typename Left::idx_t>(r),   \
        std::name<typename Left::value_t>());                              \
  }                                                                        \
                                                                           \
  template <typename Right, typename = is_const_indexable<Right>>          \
  HD_INLINE auto operator op(typename Right::value_t l, const Right& r) {  \
    return make_binary_exp(                                                \
        const_exp_t<typename Right::value_t, typename Right::idx_t>(l), r, \
        std::name<typename Right::value_t>());                             \
  }

INSTANTIATE_BINARY_OPS(plus, +)
INSTANTIATE_BINARY_OPS(minus, -)
INSTANTIATE_BINARY_OPS(multiplies, *)
INSTANTIATE_BINARY_OPS(divides, /)

template <typename Arg, typename = is_const_indexable<Arg>>
HD_INLINE auto
operator-(const Arg& arg) {
  return make_unary_exp(arg, std::negate<typename Arg::value_t>());
}

}  // namespace Aperture

#include "core/multi_array.hpp"

namespace Aperture {

#define INSTANTIATE_MULTIARRAY_BINARY_OPS(name, op)                      \
  template <typename T, int Rank, typename Idx_t, typename Right>        \
  HD_INLINE auto operator op(const multi_array<T, Rank, Idx_t>& array,   \
                             const Right& r) {                           \
    return operator op(array.host_ndptr_const(), r);                     \
  }                                                                      \
                                                                         \
  template <typename T, int Rank, typename Idx_t, typename Left>         \
  HD_INLINE auto operator op(const Left& l,                              \
                             const multi_array<T, Rank, Idx_t>& array) { \
    return operator op(l, array.host_ndptr_const());                     \
  }                                                                      \
                                                                         \
  template <typename T, int Rank, typename Idx_t>                        \
  HD_INLINE auto operator op(const multi_array<T, Rank, Idx_t>& v1,      \
                             const multi_array<T, Rank, Idx_t>& v2) {    \
    return operator op(v1.host_ndptr_const(), v2.host_ndptr_const());    \
  }

INSTANTIATE_MULTIARRAY_BINARY_OPS(plus, +);
INSTANTIATE_MULTIARRAY_BINARY_OPS(minus, -);
INSTANTIATE_MULTIARRAY_BINARY_OPS(multiplies, *);
INSTANTIATE_MULTIARRAY_BINARY_OPS(divides, /);

}  // namespace Aperture

#endif  // __MULTI_ARRAY_EXP_H_
