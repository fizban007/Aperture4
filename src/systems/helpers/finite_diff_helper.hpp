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

#ifndef __FINITE_DIFF_HELPER_H_
#define __FINITE_DIFF_HELPER_H_

#include "core/cuda_control.h"
#include "core/grid.hpp"
#include "utils/stagger.h"
#include "utils/vec.hpp"

namespace Aperture {

template <int Order>
struct order_tag {};

// First order derivatives
template <int Dir, typename PtrType>
HD_INLINE auto
diff(const PtrType& p, const typename PtrType::idx_t& idx, stagger_t stagger,
     order_tag<2> tag) {
  return p[idx.template inc<Dir>(stagger[Dir])] -
         p[idx.template dec<Dir>(1 - stagger[Dir])];
}

template <int Dir, typename PtrType>
HD_INLINE auto
diff(const PtrType& p, const typename PtrType::idx_t& idx, stagger_t stagger,
     order_tag<4> tag) {
  return (-p[idx.template inc<Dir>(stagger[Dir] + 1)] +
          27.0f * p[idx.template inc<Dir>(stagger[Dir])] -
          27.0f * p[idx.template dec<Dir>(1 - stagger[Dir])] +
          p[idx.template dec<Dir>(2 - stagger[Dir])]) /
         24.0f;
}

template <int Dir, typename PtrType>
HD_INLINE auto
diff(const PtrType& p, const typename PtrType::idx_t& idx, stagger_t stagger,
     order_tag<6> tag) {
  // TODO: Implement this!
  // #pragma message "6th order 1st derivatives not implemented!"
  return 0.0f;
}

// Second order derivatives
template <int Dir, typename PtrType>
HD_INLINE auto
diff2(const PtrType& p, const typename PtrType::idx_t& idx, order_tag<2> tag) {
  return p[idx.template inc<Dir>()] - 2.0f * p[idx] +
         p[idx.template dec<Dir>()];
}

template <int Dir, typename PtrType>
HD_INLINE auto
diff2(const PtrType& p, const typename PtrType::idx_t& idx, order_tag<4> tag) {
  return (-p[idx.template inc<Dir>(2)] + 16.0f * p[idx.template inc<Dir>(1)] -
          30.0f * p[idx] + 16.0f * p[idx.template dec<Dir>(1)] -
          p[idx.template dec<Dir>(2)]) /
         12.0f;
}

template <int Dir, typename PtrType>
HD_INLINE typename PtrType::value_t
diff2(const PtrType& p, const typename PtrType::idx_t& idx, order_tag<6> tag) {
  // TODO: Implement this!
  // #pragma message "6th order 2nd derivatives not implemented!"
  return 0.0f;
}

template <int Dim, int Order = 2>
struct finite_diff;

template <int Order>
struct finite_diff<1, Order> {
  template <typename VecType, typename Idx_t, typename Stagger, typename value_t>
  HD_INLINE static auto div(const VecType& f, const Idx_t& idx,
                            const Stagger& st, const Grid<1, value_t>& g) {
    return diff<0>(f[0], idx, st[0], order_tag<Order>{}) * g.inv_delta[0];
  }

  template <typename VecType, typename Idx_t, typename Stagger, typename value_t>
  HD_INLINE static auto curl0(const VecType& f, const Idx_t& idx,
                              const Stagger& st, const Grid<1, value_t>& g) {
    return 0.0;
  }

  template <typename VecType, typename Idx_t, typename Stagger, typename value_t>
  HD_INLINE static auto curl1(const VecType& f, const Idx_t& idx,
                              const Stagger& st, const Grid<1, value_t>& g) {
    return -diff<0>(f[2], idx, st[2], order_tag<Order>{}) * g.inv_delta[0];
  }

  template <typename VecType, typename Idx_t, typename Stagger, typename value_t>
  HD_INLINE static auto curl2(const VecType& f, const Idx_t& idx,
                              const Stagger& st, const Grid<1, value_t>& g) {
    return diff<0>(f[1], idx, st[1], order_tag<Order>{}) * g.inv_delta[0];
  }

  template <typename PtrType, typename Idx_t, typename value_t>
  HD_INLINE static auto laplacian(const PtrType& f, const Idx_t& idx,
                                  const Grid<1, value_t>& g) {
    return diff2<0>(f, idx, order_tag<Order>{}) * square(g.inv_delta[0]);
  }
};

template <int Order>
struct finite_diff<2, Order> {
  template <typename VecType, typename Idx_t, typename Stagger, typename value_t>
  HD_INLINE static auto div(const VecType& f, const Idx_t& idx,
                            const Stagger& st, const Grid<2, value_t>& g) {
    return diff<0>(f[0], idx, st[0], order_tag<Order>{}) * g.inv_delta[0] +
           diff<1>(f[1], idx, st[1], order_tag<Order>{}) * g.inv_delta[1];
  }

  template <typename VecType, typename Idx_t, typename Stagger, typename value_t>
  HD_INLINE static auto curl0(const VecType& f, const Idx_t& idx,
                              const Stagger& st, const Grid<2, value_t>& g) {
    return diff<1>(f[2], idx, st[2], order_tag<Order>{}) * g.inv_delta[1];
  }

  template <typename VecType, typename Idx_t, typename Stagger, typename value_t>
  HD_INLINE static auto curl1(const VecType& f, const Idx_t& idx,
                              const Stagger& st, const Grid<2, value_t>& g) {
    return -diff<0>(f[2], idx, st[2], order_tag<Order>{}) * g.inv_delta[0];
  }

  template <typename VecType, typename Idx_t, typename Stagger, typename value_t>
  HD_INLINE static auto curl2(const VecType& f, const Idx_t& idx,
                              const Stagger& st, const Grid<2, value_t>& g) {
    return diff<0>(f[1], idx, st[1], order_tag<Order>{}) * g.inv_delta[0] -
           diff<1>(f[0], idx, st[0], order_tag<Order>{}) * g.inv_delta[1];
  }

  template <typename PtrType, typename Idx_t, typename value_t>
  HD_INLINE static auto laplacian(const PtrType& f, const Idx_t& idx,
                                  const Grid<2, value_t>& g) {
    return diff2<0>(f, idx, order_tag<Order>{}) * square(g.inv_delta[0]) +
           diff2<1>(f, idx, order_tag<Order>{}) * square(g.inv_delta[1]);
  }
};

template <int Order>
struct finite_diff<3, Order> {
  template <typename VecType, typename Idx_t, typename Stagger, typename value_t>
  HD_INLINE static auto div(const VecType& f, const Idx_t& idx,
                            const Stagger& st, const Grid<3, value_t>& g) {
    return diff<0>(f[0], idx, st[0], order_tag<Order>{}) * g.inv_delta[0] +
           diff<1>(f[1], idx, st[1], order_tag<Order>{}) * g.inv_delta[1] +
           diff<2>(f[2], idx, st[2], order_tag<Order>{}) * g.inv_delta[2];
  }

  template <typename VecType, typename Idx_t, typename Stagger, typename value_t>
  HD_INLINE static auto curl0(const VecType& f, const Idx_t& idx,
                              const Stagger& st, const Grid<3, value_t>& g) {
    return diff<1>(f[2], idx, st[2], order_tag<Order>{}) * g.inv_delta[1] -
           diff<2>(f[1], idx, st[1], order_tag<Order>{}) * g.inv_delta[2];
  }

  template <typename VecType, typename Idx_t, typename Stagger, typename value_t>
  HD_INLINE static auto curl1(const VecType& f, const Idx_t& idx,
                              const Stagger& st, const Grid<3, value_t>& g) {
    return diff<2>(f[0], idx, st[0], order_tag<Order>{}) * g.inv_delta[2] -
           diff<0>(f[2], idx, st[2], order_tag<Order>{}) * g.inv_delta[0];
  }

  template <typename VecType, typename Idx_t, typename Stagger, typename value_t>
  HD_INLINE static auto curl2(const VecType& f, const Idx_t& idx,
                              const Stagger& st, const Grid<3, value_t>& g) {
    return diff<0>(f[1], idx, st[1], order_tag<Order>{}) * g.inv_delta[0] -
           diff<1>(f[0], idx, st[0], order_tag<Order>{}) * g.inv_delta[1];
  }

  template <typename PtrType, typename Idx_t, typename value_t>
  HD_INLINE static auto laplacian(const PtrType& f, const Idx_t& idx,
                                  const Grid<3, value_t>& g) {
    return diff2<0>(f, idx, order_tag<Order>{}) * square(g.inv_delta[0]) +
           diff2<1>(f, idx, order_tag<Order>{}) * square(g.inv_delta[1]) +
           diff2<2>(f, idx, order_tag<Order>{}) * square(g.inv_delta[2]);
  }
};

}  // namespace Aperture

#endif  // __FINITE_DIFF_HELPER_H_
