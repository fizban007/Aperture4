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

#ifndef __RANGE_HPP_
#define __RANGE_HPP_

#include "core/cuda_control.h"
#include <iterator>
#include <type_traits>

namespace Aperture {

namespace detail {

template <typename T>
struct range_iter_base : std::iterator<std::input_iterator_tag, T> {
  HD_INLINE
  range_iter_base(T current) : current(current) {}

  HD_INLINE
  T operator*() const { return current; }

  HD_INLINE
  T const* operator->() const { return &current; }

  HD_INLINE
  range_iter_base& operator++() {
    ++current;
    return *this;
  }

  HD_INLINE
  range_iter_base operator++(int) {
    auto copy = *this;
    ++*this;
    return copy;
  }

  HD_INLINE
  int operator-(const range_iter_base& it) {
    return current - it.current;
  }

  HD_INLINE
  range_iter_base operator+=(int n) {
    current += n;
    return *this;
  }

  HD_INLINE
  bool operator==(range_iter_base const& other) const {
    return current == other.current;
  }

  HD_INLINE
  bool operator!=(range_iter_base const& other) const {
    return not(*this == other);
  }

 protected:
  T current;
};

}  // namespace detail

template <typename T>
struct range_proxy {
  struct iter : detail::range_iter_base<T> {
    HD_INLINE
    iter(T current) : detail::range_iter_base<T>(current) {}
  };

  template <typename U>
  struct step_range_proxy {
    struct iter : detail::range_iter_base<T> {
      HD_INLINE
      iter(T current, U step)
          : detail::range_iter_base<T>(current), step(step) {}

      using detail::range_iter_base<T>::current;

      HD_INLINE
      iter& operator++() {
        current += step;
        return *this;
      }

      HD_INLINE
      iter operator++(int) {
        auto copy = *this;
        ++*this;
        return copy;
      }

      HD_INLINE
      int operator-(const iter& it) {
        return current - it.current;
      }

      // Loses commutativity. Iterator-based ranges are simply broken.
      // :-(
      HD_INLINE
      bool operator==(iter const& other) const {
        return step > 0 ? current >= other.current
                        : current < other.current;
      }

      HD_INLINE
      bool operator!=(iter const& other) const {
        return not(*this == other);
      }

     private:
      U step;
    };

    HD_INLINE
    step_range_proxy(T begin, T end, U step)
        : begin_(begin, step), end_(end, step) {}

    HD_INLINE
    iter begin() const { return begin_; }

    HD_INLINE
    iter end() const { return end_; }

   private:
    iter begin_;
    iter end_;
  };

  HD_INLINE
  range_proxy(T begin, T end) : begin_(begin), end_(end) {}

  template <typename U>
  HD_INLINE
  step_range_proxy<U> step(U step) { return {*begin_, *end_, step}; }

  HD_INLINE
  iter begin() const { return begin_; }

  HD_INLINE
  iter end() const { return end_; }

 private:
  iter begin_;
  iter end_;
};

template <typename U, typename T>
HD_INLINE range_proxy<T>
range(U begin, T end) {
  return {(T)begin, end};
}

namespace traits {

template <typename C>
struct has_size {
  template <typename T>
  static constexpr auto check(T*) -> typename std::is_integral<
      decltype(std::declval<T const>().size())>::type;

  template <typename>
  static constexpr auto check(...) -> std::false_type;

  using type = decltype(check<C>(0));
  static constexpr bool value = type::value;
};

}  // namespace traits

template <typename C, typename = typename std::enable_if<
                          traits::has_size<C>::value>>
HD_INLINE auto
indices(C const& cont) -> range_proxy<decltype(cont.size())> {
  return {0, cont.size()};
}

template <typename T, std::size_t N>
HD_INLINE range_proxy<std::size_t>
indices(T (&)[N]) {
  return {0, N};
}

template <typename T>
HD_INLINE range_proxy<typename std::initializer_list<T>::size_type>
indices(std::initializer_list<T>&& cont) {
  return {0, cont.size()};
}

template<typename T, typename U>
using step_range = typename range_proxy<T>::template step_range_proxy<U>;

#ifdef __CUDACC__

template <typename U, typename T>
__device__ __forceinline__
step_range<T, int> grid_stride_range(U begin, T end) {
    return range(T(begin + blockDim.x * blockIdx.x + threadIdx.x), end)
      .step(int(gridDim.x * blockDim.x));
}

#endif

}  // namespace Aperture

#endif  // ndef __RANGE_HPP_
