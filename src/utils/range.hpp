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

  struct step_range_proxy {
    struct iter : detail::range_iter_base<T> {
      HD_INLINE
      iter(T current, T step)
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
      T step;
    };

    HD_INLINE
    step_range_proxy(T begin, T end, T step)
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

  HD_INLINE
  step_range_proxy step(T step) { return {*begin_, *end_, step}; }

  HD_INLINE
  iter begin() const { return begin_; }

  HD_INLINE
  iter end() const { return end_; }

 private:
  iter begin_;
  iter end_;
};

template <typename T>
HD_INLINE range_proxy<T>
range(T begin, T end) {
  return {begin, end};
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

template<typename T>
using step_range = typename range_proxy<T>::step_range_proxy;

#ifdef __CUDA_ARCH__

template <typename T>
__device__ __forceinline__
step_range<T> grid_stride_range(T begin, T end) {
    return range(T(begin + blockDim.x * blockIdx.x + threadIdx.x), end)
      .step(gridDim.x * blockDim.x);
}

#endif

}  // namespace Aperture

#endif  // ndef __RANGE_HPP_
