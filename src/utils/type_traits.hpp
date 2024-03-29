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

#include <type_traits>

namespace Aperture {

template <int A, int B>
struct less_than {
  enum { value = A < B };
};

template <class T>
struct type_identity {
  using type = T;
};

template <class T>
using type_identity_t = typename type_identity<T>::type;

template <typename T>
using is_integral_t = typename std::enable_if<std::is_integral<T>::value>::type;

// This implementation of conjunction is taken from
// https://www.fluentcpp.com/2019/01/25/variadic-number-function-parameters-type/
// In c++17 there is std::conjunction, but this implementation is to keep
// compatibility with c++14 which CUDA uses.
template <bool...>
struct bool_pack {};

template <class... Ts>
using conjunction =
    std::is_same<bool_pack<true, Ts::value...>, bool_pack<Ts::value..., true>>;

// This disjunction implementation is from cppreference.com
template <class...>
struct disjunction : std::false_type {};
template <class B1>
struct disjunction<B1> : B1 {};
template <class B1, class... Bn>
struct disjunction<B1, Bn...>
    : std::conditional_t<B1::value, B1, disjunction<Bn...>> {};

template <typename T, typename U>
using is_convertible_to =
    typename std::enable_if<std::is_convertible<U, T>::value>::type;

// This checks if a given type pack Ts are all convertible to T
template <typename T, typename... Ts>
using all_convertible_to = typename std::enable_if<
    conjunction<std::is_convertible<Ts, T>...>::value>::type;

// Check for member function with given name AND signature.
#define CREATE_MEMBER_FUNC_CHECK(func_name)                                   \
  namespace traits {                                                          \
  template <typename, typename T>                                             \
  struct has_##func_name {                                                    \
    static_assert(std::integral_constant<T, false>::value,                    \
                  "Second template parameter needs to be of function type."); \
  };                                                                          \
                                                                              \
  template <typename C, typename Ret, typename... Args>                       \
  struct has_##func_name<C, Ret(Args...)> {                                   \
   private:                                                                   \
    template <typename T>                                                     \
    static constexpr auto check(T*) -> typename std::is_same<                 \
        decltype(std::declval<T>().func_name(std::declval<Args>()...)),       \
        Ret      /* ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ */ \
        >::type; /* attempt to call it and see if the return type is correct  \
                  */                                                          \
                                                                              \
    template <typename>                                                       \
    static constexpr std::false_type check(...);                              \
                                                                              \
    typedef decltype(check<C>(0)) type;                                       \
                                                                              \
   public:                                                                    \
    static constexpr bool value = type::value;                                \
  };                                                                          \
  }

#define CREATE_MEMBER_FUNC_CALL(func_name)                         \
  CREATE_MEMBER_FUNC_CHECK(func_name);                             \
                                                                   \
  namespace traits {                                               \
  template <typename T, typename... Args>                          \
  auto call_##func_name##_if_exists(T& t, Args... args)            \
      -> std::enable_if_t<has_update<T, void(Args...)>::value> {   \
    t.func_name(args...);                                          \
  }                                                                \
                                                                   \
  template <typename T, typename... Args>                          \
  auto call_##func_name##_if_exists(T& t, Args... args)            \
      -> std::enable_if_t<!has_update<T, void(Args...)>::value> {} \
  }

}  // namespace Aperture
