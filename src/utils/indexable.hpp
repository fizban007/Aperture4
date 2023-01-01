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

#include "utils/type_traits.hpp"
#include "utils/vec.hpp"

namespace Aperture {

// This is a series of template structs to check if a type is indexable using an
// idx type

template <typename Type, typename = typename Type::idx_t>
// struct is_indexable<Type, typename Type::idx_t> {
struct is_host_indexable {
 private:
  template <typename T>
  static constexpr auto check(T*) -> typename std::is_same<
      decltype(std::declval<T>().at(index_t<Type::idx_t::dim>{})),
      typename Type::value_t&  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      >::type;  // attempt to call it and see if the return type is correct

  template <typename>
  static constexpr std::false_type check(...);

  typedef decltype(check<Type>(0)) result_t;

 public:
  static constexpr bool value = result_t::value;
};

template <typename Type, typename = typename Type::idx_t>
// struct is_indexable<Type, typename Type::idx_t> {
struct is_dev_indexable {
 private:
  template <typename T>
  static constexpr auto check(T*) -> typename std::is_same<
      decltype(std::declval<T>().at_dev(
          std::declval<index_t<Type::idx_t::dim>>())),
      typename Type::value_t&  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      >::type;  // attempt to call it and see if the return type is correct

  template <typename>
  static constexpr std::false_type check(...);

  typedef decltype(check<Type>(0)) result_t;

 public:
  static constexpr bool value = result_t::value;
};

template <typename Type, typename = typename Type::idx_t>
// struct is_indexable<Type, typename Type::idx_t> {
struct is_plain_indexable {
 private:
  template <typename T>
  static constexpr auto check(T*) -> typename std::is_same<
      decltype(std::declval<T>()[std::declval<typename Type::idx_t>()]),
      typename Type::value_t&  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      >::type;  // attempt to call it and see if the return type is correct

  template <typename>
  static constexpr std::false_type check(...);

  typedef decltype(check<Type>(0)) result_t;

 public:
  static constexpr bool value = result_t::value;
};

// This is a template struct to check if a type is const indexable using an idx
// type template <typename Type, class Idx_t = void> struct is_const_indexable;

template <typename Type, typename = typename Type::idx_t>
struct is_plain_const_indexable {
 private:
  template <typename T>
  static constexpr auto check(T*) -> typename std::is_same<
      decltype(std::declval<const T>()[std::declval<typename Type::idx_t>()]),
      typename Type::value_t  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      >::type;  // attempt to call it and see if the return type is correct

  template <typename>
  static constexpr std::false_type check(...);

  typedef decltype(check<Type>(0)) result_t;

 public:
  static constexpr bool value = result_t::value;
};

template <typename Type, typename = typename Type::idx_t>
struct is_host_const_indexable {
 private:
  template <typename T>
  static constexpr auto check(T*) -> typename std::is_same<
      decltype(std::declval<const T>().at(
          std::declval<index_t<Type::idx_t::dim>>())),
      typename Type::value_t  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      >::type;  // attempt to call it and see if the return type is correct

  template <typename>
  static constexpr std::false_type check(...);

  typedef decltype(check<Type>(0)) result_t;

 public:
  static constexpr bool value = result_t::value;
};

template <typename Type, typename = typename Type::idx_t>
struct is_dev_const_indexable {
 private:
  template <typename T>
  static constexpr auto check(T*) -> typename std::is_same<
      decltype(std::declval<const T>().at_dev(
          std::declval<index_t<Type::idx_t::dim>>())),
      typename Type::value_t  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      >::type;  // attempt to call it and see if the return type is correct

  template <typename>
  static constexpr std::false_type check(...);

  typedef decltype(check<Type>(0)) result_t;

 public:
  static constexpr bool value = result_t::value;
};

template <typename T>
using is_const_indexable =
    disjunction<is_dev_const_indexable<T>, is_host_const_indexable<T>,
                is_plain_const_indexable<T>>;

template <typename T>
using is_indexable = disjunction<is_dev_indexable<T>, is_host_indexable<T>,
                                 is_plain_indexable<T>>;

// This checks whether a given pack of types are all const indexable
template <typename... Ts>
using all_const_indexable = typename std::enable_if<
    conjunction<is_const_indexable<Ts>...>::value>::type;

}  // namespace Aperture
