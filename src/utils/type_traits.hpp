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

#ifndef __TYPE_TRAITS_H_
#define __TYPE_TRAITS_H_

#include <type_traits>

namespace Aperture {

// This implementation of conjunction is taken from
// https://www.fluentcpp.com/2019/01/25/variadic-number-function-parameters-type/
// In c++17 there is std::conjunction, but this implementation is to keep
// compatibility with c++14 which CUDA uses.
template <bool...>
struct bool_pack {};

template <class... Ts>
using conjunction =
    std::is_same<bool_pack<true, Ts::value...>, bool_pack<Ts::value..., true>>;

template <typename T, typename... Ts>
using all_convertible_to = typename std::enable_if<
    conjunction<std::is_convertible<Ts, T>...>::value>::type;

// This is a template struct to check if a type is indexable using an idx type
template<typename Type>
struct is_indexable {
private:
    template<typename T>
    static constexpr auto check(T*)
    -> typename
        std::is_same<
            decltype( std::declval<T>()[std::declval<typename Type::idx_t>()] ),
            typename Type::value_t& // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        >::type;  // attempt to call it and see if the return type is correct

    template<typename>
    static constexpr std::false_type check(...);

    typedef decltype(check<Type>(0)) result_t;

public:
    static constexpr bool value = result_t::value;
};

// This is a template struct to check if a type is indexable using an idx type
template<typename Type>
struct is_const_indexable {
private:
    template<typename T>
    static constexpr auto check(T*)
    -> typename
        std::is_same<
            decltype( std::declval<const T>()[std::declval<typename Type::idx_t>()] ),
            typename Type::value_t // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        >::type;  // attempt to call it and see if the return type is correct

    template<typename>
    static constexpr std::false_type check(...);

    typedef decltype(check<Type>(0)) result_t;

public:
    static constexpr bool value = result_t::value;
};



}  // namespace Aperture

#endif  // __TYPE_TRAITS_H_
