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

// This is taken from
// https://www.fluentcpp.com/2019/01/25/variadic-number-function-parameters-type/
template <bool...>
struct bool_pack {};
template <class... Ts>
using conjunction =
    std::is_same<bool_pack<true, Ts::value...>, bool_pack<Ts::value..., true>>;
template <typename T, typename... Ts>
using all_convertible_to = typename std::enable_if<
    conjunction<std::is_convertible<Ts, T>...>::value>::type;

}  // namespace Aperture

#endif  // __TYPE_TRAITS_H_
