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

#ifndef __FOR_EACH_DUAL_H_
#define __FOR_EACH_DUAL_H_

#include "visit_struct/visit_struct.hpp"

namespace Aperture {

template <int N, typename T, typename U, typename Op>
struct iterate_struct {
  static void run(T& t, U& u, const Op& op) {
    op(visit_struct::get<N>(t), visit_struct::get<N>(u));
    iterate_struct<N - 1, T, U, Op>{}.run(t, u, op);
  }
  static void run_with_name(T& t, U& u, const Op& op) {
    op(visit_struct::get_name<N>(t), visit_struct::get<N>(t),
       visit_struct::get<N>(u));
    iterate_struct<N - 1, T, U, Op>{}.run_with_name(t, u, op);
  }
};

template <typename T, typename U, typename Op>
struct iterate_struct<-1, T, U, Op> {
  static void run(T& t, U& u, const Op& op) {}
  static void run_with_name(T& t, U& u, const Op& op) {}
};

template <typename U1, typename U2, typename Op>
void
for_each_double(U1& u1, U2& u2, const Op& op) {
  iterate_struct<visit_struct::field_count<U1>() - 1, U1, U2, Op>::run(
      u1, u2, op);
}

template <typename U1, typename U2, typename Op>
void
for_each_double_with_name(U1& u1, U2& u2, const Op& op) {
  iterate_struct<visit_struct::field_count<U1>() - 1, U1, U2,
                 Op>::run_with_name(u1, u2, op);
}

}  // namespace Aperture

#endif  // __FOR_EACH_DUAL_H_
