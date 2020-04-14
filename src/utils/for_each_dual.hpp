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
