#ifndef __FINITE_DIFF_HELPER_H_
#define __FINITE_DIFF_HELPER_H_

#include "core/cuda_control.h"
#include "utils/vec.hpp"
#include "utils/stagger.h"

namespace Aperture {

template <int Dir, typename PtrType>
HD_INLINE typename PtrType::value_t
diff(const PtrType& p, const typename PtrType::idx_t& idx, stagger_t stagger) {
  return p[idx.inc<Dir>(1 - stagger[Dir])] - p[idx.dec<Dir>(stagger[Dir])];
}

template <int Dim>
struct finite_diff;

template <>
struct finite_diff<1> {
  template <typename VecType, typename Idx_t, typename Stagger>
  HD_INLINE static Scalar curl0(const VecType& f, const Idx_t& idx, const Stagger& st) {
    return 0.0;
  }

  template <typename VecType, typename Idx_t, typename Stagger>
  HD_INLINE static Scalar curl1(const VecType& f, const Idx_t& idx, const Stagger& st) {
    return -diff<0>(f[2], idx, st[2]);
  }

  template <typename VecType, typename Idx_t, typename Stagger>
  HD_INLINE static Scalar curl2(const VecType& f, const Idx_t& idx, const Stagger& st) {
    return diff<0>(f[1], idx, st[1]);
  }
};

template <>
struct finite_diff<2> {
  template <typename VecType, typename Idx_t, typename Stagger>
  HD_INLINE static Scalar curl0(const VecType& f, const Idx_t& idx, const Stagger& st) {
    return diff<1>(f[2], idx, st[2]);
  }

  template <typename VecType, typename Idx_t, typename Stagger>
  HD_INLINE static Scalar curl1(const VecType& f, const Idx_t& idx, const Stagger& st) {
    return -diff<0>(f[2], idx, st[2]);
  }

  template <typename VecType, typename Idx_t, typename Stagger>
  HD_INLINE static Scalar curl2(const VecType& f, const Idx_t& idx, const Stagger& st) {
    return diff<0>(f[1], idx, st[1]) - diff<1>(f[0], idx, st[0]);
  }
};

template <>
struct finite_diff<3> {
  template <typename VecType, typename Idx_t, typename Stagger>
  HD_INLINE static Scalar curl0(const VecType& f, const Idx_t& idx, const Stagger& st) {
    return diff<1>(f[2], idx, st[2]) - diff<2>(f[1], idx, st[1]);
  }

  template <typename VecType, typename Idx_t, typename Stagger>
  HD_INLINE static Scalar curl1(const VecType& f, const Idx_t& idx, const Stagger& st) {
    return diff<2>(f[0], idx, st[0]) - diff<0>(f[2], idx, st[2]);
  }

  template <typename VecType, typename Idx_t, typename Stagger>
  HD_INLINE static Scalar curl2(const VecType& f, const Idx_t& idx, const Stagger& st) {
    return diff<0>(f[1], idx, st[1]) - diff<1>(f[0], idx, st[0]);
  }
};


}

#endif // __FINITE_DIFF_HELPER_H_
