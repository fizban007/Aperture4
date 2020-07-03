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

#ifndef __CORE_GRID_H_
#define __CORE_GRID_H_

#include "cuda_control.h"
#include "typedefs_and_constants.h"
#include "utils/vec.hpp"
#include "utils/index.hpp"
#include "utils/stagger.h"
#include <type_traits>

namespace Aperture {

template <int Dim>
struct Grid {
  uint32_t dims[Dim];  //!< Dimensions of the grid of each direction
  int guard[Dim];      //!< Number of guard cells at either end of each
                       //!< direction
  int skirt[Dim];

  Scalar delta[Dim];  //!< Grid spacing on each direction (spacing in
                      //!< coordinate space)
  Scalar inv_delta[Dim];
  Scalar lower[Dim];  //!< Lower limit of the grid on each direction
  Scalar sizes[Dim];  //!< Size of the grid in coordinate space

  // int tileSize[Dim];
  int offset[Dim];

  HOST_DEVICE Grid() = default;
  HOST_DEVICE Grid(const Grid& grid) = default;

  HOST_DEVICE Grid& operator=(const Grid& grid) = default;

  ///  Reduced dimension in one direction.
  ///
  ///  Reduced dimension means the total size of the grid minus the
  ///  guard cells in both ends. This function is only defined for N >=
  ///  0 and N < Dim.
  template <int N>
      HD_INLINE std::enable_if_t <
      N<Dim, uint32_t> reduced_dim() const {
    return (dims[N] - 2 * skirt[N]);
  }

  HD_INLINE uint32_t reduced_dim(int i) const { return (dims[i] - 2 * skirt[i]); }

  ///  Coordinate of a point inside cell n in dimension N.
  ///
  ///  This function applies to field points. Stagger = false means
  ///  field is defined at cell center, while stagger = true means field
  ///  defined at cell boundary at the end.
  ///
  ///  This calculation is assuming boundary located at the interface
  ///  between guard cells and physical cells. The function is only
  ///  defined for N >= 0 and N < Dim.
  template <int N>
      HD_INLINE std::enable_if_t <
      N<Dim, Scalar> pos(int n, bool stagger) const {
    return pos<N>(n, (int)stagger);
  }

  ///  Coordinate of a point inside cell n in dimension i.
  ///
  ///  This function applies to field points. Stagger = 0 means field is
  ///  defined at cell center, while stagger = 1 means field defined at
  ///  cell boundary at the end.
  ///
  ///  This calculation is assuming boundary located at the interface
  ///  between guard cells and physical cells. The function is only
  ///  defined for N >= 0 and N < DIM.
  template <int N>
      HD_INLINE std::enable_if_t <
      N<Dim, Scalar> pos(int n, int stagger) const {
    return pos<N>(n, (Scalar)(0.5 - 0.5 * stagger));
  }

  ///  Coordinate of a point inside cell n in dimension i.
  ///
  ///  This function applies to particles. pos_in_cell is the relative
  ///  position of the particle in the cell and varies from 0.0 to 1.0.
  ///
  ///  This calculation is assuming boundary located at the interface
  ///  between guard cells and physical cells. The function is only
  ///  defined for N >= 0 and N < Dim. When N >= Dim, it simply returns
  ///  pos_in_cell
  template <int N>
      HD_INLINE std::enable_if_t <
      N<Dim, Scalar> pos(int n, Scalar pos_in_cell) const {
    return (lower[N] + delta[N] * (n - skirt[N] + pos_in_cell));
  }

  template <int N>
  HD_INLINE std::enable_if_t<N >= Dim, Scalar> pos(
      int n, Scalar pos_in_cell) const {
    return pos_in_cell;
  }

  HD_INLINE Scalar pos(int i, int n, Scalar pos_in_cell) const {
    if (i < Dim)
      return lower[i] + delta[i] * (n - skirt[i] + pos_in_cell);
    else
      return pos_in_cell;
  }

  HD_INLINE Scalar pos(int i, int n, bool stagger) const {
    return pos(i, n, (Scalar)(stagger ? 0.0 : 0.5));
  }

  template <int N>
  HD_INLINE Scalar pos(const index_t<Dim>& idx,
                       Scalar pos_in_cell) const {
    return pos<N>(idx[N], pos_in_cell);
  }

  template <int N>
  HD_INLINE Scalar pos(const index_t<Dim>& idx,
                       stagger_t st) const {
    return pos<N>(idx[N], st[N]);
  }

  HD_INLINE vec_t<Scalar, Dim> pos_global(const index_t<Dim>& idx,
                                          const vec_t<Scalar, Dim>& rel_pos) {
    vec_t<Scalar, Dim> result;
#pragma unroll
    for (int i = 0; i < Dim; i++) {
      result[i] = pos(i, idx[i], rel_pos[i]);
    }
    return result;
  }

  ///  Find the zone the cell belongs to (for communication purposes)
  HD_INLINE int find_zone(const index_t<Dim>& pos) const {
    auto z = index_t<Dim>{};
    auto ext = extent_t<Dim>{};
#pragma unroll
    for (int i = 0; i < Dim; i++) {
      z[i] = (pos[i] >= skirt[i]) + (pos[i] >= (dims[i] - skirt[i]));
      ext[i] = 3;
    }

    // For now we always use col major for zone index
    idx_col_major_t<Dim> idx(z, ext);
    return idx.linear;
  }

  HD_INLINE bool is_in_bound(const index_t<Dim>& idx) const {
#pragma unroll
    for (int i = 0; i < Dim; i++) {
      if (idx[i] < skirt[i] || idx[i] >= dims[i] - skirt[i])
        return false;
    }
    return true;
  }

  template <typename... Args>
  HD_INLINE bool is_in_bound(Args... args) const {
    return is_in_bound(index(args...));
  }

  HD_INLINE bool is_in_grid(const index_t<Dim>& idx) const {
#pragma unroll
    for (int i = 0; i < Dim; i++) {
      if (idx[i] >= dims[i])
        return false;
    }
    return true;
  }

  template <typename... Args>
  HD_INLINE bool is_in_grid(Args... args) const {
    return is_in_grid(index(args...));
  }

  HD_INLINE extent_t<Dim> extent() const {
    extent_t<Dim> result;
#pragma unroll
    for (int i = 0; i < Dim; i++)
      result[i] = dims[i];
    return result;
  }

  HD_INLINE extent_t<Dim> extent_less() const {
    extent_t<Dim> result;
#pragma unroll
    for (int i = 0; i < Dim; i++)
      result[i] = reduced_dim(i);
    return result;
  }

  HD_INLINE index_t<Dim> guards() const {
    index_t<Dim> result;
#pragma unroll
    for (int i = 0; i < Dim; i++) {
      result[i] = guard[i];
    }
    return result;
  }

  HD_INLINE index_t<Dim> offsets() const {
    index_t<Dim> result;
#pragma unroll
    for (int i = 0; i < Dim; i++) {
      result[i] = offset[i];
    }
    return result;
  }

};

}  // namespace Aperture

#endif  // __CORE_GRID_H_
