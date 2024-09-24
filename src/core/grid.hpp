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

#include "cuda_control.h"
#include "typedefs_and_constants.h"
// #include "utils/index.hpp"
#include "utils/stagger.h"
#include "utils/vec.hpp"
#include <type_traits>

namespace Aperture {

template <int Dim, typename value_t>
struct Grid {
  uint32_t dims[Dim];  //!< Dimensions of the grid of each direction
  uint32_t N[Dim];     //!< Physical dim of the grid of each direction
  int guard[Dim];      //!< Number of guard cells at either end of each
                       //!< direction

  value_t delta[Dim];  //!< Grid spacing on each direction (spacing in
                       //!< coordinate space)
  value_t inv_delta[Dim];

  value_t lower[Dim];  //!< Lower limit of the grid on each direction
  value_t sizes[Dim];  //!< Size of the grid in coordinate space

  int offset[Dim];  //!< The offset of this grid in the global domain
                    //!< decomposition

  HOST_DEVICE Grid() = default;
  HOST_DEVICE Grid(const Grid& grid) = default;

  HOST_DEVICE Grid& operator=(const Grid& grid) = default;

  ///  Reduced dimension in one direction.
  ///
  ///  Reduced dimension means the total size of the grid minus the
  ///  guard cells in both ends. This function is only defined for N >=
  ///  0 and N < Dim.
  template <int n>
      HD_INLINE std::enable_if_t < n<Dim, uint32_t> reduced_dim() const {
    // return (dims[N] - 2 * guard[N]);
    return N[n];
  }

  HD_INLINE uint32_t reduced_dim(int i) const {
    // return (dims[i] - 2 * guard[i]);
    return N[i];
  }

  ///  Coordinate of a point inside cell n in dimension N.
  ///
  ///  This function applies to field points. Stagger = false means
  ///  field is defined at cell center, while stagger = true means field
  ///  defined at cell boundary at the end.
  ///
  ///  This calculation is assuming boundary located at the interface
  ///  between guard cells and physical cells. The function is only
  ///  defined for n >= 0 and n < Dim.
  template <int n>
      HD_INLINE std::enable_if_t <
      n<Dim, value_t> coord(int i, bool stagger) const {
    return coord<n>(i, (int)stagger);
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
  template <int n>
      HD_INLINE std::enable_if_t <
      n<Dim, value_t> coord(int i, int stagger) const {
    return coord<n>(i, (value_t)(0.5 - 0.5 * stagger));
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
  template <int n>
      HD_INLINE std::enable_if_t <
      n<Dim, value_t> coord(int i, float pos_in_cell) const {
    return (lower[n] + delta[n] * (i - guard[n] + pos_in_cell));
  }

  template <int n>
      HD_INLINE std::enable_if_t <
      n<Dim, value_t> coord(int i, double pos_in_cell) const {
    return (lower[n] + delta[n] * (i - guard[n] + pos_in_cell));
  }

  template <int n>
  HD_INLINE std::enable_if_t<n >= Dim, value_t> coord(int i,
                                                      float pos_in_cell) const {
    return pos_in_cell;
  }

  template <int n>
  HD_INLINE std::enable_if_t<n >= Dim, value_t> coord(
      int i, double pos_in_cell) const {
    return pos_in_cell;
  }

  HD_INLINE value_t coord(int dir, int32_t i, double pos_in_cell) const {
    if (dir < Dim)
      return lower[dir] + delta[dir] * (i - guard[dir] + pos_in_cell);
    else
      return pos_in_cell;
  }

  HD_INLINE value_t coord(int dir, int32_t i, float pos_in_cell) const {
    if (dir < Dim)
      return lower[dir] + delta[dir] * (i - guard[dir] + pos_in_cell);
    else
      return pos_in_cell;
  }

  HD_INLINE value_t coord(int dir, int32_t n, int stagger) const {
    return coord(dir, n, (value_t)(0.5 - 0.5 * stagger));
  }

  HD_INLINE value_t coord(int dir, int32_t n, bool stagger) const {
    return coord(dir, n, (int)stagger);
  }

  template <int n>
  HD_INLINE value_t coord(const index_t<Dim>& pos, value_t pos_in_cell) const {
    return coord<n>(pos[n], pos_in_cell);
  }

  template <int n>
  HD_INLINE value_t coord(const index_t<Dim>& pos, stagger_t st) const {
    return coord<n>(pos[n], st[n]);
  }

  // template <typename value_t = Scalar>
  HD_INLINE vec_t<value_t, 3> coord_global(
      const index_t<Dim>& pos, const vec_t<value_t, 3>& rel_x) const {
    vec_t<value_t, 3> result = rel_x;
#pragma unroll
    for (int i = 0; i < Dim; i++) {
      result[i] = coord(i, pos[i], rel_x[i]);
    }
    return result;
  }

  HD_INLINE vec_t<value_t, 3> coord_global(
      const index_t<Dim>& pos) const {
    vec_t<value_t, 3> result;
#pragma unroll
    for (int i = 0; i < Dim; i++) {
      result[i] = coord(i, pos[i], false);
    }
    return result;
  }

  template <typename FloatT>
  HD_INLINE void from_global(const vec_t<value_t, 3>& global_x,
                             index_t<Dim>& pos, vec_t<FloatT, 3>& rel_x) const {
    rel_x = global_x;
#pragma unroll
    for (int i = 0; i < Dim; i++) {
      pos[i] = int(floor((global_x[i] - lower[i]) / delta[i]));
      rel_x[i] = (global_x[i] - lower[i] - pos[i] * delta[i]) * inv_delta[i];
      pos[i] += guard[i];
    }
  }

  ///  Find the zone the cell belongs to (for communication purposes)
  HD_INLINE int find_zone(const index_t<Dim>& pos) const {
    int stride = 1;
    int linear = 0;
#pragma unroll
    for (int i = 0; i < Dim; i++) {
      // z[i] = (pos[i] >= guard[i]) + (pos[i] >= (dims[i] - guard[i]));
      linear +=
          ((pos[i] >= guard[i]) + (pos[i] >= (dims[i] - guard[i]))) * stride;
      stride *= 3;
    }
    // ext.get_strides();

    // For now we always use col major for zone index
    // idx_col_major_t<Dim> idx(z, ext);
    return linear;
  }

  HD_INLINE bool is_in_bound(const vec_t<value_t, 3>& global_x) const {
    bool result = true;
    for (int n = 0; n < Dim; n++) {
      if (global_x[n] < lower[n] || global_x[n] > lower[n] + sizes[n]) {
        result = false;
        break;
      }
    }
    return result;
  }

  HD_INLINE bool is_in_bound(const index_t<Dim>& pos) const {
#pragma unroll
    for (int i = 0; i < Dim; i++) {
      if (pos[i] < guard[i] || pos[i] >= N[i] + guard[i]) return false;
    }
    return true;
  }

  HD_INLINE bool is_in_bound(const index_t<Dim>& pos, stagger_t stagger) const {
#pragma unroll
    for (int i = 0; i < Dim; i++) {
      // The convention is that stagger moves the grid point from 0.5 to 0, therefore
      // we need to include an extra cell at the very end
      if (pos[i] < guard[i] || pos[i] >= N[i] + guard[i] + stagger[i]) return false;
    }
    return true;
  }

  template <typename... Args>
  HD_INLINE bool is_in_bound(Args... args) const {
    return is_in_bound(index(args...));
  }

  template <typename Idx_t>
  HD_INLINE index_t<Dim> idx_to_pos(const Idx_t& idx) const {
    auto ext = extent();
    return get_pos(idx, ext);
  }

  HD_INLINE bool is_in_grid(const index_t<Dim>& pos) const {
#pragma unroll
    for (int i = 0; i < Dim; i++) {
      if (pos[i] >= dims[i]) return false;
    }
    return true;
  }

  template <typename... Args>
  HD_INLINE bool is_in_grid(Args... args) const {
    return is_in_grid(index(args...));
  }

  HD_INLINE extent_t<Dim> extent() const {
    extent_t<Dim> result(dims);
    // #pragma unroll
    //     for (int i = 0; i < Dim; i++) result[i] = dims[i];
    result.get_strides();
    return result;
  }

  HD_INLINE extent_t<Dim> extent_less() const {
    extent_t<Dim> result(N);
    // #pragma unroll
    //     for (int i = 0; i < Dim; i++) result[i] = reduced_dim(i);
    result.get_strides();
    return result;
  }

  HD_INLINE index_t<Dim> guards() const {
    index_t<Dim> result(guard);
    // #pragma unroll
    //     for (int i = 0; i < Dim; i++) {
    //       result[i] = guard[i];
    //     }
    return result;
  }

  HD_INLINE index_t<Dim> offsets() const {
    index_t<Dim> result(offset);
    // #pragma unroll
    //     for (int i = 0; i < Dim; i++) {
    //       result[i] = offset[i];
    //     }
    return result;
  }

  HD_INLINE value_t cell_size() const {
    value_t result = 1.0;
#pragma unroll
    for (int i = 0; i < Dim; i++) {
      result *= delta[i];
    }
    return result;
  }

  HD_INLINE uint32_t size() const {
    uint32_t result = dims[0];
#pragma unroll
    for (int i = 1; i < Dim; i++) {
      result *= dims[i];
    }
    return result;
  }
};

template <int Dim, typename value_t>
Grid<Dim, value_t>
make_grid(const vec_t<uint32_t, Dim>& N, const vec_t<uint32_t, Dim>& guard,
          const vec_t<value_t, Dim>& sizes, const vec_t<value_t, Dim>& lower) {
  Grid<Dim, value_t> grid;
  for (int i = 0; i < Dim; i++) {
    grid.N[i] = N[i];
    grid.guard[i] = guard[i];
    grid.dims[i] = N[i] + 2 * guard[i];
    grid.sizes[i] = sizes[i];
    grid.lower[i] = lower[i];
    grid.delta[i] = sizes[i] / N[i];
    grid.inv_delta[i] = 1.0 / grid.delta[i];
    grid.offset[i] = 0;
  }
  return grid;
}

}  // namespace Aperture
