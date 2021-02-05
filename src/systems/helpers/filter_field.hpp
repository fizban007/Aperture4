/*
 * Copyright (c) 2021 Alex Chen.
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

#ifndef __FIELD_FILTER_H_
#define __FIELD_FILTER_H_

#include "core/multi_array.hpp"

namespace Aperture {

namespace detail {

template <int Dim>
struct field_filter;

template <>
struct field_filter<1> {
  template <typename Array>
  static HD_INLINE typename Array::value_t apply(
      const typename Array::idx_t& idx, const Array& f,
      const vec_t<bool, 2>& is_boundary) {
    return 0.5f * f[idx] + 0.25f * f[idx.inc_x(!is_boundary[1])] +
           0.25f * f[idx.dec_x(!is_boundary[0])];
  }
};

template <int Dim>
struct field_filter {
  template <typename Array>
  static HD_INLINE typename Array::value_t apply(
      const typename Array::idx_t& idx, const Array& f,
      const vec_t<bool, Dim * 2>& is_boundary) {
    auto boundary_lower = is_boundary.template subset<0, (Dim - 1) * 2>();
    auto f_m = field_filter<Dim - 1>::apply(idx, f, boundary_lower);

    auto idx_l = idx.template dec<Dim - 1>(!is_boundary[(Dim - 1) * 2]);
    auto f_l = field_filter<Dim - 1>::apply(idx_l, f, boundary_lower);

    auto idx_r = idx.template inc<Dim - 1>(!is_boundary[(Dim - 1) * 2 + 1]);
    auto f_r = field_filter<Dim - 1>::apply(idx_r, f, boundary_lower);
    return 0.25f * f_l + 0.5f * f_m + 0.25f * f_r;
  }
};

template <int Dim>
struct field_filter_with_geom_factor;

template <>
struct field_filter_with_geom_factor<1> {
  template <typename Array, typename ConstArray>
  static HD_INLINE typename Array::value_t apply(
      const typename Array::idx_t& idx, const Array& f,
      const ConstArray& factor, const vec_t<bool, 2>& is_boundary) {
    return 0.5f * f[idx] * factor[idx] +
           0.25f * f[idx.inc_x(!is_boundary[1])] *
               factor[idx.inc_x(!is_boundary[1])] +
           0.25f * f[idx.dec_x(!is_boundary[0])] *
               factor[idx.dec_x(!is_boundary[0])];
  }
};

template <int Dim>
struct field_filter_with_geom_factor {
  template <typename Array, typename ConstArray>
  static HD_INLINE typename Array::value_t apply(
      const typename Array::idx_t& idx, const Array& f,
      const ConstArray& factor, const vec_t<bool, Dim * 2>& is_boundary) {
    auto boundary_lower = is_boundary.template subset<0, (Dim - 1) * 2>();
    auto f_m = field_filter_with_geom_factor<Dim - 1>::apply(idx, f, factor,
                                                             boundary_lower);

    auto idx_l = idx.template dec<Dim - 1>(!is_boundary[(Dim - 1) * 2]);
    auto f_l = field_filter_with_geom_factor<Dim - 1>::apply(idx_l, f, factor,
                                                             boundary_lower);

    auto idx_r = idx.template inc<Dim - 1>(!is_boundary[(Dim - 1) * 2 + 1]);
    auto f_r = field_filter_with_geom_factor<Dim - 1>::apply(idx_r, f, factor,
                                                             boundary_lower);
    return 0.25f * f_l + 0.5f * f_m + 0.25f * f_r;
  }
};

}  // namespace detail

template <typename ExecPolicy, int Dim, typename value_t, typename Idx_t>
void
filter_field_component(multi_array<value_t, Dim, Idx_t>& field,
                       multi_array<value_t, Dim, Idx_t>& tmp,
                       const vec_t<bool, 2 * Dim>& is_boundary) {
  ExecPolicy::launch(
      [is_boundary] LAMBDA(auto result, auto f) {
        auto& grid = ExecPolicy::grid();
        auto ext = grid.extent();
        ExecPolicy::loop(
            Idx_t(0, ext), Idx_t(ext.size(), ext),
            [&grid, &ext, &is_boundary] LAMBDA(auto idx, auto& result,
                                               auto& f) {
              auto pos = get_pos(idx, ext);
              if (grid.is_in_bound(pos)) {
                vec_t<bool, 2 * Dim> boundary_cell = is_boundary;
#pragma unroll
                for (int i = 0; i < Dim; i++) {
                  boundary_cell[i * 2] =
                      is_boundary[i * 2] && (pos[i] == grid.guard[i]);
                  boundary_cell[i * 2 + 1] =
                      is_boundary[i * 2 + 1] &&
                      (pos[i] == grid.dims[i] - grid.guard[i] - 1);
                  result[idx] =
                      detail::field_filter<Dim>::apply(idx, f, boundary_cell);
                }
              }
            },
            result, f);
      },
      tmp, field);
  ExecPolicy::sync();
  // Note: This drops the guard cell content in the original field array
  field.copy_from(tmp);
}

template <typename ExecPolicy, int Dim, typename value_t, typename Idx_t>
void
filter_field_component(multi_array<value_t, Dim, Idx_t>& field,
                       multi_array<value_t, Dim, Idx_t>& tmp,
                       const multi_array<value_t, Dim, Idx_t>& geom_factor,
                       const vec_t<bool, 2 * Dim>& is_boundary) {
  ExecPolicy::launch(
      [is_boundary] LAMBDA(auto result, auto f, auto factor) {
        auto& grid = ExecPolicy::grid();
        auto ext = grid.extent();
        ExecPolicy::loop(
            Idx_t(0, ext), Idx_t(ext.size(), ext),
            [&grid, &ext, &is_boundary] LAMBDA(auto idx, auto& result, auto& f,
                                               auto& factor) {
              auto pos = get_pos(idx, ext);
              if (grid.is_in_bound(pos)) {
                vec_t<bool, 2 * Dim> boundary_cell = is_boundary;
#pragma unroll
                for (int i = 0; i < Dim; i++) {
                  boundary_cell[i * 2] =
                      is_boundary[i * 2] && (pos[i] == grid.guard[i]);
                  boundary_cell[i * 2 + 1] =
                      is_boundary[i * 2 + 1] &&
                      (pos[i] == grid.dims[i] - grid.guard[i] - 1);
                  result[idx] =
                      detail::field_filter_with_geom_factor<Dim>::apply(
                          idx, f, factor, boundary_cell) /
                      factor[idx];
                }
              }
            },
            result, f, factor);
      },
      tmp, field, geom_factor);
  ExecPolicy::sync();
  // Note: This drops the guard cell content in the original field array
  field.copy_from(tmp);
}

}  // namespace Aperture

#endif  // __FIELD_FILTER_H_
