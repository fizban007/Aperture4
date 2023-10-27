/*
 * Copyright (c) 2023 Alex Chen.
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

#include "core/multi_array.hpp"
#include "framework/data.h"

namespace Aperture {

/// \brief Phase space data. Dim is the dimension of the phase space, which can
/// be from 1 to 3.
template <typename Conf, int Dim = 1>
class phase_space : public data_t {
 public:
  using value_t = typename Conf::value_t;
  using grid_t = typename Conf::grid_t;
  multi_array<value_t, Conf::dim + Dim, idx_col_major_t<Conf::dim + Dim>> data;
  const grid_t& m_grid;

  extent_t<Conf::dim> m_grid_ext;
  extent_t<Dim> m_momentum_ext;
  int m_downsample = 16;
  int m_log_scale = false;
  int m_num_bins[Dim];
  value_t m_lower[Dim];
  value_t m_upper[Dim];
  value_t m_dp[Dim];

  phase_space(const grid_t& grid, int downsample, const int* num_bins,
              const value_t* lower, const value_t* upper,
              // const float* lower, const float* upper,
              bool use_log_scale = false, MemType memtype = default_mem_type)
      : m_grid(grid) {
    m_downsample = downsample;
    m_log_scale = use_log_scale;

    for (int i = 0; i < Dim; i++) {
      m_num_bins[i] = num_bins[i];
      m_lower[i] = lower[i];
      m_upper[i] = upper[i];
      m_momentum_ext[i] = num_bins[i];
      m_dp[i] = (upper[i] - lower[i]) / num_bins[i];
    }

    auto g_ext = grid.extent_less();
    // ext is the total extent of the phase space
    extent_t<Conf::dim + Dim> ext;

    // The first Dim dimensions are momentum space dimensions, while the
    // subsequent Conf::dim dimensions are the spatial grid dimensions
    for (int i = 0; i < Conf::dim; i++) {
      ext[i + Dim] = g_ext[i] / m_downsample;
      m_grid_ext[i] = g_ext[i] / m_downsample;
    }
    for (int i = 0; i < Dim; i++) {
      ext[i] = m_num_bins[i];
    }
    ext.get_strides();
    m_grid_ext.get_strides();

    data.resize(ext);
  }

  void init() override {
#ifdef GPU_ENABLED
    data.assign_dev(0.0);
#else
    data.assign(0.0);
#endif
  }

  template <typename F>
  void set_value(F f) {
    for (auto idx : data.indices()) {
      auto pos = idx.get_pos();
      double x0 = m_grid.coord(0, pos[Dim], false);
      double x1 = (Conf::dim > 1 ?
        m_grid.coord(1, pos[Dim + 1], false) : 0.0);
      double x2 = (Conf::dim > 2 ?
        m_grid.coord(2, pos[Dim + 2], false) : 0.0);
      double p0 = m_lower[0] + pos[0] * m_dp[0];
      double p1 = (Dim > 1 ?
        m_lower[1] + pos[1] * m_dp[1] : 0.0);
      double p2 = (Dim > 2 ?
        m_lower[2] + pos[2] * m_dp[2] : 0.0);
      data[idx] = f(p0, p1, p2, x0, x1, x2);
    }
    if (data.mem_type() != MemType::host_only) {
      data.copy_to_device();
    }
  }

  void copy_to_host() { data.copy_to_host(); }

  void copy_to_device() { data.copy_to_device(); }
};

template <typename Conf, int Dim>
struct host_adapter<phase_space<Conf, Dim>> {
  typedef ndptr<typename Conf::value_t, Conf::dim + Dim, idx_col_major_t<Conf::dim + Dim>> type;
  typedef ndptr_const<typename Conf::value_t, Conf::dim + Dim, idx_col_major_t<Conf::dim + Dim>>
      const_type;

  static inline const_type apply(const phase_space<Conf, Dim>& data) {
    return data.data.host_ptr();
  }

  static inline type apply(phase_space<Conf, Dim>& data) {
    return data.data.host_ptr();
  }
};

template <typename Conf, int Dim>
struct gpu_adapter<phase_space<Conf, Dim>> {
  typedef ndptr<typename Conf::value_t, Conf::dim + Dim, idx_col_major_t<Conf::dim + Dim>> type;
  typedef ndptr_const<typename Conf::value_t, Conf::dim + Dim, idx_col_major_t<Conf::dim + Dim>>
      const_type;

  static inline const_type apply(const phase_space<Conf, Dim>& data) {
    return data.data.dev_ptr();
  }

  static inline type apply(phase_space<Conf, Dim>& data) {
    return data.data.dev_ptr();
  }
};

}  // namespace Aperture
