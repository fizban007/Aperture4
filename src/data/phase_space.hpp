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

#ifndef PHASE_SPACE_H_
#define PHASE_SPACE_H_

#include "core/multi_array.hpp"
#include "framework/data.h"

namespace Aperture {

template <typename Conf, int Dim = 1>
class phase_space : public data_t {
 public:
  multi_array<float, Conf::dim + Dim, idx_col_major_t<Conf::dim + Dim>> data;
  using grid_t = typename Conf::grid_t;
  const grid_t& m_grid;

  extent_t<Conf::dim> m_grid_ext;
  int m_downsample = 16;
  int m_log_scale = false;
  int m_num_bins[Dim];
  float m_lower[Dim];
  float m_upper[Dim];

  phase_space(const grid_t& grid, int downsample, const int* num_bins,
              const float* lower, const float* upper,
              bool use_log_scale = false, MemType memtype = default_mem_type)
      : m_grid(grid) {
    m_downsample = downsample;
    m_log_scale = use_log_scale;

    for (int i = 0; i < Dim; i++) {
      m_num_bins[i] = num_bins[i];
      m_lower[i] = lower[i];
      m_upper[i] = upper[i];
    }

    auto g_ext = grid.extent_less();
    extent_t<Conf::dim + Dim> ext;

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
#ifdef CUDA_ENABLED
    data.assign_dev(0.0f);
#else
    data.assign(0.0f);
#endif
  }

  void copy_to_host() { data.copy_to_host(); }

  void copy_to_device() { data.copy_to_device(); }
};

template <typename Conf, int Dim>
struct host_adapter<phase_space<Conf, Dim>> {
  typedef ndptr<float, Conf::dim + Dim, idx_col_major_t<Conf::dim + Dim>> type;
  typedef ndptr_const<float, Conf::dim + Dim, idx_col_major_t<Conf::dim + Dim>>
      const_type;

  static inline const_type apply(const phase_space<Conf, Dim>& data) {
    return data.data.host_ptr();
  }

  static inline type apply(phase_space<Conf, Dim>& data) {
    return data.data.host_ptr();
  }
};

template <typename Conf, int Dim>
struct cuda_adapter<phase_space<Conf, Dim>> {
  typedef ndptr<float, Conf::dim + Dim, idx_col_major_t<Conf::dim + Dim>> type;
  typedef ndptr_const<float, Conf::dim + Dim, idx_col_major_t<Conf::dim + Dim>>
      const_type;

  static inline const_type apply(const phase_space<Conf, Dim>& data) {
    return data.data.dev_ptr();
  }

  static inline type apply(phase_space<Conf, Dim>& data) {
    return data.data.dev_ptr();
  }
};

}  // namespace Aperture

#endif  // PHASE_SPACE_H_
