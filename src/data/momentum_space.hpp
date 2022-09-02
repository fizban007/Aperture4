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

#ifndef __MOMENTUM_SPACE_H_
#define __MOMENTUM_SPACE_H_

#include "core/multi_array.hpp"
#include "framework/data.h"

namespace Aperture {

template <typename Conf>
class momentum_space : public data_t {
 public:
  multi_array<float, Conf::dim + 1, idx_col_major_t<Conf::dim + 1>> e_p1, e_p2,
      e_p3, e_E;
  multi_array<float, Conf::dim + 1, idx_col_major_t<Conf::dim + 1>> p_p1, p_p2,
      p_p3, p_E;
  const typename Conf::grid_t& m_grid;
  extent_t<Conf::dim> m_grid_ext;
  int m_downsample = 16;
  int m_num_bins[4] = {256, 256, 256, 256};
  int m_log_scale = false;
  float m_lower[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float m_upper[4] = {1.0f, 1.0f, 1.0f, 1.0f};

  momentum_space(const typename Conf::grid_t& grid, int downsample,
                 int num_bins[4], float lower[4], float upper[4],
                 bool log_scale = false, MemType memtype = default_mem_type)
      : m_grid(grid) {
    m_downsample = downsample;
    // m_num_bins = num_bins;
    auto g_ext = grid.extent_less();
    extent_t<Conf::dim + 1> ext;
    m_log_scale = log_scale;
    for (int i = 0; i < Conf::dim; i++) {
      ext[i + 1] = g_ext[i] / m_downsample;
      m_grid_ext[i] = g_ext[i] / m_downsample;
    }
    for (int i = 0; i < 4; i++) {
      m_num_bins[i] = num_bins[i];
      m_lower[i] = lower[i];
      m_upper[i] = upper[i];
    }
    ext[0] = m_num_bins[0];
    Logger::print_info("resizing p1 to {}x{}x{}", ext[0], ext[1], ext[2]);
    e_p1.resize(ext);
    p_p1.resize(ext);
    ext[0] = m_num_bins[1];
    Logger::print_info("resizing p2 to {}x{}x{}", ext[0], ext[1], ext[2]);
    e_p2.resize(ext);
    p_p2.resize(ext);
    ext[0] = m_num_bins[2];
    Logger::print_info("resizing p3 to {}x{}x{}", ext[0], ext[1], ext[2]);
    e_p3.resize(ext);
    p_p3.resize(ext);
    ext[0] = m_num_bins[3];
    Logger::print_info("resizing E to {}x{}x{}", ext[0], ext[1], ext[2]);
    e_E.resize(ext);
    p_E.resize(ext);
  }

  void init() override {
#ifdef GPU_ENABLED
    e_p1.assign_dev(0.0f);
    e_p2.assign_dev(0.0f);
    e_p3.assign_dev(0.0f);
    e_E.assign_dev(0.0f);
#else
    e_p1.assign(0.0f);
    e_p2.assign(0.0f);
    e_p3.assign(0.0f);
    e_E.assign(0.0f);
#endif

#ifdef GPU_ENABLED
    p_p1.assign_dev(0.0f);
    p_p2.assign_dev(0.0f);
    p_p3.assign_dev(0.0f);
    p_E.assign_dev(0.0f);
#else
    p_p1.assign(0.0f);
    p_p2.assign(0.0f);
    p_p3.assign(0.0f);
    p_E.assign(0.0f);
#endif
  }

  void copy_to_host() {
    e_p1.copy_to_host();
    e_p2.copy_to_host();
    e_p3.copy_to_host();
    e_E.copy_to_host();
    p_p1.copy_to_host();
    p_p2.copy_to_host();
    p_p3.copy_to_host();
    p_E.copy_to_host();
  }

  void copy_to_device() {
    e_p1.copy_to_device();
    e_p2.copy_to_device();
    e_p3.copy_to_device();
    e_E.copy_to_device();
    p_p1.copy_to_device();
    p_p2.copy_to_device();
    p_p3.copy_to_device();
    p_E.copy_to_device();
  }
};

}  // namespace Aperture

#endif  // __MOMENTUM_SPACE_H_
