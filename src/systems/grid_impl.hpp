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

#pragma once

#include "core/constant_mem_func.h"
#include "core/domain_info.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "grid.h"
#include "systems/domain_comm.h"
#include <exception>

namespace Aperture {

template <typename Conf>
grid_t<Conf>::grid_t(const domain_info_t<Conf::dim>& domain_info) {
  // Start with some sane defaults so that it doesn't blow up when no parameter
  // is provided
  uint32_t vec_N[Conf::dim];
  for (int i = 0; i < Conf::dim; i++) {
    vec_N[i] = 1;
    this->guard[i] = 1;
    this->sizes[i] = 1.0;
    this->lower[i] = 1.0;
  }

  // Obtain grid parameters from the params store
  sim_env().params().get_array("N", vec_N);
  sim_env().params().get_array("guard", this->guard);
  sim_env().params().get_array("size", this->sizes);
  sim_env().params().get_array("lower", this->lower);
  typename Conf::value_t dt = 0.0f;
  sim_env().params().get_value("dt", dt);

  // Initialize the grid parameters
  for (int i = 0; i < Conf::dim; i++) {
    this->delta[i] = this->sizes[i] / vec_N[i];
    this->inv_delta[i] = 1.0 / this->delta[i];
    this->guard[i] = this->guard[i];
    this->dims[i] = vec_N[i] + 2 * this->guard[i];
    this->N[i] = vec_N[i];
    Logger::print_debug("Dim {} has size {}", i, this->dims[i]);
    // if (dt > this->delta[i]) {
    //   Logger::print_err(
    //       "dt is larger than the grid spacing in direction {}, which is {}", i,
    //       this->delta[i]);
    //   exit(1);
    // }
  }

  // Adjust the grid according to domain decomposition
  for (int d = 0; d < Conf::dim; d++) {
    this->N[d] /= domain_info.mpi_dims[d];
    this->dims[d] = this->N[d] + 2 * this->guard[d];
    this->sizes[d] /= domain_info.mpi_dims[d];
    this->lower[d] += domain_info.mpi_coord[d] * this->sizes[d];
    // TODO: In a non-uniform domain decomposition, the offset could
    // change, need a more robust way to count this
    this->offset[d] = domain_info.mpi_coord[d] * this->reduced_dim(d);
    // TODO: the z-order case is very ad-hoc. Is there any special
    // repercussions?
    if (Conf::is_zorder) {
      this->dims[d] = next_power_of_two(this->dims[d]);
    }
  }

  // Copy the grid parameters to gpu
// #if defined(GPU_ENABLED) && (defined(__CUDACC__) || defined(__HIPCC__))
#if defined(GPU_ENABLED)
  init_dev_grid<Conf::dim, typename Conf::value_t>(*this);
#endif

  m_ext = this->extent();
}

template <typename Conf>
grid_t<Conf>::grid_t() : grid_t(domain_info_t<Conf::dim>{}) {}

// template <typename Conf>
// grid_t<Conf>::grid_t(const domain_comm<Conf>& comm)
//     : grid_t(comm.domain_info()) {
//   comm.resize_buffers(*this);
// }

template <typename Conf>
grid_t<Conf>::~grid_t() {}

}  // namespace Aperture

