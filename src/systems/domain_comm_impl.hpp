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

#include "core/constant_mem_func.h"
#include "core/detail/multi_array_helpers.h"
#include "domain_comm.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "framework/params_store.h"
#include "utils/logger.h"
#include "utils/timer.h"

#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>  // Needed for CUDA-aware check
#endif

#define USE_CUDA_AWARE_MPI true

#if CUDA_ENABLED && USE_CUDA_AWARE_MPI && defined(MPIX_CUDA_AWARE_SUPPORT) && \
    MPIX_CUDA_AWARE_SUPPORT
#pragma message "CUDA-aware MPI found!"
constexpr bool use_cuda_mpi = true;
#else
constexpr bool use_cuda_mpi = false;
#endif

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
// domain_comm<Conf>::domain_comm(sim_environment &env) : system_t(env) {
domain_comm<Conf, ExecPolicy>::domain_comm(int *argc, char ***argv) {
  int is_initialized = 0;
  MPI_Initialized(&is_initialized);

  if (!is_initialized) {
    if (argc == nullptr && argv == nullptr) {
      MPI_Init(NULL, NULL);
    } else {
      MPI_Init(argc, argv);
    }
  }
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int log_level = (int)LogLevel::debug;
  sim_env().params().get_value("log_level", log_level);
  Logger::init(rank, (LogLevel)log_level);

  setup_domain();
  MPI_Helper::register_particle_type(single_ptc_t{}, &MPI_PARTICLES);
  MPI_Helper::register_particle_type(single_ph_t{}, &MPI_PHOTONS);
  // Logger::print_debug("m_is_device is {}", m_is_device);
}

template <typename Conf, template <class> class ExecPolicy>
domain_comm<Conf, ExecPolicy>::~domain_comm() {
  MPI_Type_free(&MPI_PARTICLES);
  MPI_Type_free(&MPI_PHOTONS);

  int is_finalized = 0;
  MPI_Finalized(&is_finalized);

  if (!is_finalized) MPI_Finalize();
}

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::setup_domain() {
  m_world = MPI_COMM_WORLD;
  MPI_Comm_rank(m_world, &m_rank);
  MPI_Comm_size(m_world, &m_size);

  m_scalar_type = MPI_Helper::get_mpi_datatype(typename Conf::value_t{});

  auto dims = sim_env().params().template get_as<std::vector<int64_t>>("nodes");
  if (dims.size() < Conf::dim) dims.resize(Conf::dim, 1);

  int64_t total_dim = 1;
  for (int i = 0; i < Conf::dim; i++) {
    total_dim *= dims[i];
  }

  if (total_dim != m_size) {
    // Given node configuration is not correct, create one on our own
    Logger::print_err(
        "Domain decomp in config file does not make sense, generating "
        "our own.");
    for (int i = 0; i < Conf::dim; i++) dims[i] = 0;

    MPI_Dims_create(m_size, Conf::dim, m_domain_info.mpi_dims);
    Logger::err("Created domain decomp as");
    for (int i = 0; i < Conf::dim; i++) {
      Logger::err("{}", m_domain_info.mpi_dims[i]);
      if (i != Conf::dim - 1) Logger::err(" x ");
    }
    Logger::err("\n");
  } else {
    for (int i = 0; i < Conf::dim; i++) m_domain_info.mpi_dims[i] = dims[i];
  }

  auto periodic = sim_env().params().template get_as<std::vector<bool>>(
      "periodic_boundary");
  for (int i = 0; i < std::min(Conf::dim, (int)periodic.size()); i++)
    m_domain_info.is_periodic[i] = periodic[i];

  // Create a cartesian MPI group for communication
  MPI_Cart_create(m_world, Conf::dim, m_domain_info.mpi_dims,
                  m_domain_info.is_periodic, true, &m_cart);

  // Obtain the mpi coordinate of the current rank
  MPI_Cart_coords(m_cart, m_rank, Conf::dim, m_domain_info.mpi_coord);
  // std::cout << "Rank " << m_rank << " has mpi coord " <<
  // m_domain_info.mpi_coord[0] << ", "
  //   << m_domain_info.mpi_coord[1] << ", " << m_domain_info.mpi_coord[2];

  // Figure out if the current rank is at any boundary
  int left = 0, right = 0;
  int rank = 0;
  for (int n = 0; n < Conf::dim; n++) {
    MPI_Cart_shift(m_cart, n, -1, &rank, &left);
    MPI_Cart_shift(m_cart, n, 1, &rank, &right);
    m_domain_info.neighbor_left[n] = left;
    m_domain_info.neighbor_right[n] = right;
    Logger::print_detail_all(
        "Rank {} has neighbors in {} direction: left {}, right {}", m_rank, n,
        left, right);
    if (left < 0) m_domain_info.is_boundary[2 * n] = true;
    if (right < 0) m_domain_info.is_boundary[2 * n + 1] = true;
  }

  setup_devices();
}

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::resize_buffers(
    const typename Conf::grid_t &grid) const {
  Logger::print_debug("Resizing comm buffers");
  if (m_buffers_ready) return;
  for (int i = 0; i < Conf::dim; i++) {
    auto ext = extent_t<Conf::dim>{};
    for (int j = 0; j < Conf::dim; j++) {
      if (j == i)
        ext[j] = grid.guard[j];
      else
        ext[j] = grid.dims[j];
    }
    ext.get_strides();
    // ext_vec is a bundled 3-pack of send/recv buffer, used for vector fields
    auto ext_vec = ext;
    ext_vec[Conf::dim - 1] *= 3;
    ext_vec.get_strides();

    m_send_buffers.emplace_back(ext, ExecPolicy<Conf>::data_mem_type());
    m_recv_buffers.emplace_back(ext, ExecPolicy<Conf>::data_mem_type());
    m_send_vec_buffers.emplace_back(ext_vec, ExecPolicy<Conf>::data_mem_type());
    m_recv_vec_buffers.emplace_back(ext_vec, ExecPolicy<Conf>::data_mem_type());
  }

  size_t ptc_buffer_size =
      sim_env().params().template get_as<int64_t>("ptc_buffer_size", 100000l);
  size_t ph_buffer_size =
      sim_env().params().template get_as<int64_t>("ph_buffer_size", 100000l);
  int num_ptc_buffers = std::pow(3, Conf::dim);
  for (int i = 0; i < num_ptc_buffers; i++) {
    m_ptc_buffers.emplace_back(ptc_buffer_size,
                               ExecPolicy<Conf>::data_mem_type());
    m_ph_buffers.emplace_back(ph_buffer_size,
                              ExecPolicy<Conf>::data_mem_type());
  }
  m_ptc_buffer_ptrs.resize(num_ptc_buffers);
  m_ph_buffer_ptrs.resize(num_ptc_buffers);
  m_ptc_buffer_num.resize(num_ptc_buffers);
  m_ph_buffer_num.resize(num_ptc_buffers);
  for (int i = 0; i < num_ptc_buffers; i++) {
    m_ptc_buffer_ptrs[i] = m_ptc_buffers[i].dev_ptr();
    m_ph_buffer_ptrs[i] = m_ph_buffers[i].dev_ptr();
  }
  m_ptc_buffer_ptrs.copy_to_device();
  m_ph_buffer_ptrs.copy_to_device();
  m_ptc_buffer_num.assign(0);
  m_ph_buffer_num.assign(0);

  // Logger::print_debug("m_ptc_buffers has size {}", m_ptc_buffers.size());
  m_buffers_ready = true;
}

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::send_array_guard_cells_single_dir(
    typename Conf::multi_array_t &array, const typename Conf::grid_t &grid,
    int dim, int dir) const {
  if (dim < 0 || dim >= Conf::dim) return;

  int dest, origin;
  MPI_Status status;

  dest = (dir == -1 ? m_domain_info.neighbor_left[dim]
                    : m_domain_info.neighbor_right[dim]);
  origin = (dir == -1 ? m_domain_info.neighbor_right[dim]
                      : m_domain_info.neighbor_left[dim]);

  // Index send_idx(0, 0, 0);
  auto send_idx = index_t<Conf::dim>{};
  send_idx[dim] =
      (dir == -1 ? grid.guard[dim] : grid.dims[dim] - 2 * grid.guard[dim]);
  auto recv_idx = index_t<Conf::dim>{};
  recv_idx[dim] = (dir == -1 ? grid.dims[dim] - grid.guard[dim] : 0);

  if (dest == m_rank && origin == m_rank) {
    // if (array.mem_type() == MemType::host_only) {
    //   copy(exec_tags::host{}, array, array, recv_idx, send_idx,
    //   m_send_buffers[dim].extent());
    // } else {
    //   copy(exec_tags::device{}, array, array, recv_idx, send_idx,
    //   m_send_buffers[dim].extent());
    // }
    copy(typename ExecPolicy<Conf>::exec_tag{}, array, array, recv_idx,
         send_idx, m_send_buffers[dim].extent());
  } else {
    // timer::stamp();
    // if (array.mem_type() == MemType::host_only) {
    //   copy(exec_tags::host{}, m_send_buffers[dim], array,
    //   index_t<Conf::dim>{}, send_idx,
    //        m_send_buffers[dim].extent());
    // } else {
    //   copy(exec_tags::device{}, m_send_buffers[dim], array,
    //   index_t<Conf::dim>{}, send_idx,
    //            m_send_buffers[dim].extent());
    // }
    copy(typename ExecPolicy<Conf>::exec_tag{}, m_send_buffers[dim], array,
         index_t<Conf::dim>{}, send_idx, m_send_buffers[dim].extent());
    // timer::show_duration_since_stamp("copy guard cells", "ms");

    auto send_ptr = m_send_buffers[dim].host_ptr();
    auto recv_ptr = m_recv_buffers[dim].host_ptr();
    if constexpr (m_is_device && use_cuda_mpi) {
      send_ptr = m_send_buffers[dim].dev_ptr();
      recv_ptr = m_recv_buffers[dim].dev_ptr();
    } else {
      m_send_buffers[dim].copy_to_host();
    }

    // timer::stamp();
    MPI_Sendrecv(send_ptr, m_send_buffers[dim].size(), m_scalar_type, dest, dim,
                 recv_ptr, m_recv_buffers[dim].size(), m_scalar_type, origin,
                 dim, m_cart, &status);

    if (origin != MPI_PROC_NULL) {
      if constexpr (m_is_device && !use_cuda_mpi) {
        m_recv_buffers[dim].copy_to_device();
      }
      copy(typename ExecPolicy<Conf>::exec_tag{}, array, m_recv_buffers[dim],
           recv_idx, index_t<Conf::dim>{}, m_recv_buffers[dim].extent());
    }
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::send_add_array_guard_cells_single_dir(
    typename Conf::multi_array_t &array, const typename Conf::grid_t &grid,
    int dim, int dir) const {
  if (dim < 0 || dim >= Conf::dim) return;

  int dest, origin;
  MPI_Status status;

  dest = (dir == -1 ? m_domain_info.neighbor_left[dim]
                    : m_domain_info.neighbor_right[dim]);
  origin = (dir == -1 ? m_domain_info.neighbor_right[dim]
                      : m_domain_info.neighbor_left[dim]);

  // Index send_idx(0, 0, 0);
  auto send_idx = index_t<Conf::dim>{};
  send_idx[dim] = (dir == -1 ? 0 : grid.dims[dim] - grid.guard[dim]);

  auto recv_idx = index_t<Conf::dim>{};
  recv_idx[dim] =
      (dir == -1 ? grid.dims[dim] - 2 * grid.guard[dim] : grid.guard[dim]);

  if (dest == m_rank && origin == m_rank) {
    // if (array.mem_type() == MemType::host_only) {
    //   add(exec_tags::host{}, array, array, recv_idx, send_idx,
    //   m_recv_buffers[dim].extent());
    // } else {
    //   add(exec_tags::device{}, array, array, recv_idx, send_idx,
    //   m_recv_buffers[dim].extent());
    // }
    add(typename ExecPolicy<Conf>::exec_tag{}, array, array, recv_idx, send_idx,
        m_recv_buffers[dim].extent());
    // if (array.mem_type() == MemType::host_only) {
    //   add(exec_tags::host{}, array, array, recv_idx, send_idx,
    //   m_recv_buffers[dim].extent());
    // } else {
    //   add(exec_tags::device{}, array, array, recv_idx, send_idx,
    //   m_recv_buffers[dim].extent());
    // }
    add(typename ExecPolicy<Conf>::exec_tag{}, array, array, recv_idx, send_idx,
        m_recv_buffers[dim].extent());
  } else {
    // if (array.mem_type() == MemType::host_only) {
    //   copy(exec_tags::host{}, m_send_buffers[dim], array,
    //   index_t<Conf::dim>{}, send_idx,
    //        m_send_buffers[dim].extent());
    // } else {
    //   copy(exec_tags::device{}, m_send_buffers[dim], array,
    //   index_t<Conf::dim>{}, send_idx,
    //            m_send_buffers[dim].extent());
    // }
    copy(typename ExecPolicy<Conf>::exec_tag{}, m_send_buffers[dim], array,
         index_t<Conf::dim>{}, send_idx, m_send_buffers[dim].extent());

    auto send_ptr = m_send_buffers[dim].host_ptr();
    auto recv_ptr = m_recv_buffers[dim].host_ptr();
    if constexpr (m_is_device && use_cuda_mpi) {
      send_ptr = m_send_buffers[dim].dev_ptr();
      recv_ptr = m_recv_buffers[dim].dev_ptr();
    } else {
      m_send_buffers[dim].copy_to_host();
    }

    MPI_Sendrecv(send_ptr, m_send_buffers[dim].size(), m_scalar_type, dest, 0,
                 recv_ptr, m_recv_buffers[dim].size(), m_scalar_type, origin, 0,
                 m_cart, &status);

    if (origin != MPI_PROC_NULL) {
      if constexpr (m_is_device && !use_cuda_mpi) {
        m_recv_buffers[dim].copy_to_device();
      }
      add(typename ExecPolicy<Conf>::exec_tag{}, array, m_recv_buffers[dim],
          recv_idx, index_t<Conf::dim>{}, m_recv_buffers[dim].extent());
    }
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::send_vector_field_guard_cells_single_dir(
    vector_field<Conf> &field, int dim, int dir) const {
  if (dim < 0 || dim >= Conf::dim) return;

  int dest, origin;
  auto &grid = field.grid();
  MPI_Status status;

  dest = (dir == -1 ? m_domain_info.neighbor_left[dim]
                    : m_domain_info.neighbor_right[dim]);
  origin = (dir == -1 ? m_domain_info.neighbor_right[dim]
                      : m_domain_info.neighbor_left[dim]);

  // Index send_idx(0, 0, 0);
  auto send_idx = index_t<Conf::dim>{};
  send_idx[dim] =
      (dir == -1 ? grid.guard[dim] : grid.dims[dim] - 2 * grid.guard[dim]);
  auto recv_idx = index_t<Conf::dim>{};
  recv_idx[dim] = (dir == -1 ? grid.dims[dim] - grid.guard[dim] : 0);

  if (dest == m_rank && origin == m_rank) {
    for (int n = 0; n < 3; n++) {
      auto &array = field[n];
      // if (array.mem_type() == MemType::host_only) {
      //   copy(exec_tags::host{}, array, array, recv_idx, send_idx,
      //   m_send_buffers[dim].extent());
      // } else {
      //   copy(exec_tags::device{}, array, array, recv_idx, send_idx,
      //            m_send_buffers[dim].extent());
      // }
      copy(typename ExecPolicy<Conf>::exec_tag{}, array, array, recv_idx,
           send_idx, m_send_buffers[dim].extent());
    }
  } else {
    // timer::stamp();
    // Logger::print_debug_all("At rank {}; Recving from rank {}, and sending to
    // rank {}", m_rank, origin, dest);
    for (int n = 0; n < 3; n++) {
      auto &array = field[n];
      index_t<Conf::dim> vec_buf_idx{};
      vec_buf_idx[Conf::dim - 1] =
          n * m_send_buffers[dim].extent()[Conf::dim - 1];
      // if (array.mem_type() == MemType::host_only) {
      //   copy(exec_tags::host{}, m_send_vec_buffers[dim], array, vec_buf_idx,
      //   send_idx,
      //        m_send_buffers[dim].extent());
      // } else {
      //   copy(exec_tags::device{}, m_send_vec_buffers[dim], array,
      //   vec_buf_idx, send_idx,
      //            m_send_buffers[dim].extent());
      // }
      copy(typename ExecPolicy<Conf>::exec_tag{}, m_send_vec_buffers[dim],
           array, vec_buf_idx, send_idx, m_send_buffers[dim].extent());
    }
    // timer::show_duration_since_stamp("copy guard cells", "ms");

    auto send_ptr = m_send_vec_buffers[dim].host_ptr();
    auto recv_ptr = m_recv_vec_buffers[dim].host_ptr();
    if constexpr (m_is_device && use_cuda_mpi) {
      send_ptr = m_send_vec_buffers[dim].dev_ptr();
      recv_ptr = m_recv_vec_buffers[dim].dev_ptr();
    } else {
      m_send_vec_buffers[dim].copy_to_host();
    }

    // timer::stamp();
    MPI_Sendrecv(send_ptr, m_send_vec_buffers[dim].size(), m_scalar_type, dest,
                 0, recv_ptr, m_recv_vec_buffers[dim].size(), m_scalar_type,
                 origin, 0, m_cart, &status);
    // timer::show_duration_since_stamp("MPI sendrecv", "ms");

    if (origin != MPI_PROC_NULL) {
      for (int n = 0; n < 3; n++) {
        auto &array = field[n];
        index_t<Conf::dim> vec_buf_idx{};
        vec_buf_idx[Conf::dim - 1] =
            n * m_recv_buffers[dim].extent()[Conf::dim - 1];

        if constexpr (m_is_device && !use_cuda_mpi) {
          m_recv_vec_buffers[dim].copy_to_device();
        }
        copy(typename ExecPolicy<Conf>::exec_tag{}, array,
             m_recv_vec_buffers[dim], recv_idx, vec_buf_idx,
             m_recv_buffers[dim].extent());
      }
    }
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::send_add_vector_field_guard_cells_single_dir(
    vector_field<Conf> &field, int dim, int dir) const {
  if (dim < 0 || dim >= Conf::dim) return;

  int dest, origin;
  auto &grid = field.grid();
  MPI_Status status;

  dest = (dir == -1 ? m_domain_info.neighbor_left[dim]
                    : m_domain_info.neighbor_right[dim]);
  origin = (dir == -1 ? m_domain_info.neighbor_right[dim]
                      : m_domain_info.neighbor_left[dim]);

  // Index send_idx(0, 0, 0);
  auto send_idx = index_t<Conf::dim>{};
  send_idx[dim] = (dir == -1 ? 0 : grid.dims[dim] - grid.guard[dim]);
  auto recv_idx = index_t<Conf::dim>{};
  recv_idx[dim] =
      (dir == -1 ? grid.dims[dim] - 2 * grid.guard[dim] : grid.guard[dim]);
  // Logger::print_debug_all("recv_idx is ({}, {}, {}), dim is {}, dir is {}",
  //                         recv_idx[0], recv_idx[1], recv_idx[2],
  //                         dim, dir);

  if (dest == m_rank && origin == m_rank) {
    for (int n = 0; n < 3; n++) {
      auto &array = field[n];
      // if (array.mem_type() == MemType::host_only) {
      //   add(exec_tags::host{}, array, array, recv_idx, send_idx,
      //   m_send_buffers[dim].extent());
      // } else {
      //   add(exec_tags::device{}, array, array, recv_idx, send_idx,
      //            m_send_buffers[dim].extent());
      // }
      add(typename ExecPolicy<Conf>::exec_tag{}, array, array, recv_idx,
          send_idx, m_send_buffers[dim].extent());
    }
  } else {
    // timer::stamp();
    for (int n = 0; n < 3; n++) {
      auto &array = field[n];
      index_t<Conf::dim> vec_buf_idx{};
      vec_buf_idx[Conf::dim - 1] =
          n * m_send_buffers[dim].extent()[Conf::dim - 1];
      // if (array.mem_type() == MemType::host_only) {
      //   copy(exec_tags::host{}, m_send_vec_buffers[dim], array, vec_buf_idx,
      //   send_idx,
      //        m_send_buffers[dim].extent());
      // } else {
      //   copy(exec_tags::device{}, m_send_vec_buffers[dim], array,
      //   vec_buf_idx, send_idx,
      //            m_send_buffers[dim].extent());
      // }
      copy(typename ExecPolicy<Conf>::exec_tag{}, m_send_vec_buffers[dim],
           array, vec_buf_idx, send_idx, m_send_buffers[dim].extent());
    }
    // timer::show_duration_since_stamp("copy guard cells", "ms");

    auto send_ptr = m_send_vec_buffers[dim].host_ptr();
    auto recv_ptr = m_recv_vec_buffers[dim].host_ptr();
    if constexpr (m_is_device && use_cuda_mpi) {
      send_ptr = m_send_vec_buffers[dim].dev_ptr();
      recv_ptr = m_recv_vec_buffers[dim].dev_ptr();
    } else {
      m_send_vec_buffers[dim].copy_to_host();
    }

    // timer::stamp();
    MPI_Sendrecv(send_ptr, m_send_vec_buffers[dim].size(), m_scalar_type, dest,
                 0, recv_ptr, m_recv_vec_buffers[dim].size(), m_scalar_type,
                 origin, 0, m_cart, &status);
    // timer::show_duration_since_stamp("MPI sendrecv", "ms");

    if (origin != MPI_PROC_NULL) {
      for (int n = 0; n < 3; n++) {
        auto &array = field[n];
        index_t<Conf::dim> vec_buf_idx{};
        vec_buf_idx[Conf::dim - 1] =
            n * m_send_buffers[dim].extent()[Conf::dim - 1];

        if constexpr (m_is_device && !use_cuda_mpi) {
          m_recv_vec_buffers[dim].copy_to_device();
        }
        add(typename ExecPolicy<Conf>::exec_tag{}, array,
            m_recv_vec_buffers[dim], recv_idx, vec_buf_idx,
            m_recv_buffers[dim].extent());
      }
    }
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::send_guard_cells(
    vector_field<Conf> &field) const {
  if (!m_buffers_ready) resize_buffers(field.grid());
  // send_guard_cells(field[0], field.grid());
  // send_guard_cells(field[1], field.grid());
  // send_guard_cells(field[2], field.grid());
  send_vector_field_guard_cells_single_dir(field, 0, -1);
  send_vector_field_guard_cells_single_dir(field, 0, 1);
  send_vector_field_guard_cells_single_dir(field, 1, -1);
  send_vector_field_guard_cells_single_dir(field, 1, 1);
  send_vector_field_guard_cells_single_dir(field, 2, -1);
  send_vector_field_guard_cells_single_dir(field, 2, 1);
}

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::send_guard_cells(
    scalar_field<Conf> &field) const {
  if (!m_buffers_ready) resize_buffers(field.grid());
  send_guard_cells(field[0], field.grid());
}

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::send_guard_cells(
    typename Conf::multi_array_t &array,
    const typename Conf::grid_t &grid) const {
  send_array_guard_cells_single_dir(array, grid, 0, -1);
  send_array_guard_cells_single_dir(array, grid, 0, 1);
  send_array_guard_cells_single_dir(array, grid, 1, -1);
  send_array_guard_cells_single_dir(array, grid, 1, 1);
  send_array_guard_cells_single_dir(array, grid, 2, -1);
  send_array_guard_cells_single_dir(array, grid, 2, 1);
}

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::send_add_guard_cells(
    vector_field<Conf> &field) const {
  if (!m_buffers_ready) resize_buffers(field.grid());
  // send_add_guard_cells(field[0], field.grid());
  // send_add_guard_cells(field[1], field.grid());
  // send_add_guard_cells(field[2], field.grid());
  send_add_vector_field_guard_cells_single_dir(field, 0, -1);
  send_add_vector_field_guard_cells_single_dir(field, 0, 1);
  send_add_vector_field_guard_cells_single_dir(field, 1, -1);
  send_add_vector_field_guard_cells_single_dir(field, 1, 1);
  send_add_vector_field_guard_cells_single_dir(field, 2, -1);
  send_add_vector_field_guard_cells_single_dir(field, 2, 1);
}

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::send_add_guard_cells(
    scalar_field<Conf> &field) const {
  if (!m_buffers_ready) resize_buffers(field.grid());
  send_add_guard_cells(field[0], field.grid());
}

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::send_add_guard_cells(
    typename Conf::multi_array_t &array,
    const typename Conf::grid_t &grid) const {
  send_add_array_guard_cells_single_dir(array, grid, 0, -1);
  send_add_array_guard_cells_single_dir(array, grid, 0, 1);
  send_add_array_guard_cells_single_dir(array, grid, 1, -1);
  send_add_array_guard_cells_single_dir(array, grid, 1, 1);
  send_add_array_guard_cells_single_dir(array, grid, 2, -1);
  send_add_array_guard_cells_single_dir(array, grid, 2, 1);
}

template <typename Conf, template <class> class ExecPolicy>
template <typename SingleType>
void
domain_comm<Conf, ExecPolicy>::send_particle_array(
    std::vector<buffer<SingleType>> &buffers, buffer<int> &buf_nums,
    const std::vector<int> &buf_send_idx, const std::vector<int> &buf_recv_idx,
    int src, int dst, std::vector<MPI_Request> &req_send,
    std::vector<MPI_Request> &req_recv,
    std::vector<MPI_Status> &stat_recv) const {
  // Logger::print_debug("Sending particle array in batch of {}",
  // buf_send_idx.size());
  for (int i = 0; i < buf_send_idx.size(); i++) {
    auto &send_buffer = buffers[buf_send_idx[i]];
    auto &recv_buffer = buffers[buf_recv_idx[i]];

    auto send_ptr = send_buffer.host_ptr();
    auto recv_ptr = recv_buffer.host_ptr() + buf_nums[buf_recv_idx[i]];
    if constexpr (m_is_device && use_cuda_mpi) {
      send_ptr = send_buffer.dev_ptr();
      recv_ptr = recv_buffer.dev_ptr() + buf_nums[buf_recv_idx[i]];
    } else {
      send_buffer.copy_to_host();
    }

    if (src == dst && src == m_rank) {
      if constexpr (m_is_device && use_cuda_mpi) {
#ifdef GPU_ENABLED
        gpuMemcpy(recv_ptr, send_ptr,
                  buf_nums[buf_send_idx[i]] * sizeof(send_buffer[0]),
                  gpuMemcpyDeviceToDevice);
#endif
        // ptr_copy(typename ExecPolicy<Conf>::exec_tag{}, send_ptr, recv_ptr,
        //          buf_nums[buf_send_idx[i]] * sizeof(send_buffer[0]), 0, 0);
      } else {
        std::copy_n(send_ptr, buf_nums[buf_send_idx[i]], recv_ptr);
      }
      buf_nums[buf_recv_idx[i]] += buf_nums[buf_send_idx[i]];
    } else {
      MPI_Sendrecv(send_ptr, buf_nums[buf_send_idx[i]] * sizeof(send_buffer[0]),
                   MPI_BYTE, dst, i, recv_ptr,
                   recv_buffer.size() * sizeof(send_buffer[0]), MPI_BYTE, src,
                   i, m_cart, &stat_recv[i]);
      int num_recved = 0;
      MPI_Get_count(&stat_recv[i], MPI_BYTE, &num_recved);
      buf_nums[buf_recv_idx[i]] += num_recved / sizeof(recv_buffer[0]);
    }
  }

  //   for (int i = 0; i < buf_recv_idx.size(); i++) {
  //     auto &recv_buffer = buffers[buf_recv_idx[i]];
  //     if (src != dst || src != m_rank) {
  //     // if (true) {
  //       int num_recved = 0;
  //       MPI_Wait(&req_recv[i], &stat_recv[i]);
  //       MPI_Get_count(&stat_recv[i], MPI_BYTE, &num_recved);
  //       buf_nums[buf_recv_idx[i]] += num_recved / sizeof(recv_buffer[0]);
  //       // Logger::print_debug_all("recved is {}, buf_num + {}", num_recved,
  //       num_recved / sizeof(recv_buffer[0]));
  //     }
  // #if CUDA_ENABLED &&                                              \
//     (!USE_CUDA_AWARE_MPI || !defined(MPIX_CUDA_AWARE_SUPPORT) || \
//      !MPIX_CUDA_AWARE_SUPPORT)
  //     recv_buffer.copy_to_device();
  // #endif
  //   }
  for (int i = 0; i < buf_recv_idx.size(); i++) {
    buf_nums[buf_send_idx[i]] = 0;
  }
}

template <typename Conf, template <class> class ExecPolicy>
template <typename T>
void
domain_comm<Conf, ExecPolicy>::send_particle_array(
    T &send_buffer, int &send_num, T &recv_buffer, int &recv_num, int src,
    int dst, int tag, MPI_Request *send_req, MPI_Request *recv_req,
    MPI_Status *recv_stat) const {
  int recv_offset = recv_num;

  auto send_ptr = send_buffer.host_ptr();
  auto recv_ptr = recv_buffer.host_ptr();
  if constexpr (m_is_device && use_cuda_mpi) {
    send_ptr = send_buffer.dev_ptr();
    recv_ptr = recv_buffer.dev_ptr() + recv_offset;
  } else {
    send_buffer.copy_to_host();
  }

  // Logger::print_debug_all("Send count is {}, send size is {}, recv_size is
  // {}", send_num,
  //                       send_num * sizeof(send_buffer[0]),
  //                       recv_buffer.size() * sizeof(recv_buffer[0]));
  MPI_Sendrecv(send_ptr, send_num * sizeof(send_buffer[0]), MPI_BYTE, dst, tag,
               recv_ptr, recv_buffer.size() * sizeof(recv_buffer[0]), MPI_BYTE,
               src, tag, m_world, recv_stat);

  int num_recved = 0;
  MPI_Get_count(recv_stat, MPI_BYTE, &num_recved);
  // MPI_Get_count(recv_stat, MPI_Helper::get_mpi_datatype(recv_ptr[0]),
  // &num_recved);
  // Logger::print_debug_all("Rank {} received {}", m_rank,
  // num_recved);
  recv_num = recv_offset + num_recved / sizeof(recv_buffer[0]);

  // MPI_Barrier(m_cart);

  if constexpr (m_is_device && !use_cuda_mpi) {
    recv_buffer.copy_to_device();
  }
  // }
  // send_buffer.set_num(0);
  send_num = 0;
}

template <typename Conf, template <class> class ExecPolicy>
template <typename PtcType>
void
domain_comm<Conf, ExecPolicy>::send_particles_impl(
    PtcType &ptc, const grid_t<Conf> &grid) const {
  Logger::print_detail("Sending paticles");
  // timer::stamp("send_ptc");
  if (!m_buffers_ready) resize_buffers(grid);
  auto &buffers = ptc_buffers(ptc);
  auto &buf_ptrs = ptc_buffer_ptrs(ptc);
  auto &buf_nums = ptc_buffer_nums(ptc);
  buf_nums.assign_host(0);
  // timer::stamp("copy_comm");
  // Logger::print_detail("Copying to comm buffers");
  // ptc.copy_to_comm_buffers(exec_tag{}, buffers, buf_ptrs, buf_nums, grid);
  ptc_copy_to_comm_buffers(exec_tag{}, ptc, buffers, buf_ptrs, buf_nums, grid);
  // Logger::print_detail("Finished copying to comm buffers");
  // timer::show_duration_since_stamp("Coping to comm buffers", "ms",
  // "copy_comm");

  // Define the central zone and number of send_recv in x direction
  int central = 13;
  int num_send_x = 9;
  if (Conf::dim == 2) {
    central = 4;
    num_send_x = 3;
  } else if (Conf::dim == 1) {
    central = 1;
    num_send_x = 1;
  }

  std::vector<MPI_Request> req_send(num_send_x);
  std::vector<MPI_Request> req_recv(num_send_x);
  std::vector<MPI_Status> stat_recv(num_send_x);
  std::vector<int> vec_buf_send_x(num_send_x);
  std::vector<int> vec_buf_recv_x(num_send_x);
  // Send left in x
  for (int i = 0; i < num_send_x; i++) {
    vec_buf_send_x[i] = i * 3;
    vec_buf_recv_x[i] = i * 3 + 1;
    // int buf_send = i * 3;
    // int buf_recv = i * 3 + 1;
    // send_particle_array(buffers[buf_send], buf_nums[buf_send],
    //                     buffers[buf_recv], buf_nums[buf_recv],
    //                     m_domain_info.neighbor_right[0],
    //                     m_domain_info.neighbor_left[0], i, &req_send[i],
    //                     &req_recv[i], &stat_recv[i]);
  }
  send_particle_array(buffers, buf_nums, vec_buf_send_x, vec_buf_recv_x,
                      m_domain_info.neighbor_right[0],
                      m_domain_info.neighbor_left[0], req_send, req_recv,
                      stat_recv);
  // Send right in x
  for (int i = 0; i < num_send_x; i++) {
    vec_buf_send_x[i] = i * 3 + 2;
    vec_buf_recv_x[i] = i * 3 + 1;
    // int buf_send = i * 3 + 2;
    // int buf_recv = i * 3 + 1;
    // send_particle_array(buffers[buf_send], buf_nums[buf_send],
    //                     buffers[buf_recv], buf_nums[buf_recv],
    //                     m_domain_info.neighbor_left[0],
    //                     m_domain_info.neighbor_right[0], i, &req_send[i],
    //                     &req_recv[i], &stat_recv[i]);
  }
  send_particle_array(buffers, buf_nums, vec_buf_send_x, vec_buf_recv_x,
                      m_domain_info.neighbor_left[0],
                      m_domain_info.neighbor_right[0], req_send, req_recv,
                      stat_recv);

  // Send in y direction next
  if constexpr (Conf::dim >= 2) {
    int num_send_y = (Conf::dim == 2 ? 1 : 3);
    // if (Conf::dim == 2) num_send_y = 1;
    std::vector<int> vec_buf_send_y(num_send_y);
    std::vector<int> vec_buf_recv_y(num_send_y);
    // Send left in y
    for (int i = 0; i < num_send_y; i++) {
      vec_buf_send_y[i] = 1 + i * 9;
      vec_buf_recv_y[i] = 1 + 3 + i * 9;
      // int buf_send = 1 + i * 9;
      // int buf_recv = 1 + 3 + i * 9;
      // send_particle_array(buffers[buf_send], buf_nums[buf_send],
      //                     buffers[buf_recv], buf_nums[buf_recv],
      //                     m_domain_info.neighbor_right[1],
      //                     m_domain_info.neighbor_left[1], i, &req_send[i],
      //                     &req_recv[i], &stat_recv[i]);
    }
    send_particle_array(buffers, buf_nums, vec_buf_send_y, vec_buf_recv_y,
                        m_domain_info.neighbor_right[1],
                        m_domain_info.neighbor_left[1], req_send, req_recv,
                        stat_recv);

    // Send right in y
    for (int i = 0; i < num_send_y; i++) {
      vec_buf_send_y[i] = 1 + 6 + i * 9;
      vec_buf_recv_y[i] = 1 + 3 + i * 9;
      // int buf_send = 1 + 6 + i * 9;
      // int buf_recv = 1 + 3 + i * 9;
      // send_particle_array(buffers[buf_send], buf_nums[buf_send],
      //                     buffers[buf_recv], buf_nums[buf_recv],
      //                     m_domain_info.neighbor_left[1],
      //                     m_domain_info.neighbor_right[1], i, &req_send[i],
      //                     &req_recv[i], &stat_recv[i]);
    }
    send_particle_array(buffers, buf_nums, vec_buf_send_y, vec_buf_recv_y,
                        m_domain_info.neighbor_left[1],
                        m_domain_info.neighbor_right[1], req_send, req_recv,
                        stat_recv);

    // Finally send z direction
    if constexpr (Conf::dim == 3) {
      // Send left in z
      int buf_send = 4;
      int buf_recv = 13;
      send_particle_array(buffers[buf_send], buf_nums[buf_send],
                          buffers[buf_recv], buf_nums[buf_recv],
                          m_domain_info.neighbor_right[2],
                          m_domain_info.neighbor_left[2], 13, &req_send[0],
                          &req_recv[0], &stat_recv[0]);
      // Send right in z
      buf_send = 22;
      send_particle_array(buffers[buf_send], buf_nums[buf_send],
                          buffers[buf_recv], buf_nums[buf_recv],
                          m_domain_info.neighbor_left[2],
                          m_domain_info.neighbor_right[2], 13, &req_send[0],
                          &req_recv[0], &stat_recv[0]);
    }
  }

  // Copy the central recv buffer into the main array
  if constexpr (m_is_device && !use_cuda_mpi) {
    buffers[central].copy_to_device();
  }
  // ptc.copy_from(buffers[central], buffers[central].number(), 0,
  // ptc.number());
  // ptc.copy_from_buffer(exec_tag{}, buffers[central], buf_nums[central],
  // ptc.number());
  ptc_copy_from_buffer(exec_tag{}, ptc, buffers[central], buf_nums[central],
                       ptc.number());
  // Logger::print_debug_all(
  //     "Communication resulted in {} ptc in total, ptc has {} particles "
  //     "now",
  //     buf_nums[central], ptc.number());
  // for (unsigned int i = ptc.number() - 4; i < ptc.number(); i++) {
  //   auto c = ptc.cell[i];
  //   Logger::print_debug_all("c {}, cell {}, {}", c, c % grid.dims[0], c /
  //   grid.dims[0]);
  // }
  // buffers[central].set_num(0);
  buf_nums[central] = 0;
  // timer::show_duration_since_stamp("Send particles", "ms", "send_ptc");
}

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::send_particles(photons_t &ptc,
                                              const grid_t<Conf> &grid) const {
  send_particles_impl(ptc, grid);
}

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::send_particles(particles_t &ptc,
                                              const grid_t<Conf> &grid) const {
  send_particles_impl(ptc, grid);
}

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::get_total_num_offset(const uint64_t &num,
                                                    uint64_t &total,
                                                    uint64_t &offset) const {
  // Carry out an MPI scan to get the total number and local offset,
  // used for particle output into a file
  uint64_t result = 0;
  auto status =
      MPI_Scan(&num, &result, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  offset = result - num;
  total = 0;
  status =
      MPI_Allreduce(&num, &total, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Helper::handle_mpi_error(status, m_rank);
}

// template <typename Conf>
// template <typename T>
// void
// domain_comm<Conf>::gather_to_root(buffer<T> &buf) const {
//   buffer<T> tmp_buf(buf.size(), buf.mem_type());
//   // #if CUDA_ENABLED && (MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
//   //   auto result =
//   //       MPI_Reduce(buf.dev_ptr(), tmp_buf.dev_ptr(), buf.size(),
//   //                  MPI_Helper::get_mpi_datatype(T{}), MPI_SUM, 0,
//   //                  MPI_COMM_WORLD);
//   // #else
//   buf.copy_to_host();
//   auto result =
//       MPI_Reduce(buf.host_ptr(), tmp_buf.host_ptr(), buf.size(),
//                  MPI_Helper::get_mpi_datatype(T{}), MPI_SUM, 0,
//                  MPI_COMM_WORLD);
//   // buf.copy_to_device();
//   // #endif
//   if (is_root()) {
//     buf.host_copy_from(tmp_buf, buf.size());
//     // buf.copy_to_host();
//   }
// }

template <typename Conf, template <class> class ExecPolicy>
std::vector<buffer<single_ptc_t>> &
domain_comm<Conf, ExecPolicy>::ptc_buffers(const particles_t &ptc) const {
  return m_ptc_buffers;
}

template <typename Conf, template <class> class ExecPolicy>
std::vector<buffer<single_ph_t>> &
domain_comm<Conf, ExecPolicy>::ptc_buffers(const photons_t &ptc) const {
  return m_ph_buffers;
}

template <typename Conf, template <class> class ExecPolicy>
buffer<single_ptc_t *> &
domain_comm<Conf, ExecPolicy>::ptc_buffer_ptrs(const particles_t &ptc) const {
  return m_ptc_buffer_ptrs;
}

template <typename Conf, template <class> class ExecPolicy>
buffer<single_ph_t *> &
domain_comm<Conf, ExecPolicy>::ptc_buffer_ptrs(const photons_t &ph) const {
  return m_ph_buffer_ptrs;
}

// Explicitly instantiate some of the configurations that may occur
// template class domain_comm<Config<1>>;
// INSTANTIATE_WITH_CONFIG(domain_comm);
// template void domain_comm<Config<1, float>>::gather_to_root(
//     buffer<float> &buf) const;
// template void domain_comm<Config<1, float>>::gather_to_root(
//     buffer<double> &buf) const;
// template void domain_comm<Config<1, double>>::gather_to_root(
//     buffer<float> &buf) const;
// template void domain_comm<Config<1, double>>::gather_to_root(
//     buffer<double> &buf) const;
// // template class domain_comm<Config<2>>;
// template void domain_comm<Config<2, float>>::gather_to_root(
//     buffer<float> &buf) const;
// template void domain_comm<Config<2, float>>::gather_to_root(
//     buffer<double> &buf) const;
// template void domain_comm<Config<2, double>>::gather_to_root(
//     buffer<float> &buf) const;
// template void domain_comm<Config<2, double>>::gather_to_root(
//     buffer<double> &buf) const;
// // template class domain_comm<Config<3>>;
// template void domain_comm<Config<3, float>>::gather_to_root(
//     buffer<float> &buf) const;
// template void domain_comm<Config<3, float>>::gather_to_root(
//     buffer<double> &buf) const;
// template void domain_comm<Config<3, double>>::gather_to_root(
//     buffer<float> &buf) const;
// template void domain_comm<Config<3, double>>::gather_to_root(
//     buffer<double> &buf) const;

}  // namespace Aperture
