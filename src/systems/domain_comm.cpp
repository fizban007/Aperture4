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

#include "domain_comm.h"
#include "core/constant_mem_func.h"
#include "core/detail/multi_array_helpers.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "framework/params_store.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/mpi_helper.h"

#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>  // Needed for CUDA-aware check
#endif

namespace Aperture {

template <typename Conf>
domain_comm<Conf>::domain_comm(sim_environment &env) : system_t(env) {
  setup_domain();
}

template <typename Conf>
domain_comm<Conf>::~domain_comm() {}

template <typename Conf>
void
domain_comm<Conf>::setup_domain() {
  m_world = MPI_COMM_WORLD;
  MPI_Comm_rank(m_world, &m_rank);
  MPI_Comm_size(m_world, &m_size);

#ifdef CUDA_ENABLED
  init_dev_rank(m_rank);
#endif

  m_scalar_type = MPI_Helper::get_mpi_datatype(typename Conf::value_t{});

  // This is the first place where rank is defined. Tell logger about
  // this
  Logger::init(m_rank, (LogLevel)m_env.params().template get_as<int64_t>(
                           "log_level", (int64_t)LogLevel::info));

  auto dims = m_env.params().template get_as<std::vector<int64_t>>("nodes");
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

  auto periodic =
      m_env.params().template get_as<std::vector<bool>>("periodic_boundary");
  for (int i = 0; i < std::min(Conf::dim, (int)periodic.size()); i++)
    m_domain_info.is_periodic[i] = periodic[i];

  // Create a cartesian MPI group for communication
  MPI_Cart_create(m_world, Conf::dim, m_domain_info.mpi_dims,
                  m_domain_info.is_periodic, true, &m_cart);

  // Obtain the mpi coordinate of the current rank
  MPI_Cart_coords(m_cart, m_rank, Conf::dim, m_domain_info.mpi_coord);

  // Figure out if the current rank is at any boundary
  int left = 0, right = 0;
  int rank = 0;
  for (int n = 0; n < Conf::dim; n++) {
    MPI_Cart_shift(m_cart, n, -1, &rank, &left);
    MPI_Cart_shift(m_cart, n, 1, &rank, &right);
    m_domain_info.neighbor_left[n] = left;
    m_domain_info.neighbor_right[n] = right;
    if (left < 0) m_domain_info.is_boundary[2 * n] = true;
    if (right < 0) m_domain_info.is_boundary[2 * n + 1] = true;
  }

#ifdef CUDA_ENABLED
  // Poll the system to detect how many GPUs are on the node, set the
  // GPU corresponding to the rank
  int n_devices;
  cudaGetDeviceCount(&n_devices);
  if (n_devices <= 0) {
    std::cerr << "No usable Cuda device found!!" << std::endl;
    exit(1);
  }
  // TODO: This way of finding device id may not be reliable
  int dev_id = m_rank % n_devices;
  cudaSetDevice(dev_id);
#endif
}

template <typename Conf>
void
domain_comm<Conf>::resize_buffers(const Grid<Conf::dim> &grid) const {
  if (m_buffers_ready) return;
  for (int i = 0; i < Conf::dim; i++) {
    auto ext = extent_t<Conf::dim>{};
    for (int j = 0; j < Conf::dim; j++) {
      if (j == i)
        ext[j] = grid.guard[j];
      else
        ext[j] = grid.dims[j];
    }
    m_send_buffers.emplace_back(ext);
    m_recv_buffers.emplace_back(ext);
  }

  size_t ptc_buffer_size =
      m_env.params().template get_as<int64_t>("ptc_buffer_size", 100000l);
  size_t ph_buffer_size =
      m_env.params().template get_as<int64_t>("ph_buffer_size", 100000l);
  int num_ptc_buffers = std::pow(3, Conf::dim);
  for (int i = 0; i < num_ptc_buffers; i++) {
    m_ptc_buffers.emplace_back(ptc_buffer_size);
    m_ph_buffers.emplace_back(ph_buffer_size);
  }
  m_ptc_buffer_ptrs.resize(num_ptc_buffers);
  m_ph_buffer_ptrs.resize(num_ptc_buffers);
  for (int i = 0; i < num_ptc_buffers; i++) {
    m_ptc_buffer_ptrs[i] = m_ptc_buffers[i].dev_ptrs();
    m_ph_buffer_ptrs[i] = m_ph_buffers[i].dev_ptrs();
  }
  m_ptc_buffer_ptrs.copy_to_device();
  m_ph_buffer_ptrs.copy_to_device();
  // Logger::print_debug("m_ptc_buffers has size {}", m_ptc_buffers.size());
  m_buffers_ready = true;
}

template <typename Conf>
void
domain_comm<Conf>::send_array_guard_cells_single_dir(
    typename Conf::multi_array_t &array, const Grid<Conf::dim> &grid, int dim,
    int dir) const {
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

  if (array.mem_type() == MemType::host_only) {
    copy(m_send_buffers[dim], array, index_t<Conf::dim>{}, send_idx,
         m_send_buffers[dim].extent());
  } else {
    copy_dev(m_send_buffers[dim], array, index_t<Conf::dim>{}, send_idx,
             m_send_buffers[dim].extent());
  }

#if CUDA_ENABLED && (MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  auto send_ptr = m_send_buffers[dim].dev_ptr();
  auto recv_ptr = m_recv_buffers[dim].dev_ptr();
#else
  auto send_ptr = m_send_buffers[dim].host_ptr();
  auto recv_ptr = m_recv_buffers[dim].host_ptr();
  m_send_buffers[dim].copy_to_host();
#endif

  MPI_Sendrecv(send_ptr, m_send_buffers[dim].size(), m_scalar_type, dest, 0,
               recv_ptr, m_recv_buffers[dim].size(), m_scalar_type, origin, 0,
               m_cart, &status);
  // MPI_Request req_send, req_recv;

  // MPI_Irecv(recv_ptr, m_recv_buffers[dim].size(), m_scalar_type, origin,
  //           0, m_world, &req_recv);
  // MPI_Isend(send_ptr, m_send_buffers[dim].size(), m_scalar_type, dest,
  //           0, m_world, &req_send);
  // MPI_Wait(&req_recv, &status);

  if (origin != MPI_PROC_NULL) {
    // Index recv_idx(0, 0, 0);
    auto recv_idx = index_t<Conf::dim>{};
    recv_idx[dim] = (dir == -1 ? grid.dims[dim] - grid.guard[dim] : 0);

    if (array.mem_type() == MemType::host_only) {
      copy(array, m_recv_buffers[dim], recv_idx, index_t<Conf::dim>{},
           m_recv_buffers[dim].extent());
    } else {
#if CUDA_ENABLED && !defined(MPIX_CUDA_AWARE_SUPPORT) || \
    !MPIX_CUDA_AWARE_SUPPORT
      m_recv_buffers[dim].copy_to_device();
#endif
      copy_dev(array, m_recv_buffers[dim], recv_idx, index_t<Conf::dim>{},
               m_recv_buffers[dim].extent());
    }
  }
}

template <typename Conf>
void
domain_comm<Conf>::send_add_array_guard_cells_single_dir(
    typename Conf::multi_array_t &array, const Grid<Conf::dim> &grid, int dim,
    int dir) const {
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

  if (array.mem_type() == MemType::host_only) {
    copy(m_send_buffers[dim], array, index_t<Conf::dim>{}, send_idx,
         m_send_buffers[dim].extent());
  } else {
    copy_dev(m_send_buffers[dim], array, index_t<Conf::dim>{}, send_idx,
             m_send_buffers[dim].extent());
  }

#if CUDA_ENABLED && (MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  auto send_ptr = m_send_buffers[dim].dev_ptr();
  auto recv_ptr = m_recv_buffers[dim].dev_ptr();
#else
  auto send_ptr = m_send_buffers[dim].host_ptr();
  auto recv_ptr = m_recv_buffers[dim].host_ptr();
  m_send_buffers[dim].copy_to_host();
#endif

  MPI_Sendrecv(send_ptr, m_send_buffers[dim].size(), m_scalar_type, dest, 0,
               recv_ptr, m_recv_buffers[dim].size(), m_scalar_type, origin, 0,
               m_cart, &status);
  // MPI_Request req_send, req_recv;

  // MPI_Irecv(recv_ptr, m_recv_buffers[dim].size(), m_scalar_type, origin,
  //           0, m_world, &req_recv);
  // MPI_Isend(send_ptr, m_send_buffers[dim].size(), m_scalar_type, dest,
  //           0, m_world, &req_send);
  // MPI_Wait(&req_recv, &status);

  if (origin != MPI_PROC_NULL) {
    // Index recv_idx(0, 0, 0);
    auto recv_idx = index_t<Conf::dim>{};
    recv_idx[dim] =
        (dir == -1 ? grid.dims[dim] - 2 * grid.guard[dim] : grid.guard[dim]);

    if (array.mem_type() == MemType::host_only) {
      add(array, m_recv_buffers[dim], recv_idx, index_t<Conf::dim>{},
          m_recv_buffers[dim].extent());
    } else {
#if CUDA_ENABLED && !defined(MPIX_CUDA_AWARE_SUPPORT) || \
    !MPIX_CUDA_AWARE_SUPPORT
      m_recv_buffers[dim].copy_to_device();
#endif
      add_dev(array, m_recv_buffers[dim], recv_idx, index_t<Conf::dim>{},
              m_recv_buffers[dim].extent());
    }
  }
}

template <typename Conf>
void
domain_comm<Conf>::send_guard_cells(vector_field<Conf> &field) const {
  if (!m_buffers_ready) resize_buffers(field.grid());
  send_guard_cells(field[0], field.grid());
  send_guard_cells(field[1], field.grid());
  send_guard_cells(field[2], field.grid());
}

template <typename Conf>
void
domain_comm<Conf>::send_guard_cells(scalar_field<Conf> &field) const {
  if (!m_buffers_ready) resize_buffers(field.grid());
  send_guard_cells(field[0], field.grid());
}

template <typename Conf>
void
domain_comm<Conf>::send_guard_cells(typename Conf::multi_array_t &array,
                                    const Grid<Conf::dim> &grid) const {
  send_array_guard_cells_single_dir(array, grid, 0, -1);
  send_array_guard_cells_single_dir(array, grid, 0, 1);
  send_array_guard_cells_single_dir(array, grid, 1, -1);
  send_array_guard_cells_single_dir(array, grid, 1, 1);
  send_array_guard_cells_single_dir(array, grid, 2, -1);
  send_array_guard_cells_single_dir(array, grid, 2, 1);
}

template <typename Conf>
void
domain_comm<Conf>::send_add_guard_cells(vector_field<Conf> &field) const {
  if (!m_buffers_ready) resize_buffers(field.grid());
  send_add_guard_cells(field[0], field.grid());
  send_add_guard_cells(field[1], field.grid());
  send_add_guard_cells(field[2], field.grid());
}

template <typename Conf>
void
domain_comm<Conf>::send_add_guard_cells(scalar_field<Conf> &field) const {
  if (!m_buffers_ready) resize_buffers(field.grid());
  send_add_guard_cells(field[0], field.grid());
}

template <typename Conf>
void
domain_comm<Conf>::send_add_guard_cells(typename Conf::multi_array_t &array,
                                        const Grid<Conf::dim> &grid) const {
  send_add_array_guard_cells_single_dir(array, grid, 0, -1);
  send_add_array_guard_cells_single_dir(array, grid, 0, 1);
  send_add_array_guard_cells_single_dir(array, grid, 1, -1);
  send_add_array_guard_cells_single_dir(array, grid, 1, 1);
  send_add_array_guard_cells_single_dir(array, grid, 2, -1);
  send_add_array_guard_cells_single_dir(array, grid, 2, 1);
}

template <typename Conf>
template <typename T>
void
domain_comm<Conf>::send_particle_array(T &send_buffer, T &recv_buffer, int src,
                                       int dst, int tag, MPI_Request *send_req,
                                       MPI_Request *recv_req,
                                       MPI_Status *recv_stat) const {
  // TODO: Detect cuda-aware MPI and use that accordingly
  int recv_offset = recv_buffer.number();
  int num_send = send_buffer.number();
#if CUDA_ENABLED && (MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  auto send_ptrs = send_buffer.get_dev_ptrs();
  auto recv_ptrs = recv_buffer.get_dev_ptrs();
#else
  send_buffer.copy_to_host();
  auto send_ptrs = send_buffer.get_host_ptrs();
  auto recv_ptrs = recv_buffer.get_host_ptrs();
#endif
  // if (num_send > 0) {
  //   Logger::print_debug("Send count is {}, send cell[0] is {}, send cell[1] is {}",
  //                       num_send, send_buffer.cell[0], send_buffer.cell[1]);
  // }
  int num_recv = 0;
  visit_struct::for_each(
      send_ptrs, recv_ptrs,
      [&](const char *name, auto &u, auto &v) {
        // MPI_Irecv((void*)(v + recv_offset), recv_buffer.size(),
        //           MPI_Helper::get_mpi_datatype(v[0]), src, tag,
        //           m_cart, recv_req);
        // MPI_Isend((void*)u, num_send,
        //           MPI_Helper::get_mpi_datatype(u[0]), dst, tag,
        //           m_cart, send_req);
        MPI_Sendrecv((void *)u, num_send, MPI_Helper::get_mpi_datatype(u[0]),
                     dst, tag, (void *)(v + recv_offset), recv_buffer.size(),
                     MPI_Helper::get_mpi_datatype(v[0]), src, tag, m_world,
                     recv_stat);
        // MPI_Wait(recv_req, recv_stat);
        if (strcmp(name, "cell") == 0 && src != MPI_PROC_NULL) {
          // if (num_send > 0) {
            // Logger::print_debug("Send count is {}, send cell[0] is {}",
            //                     num_send, u[0]);
          // }
          MPI_Get_count(recv_stat, MPI_Helper::get_mpi_datatype(v[0]),
                        &num_recv);
        }
      });
#if CUDA_ENABLED && (MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
#else
  recv_buffer.copy_to_device();
#endif
  recv_buffer.set_num(recv_offset + num_recv);
  send_buffer.set_num(0);
}

template <typename Conf>
template <typename PtcType>
void
domain_comm<Conf>::send_particles_impl(PtcType &ptc,
                                       const grid_t<Conf> &grid) const {
  Logger::print_info("Sending paticles");
  timer::stamp("send_ptc");
  if (!m_buffers_ready) resize_buffers(grid);
  auto &buffers = ptc_buffers(ptc);
  auto &buf_ptrs = ptc_buffer_ptrs(ptc);
  // timer::stamp("copy_comm");
  ptc.copy_to_comm_buffers(buffers, buf_ptrs, grid);
  // timer::show_duration_since_stamp("Coping to comm buffers", "ms", "copy_comm");

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

  // Send left in x
  std::vector<MPI_Request> req_send(num_send_x);
  std::vector<MPI_Request> req_recv(num_send_x);
  std::vector<MPI_Status> stat_recv(num_send_x);
  for (int i = 0; i < num_send_x; i++) {
    int buf_send = i * 3;
    int buf_recv = i * 3 + 1;
    send_particle_array(buffers[buf_send], buffers[buf_recv],
                        m_domain_info.neighbor_right[0],
                        m_domain_info.neighbor_left[0], i, &req_send[i],
                        &req_recv[i], &stat_recv[i]);
  }
  // Send right in x
  for (int i = 0; i < num_send_x; i++) {
    int buf_send = i * 3 + 2;
    int buf_recv = i * 3 + 1;
    send_particle_array(buffers[buf_send], buffers[buf_recv],
                        m_domain_info.neighbor_left[0],
                        m_domain_info.neighbor_right[0], i, &req_send[i],
                        &req_recv[i], &stat_recv[i]);
  }

  // Send in y direction next
  if constexpr (Conf::dim >= 2) {
    int num_send_y = 3;
    if (Conf::dim == 2) num_send_y = 1;
    // Send left in y
    for (int i = 0; i < num_send_y; i++) {
      int buf_send = 1 + i * 9;
      int buf_recv = 1 + 3 + i * 9;
      send_particle_array(buffers[buf_send], buffers[buf_recv],
                          m_domain_info.neighbor_right[1],
                          m_domain_info.neighbor_left[1], i, &req_send[i],
                          &req_recv[i], &stat_recv[i]);
    }
    // Send right in y
    for (int i = 0; i < num_send_y; i++) {
      int buf_send = 1 + 6 + i * 9;
      int buf_recv = 1 + 3 + i * 9;
      send_particle_array(buffers[buf_send], buffers[buf_recv],
                          m_domain_info.neighbor_left[1],
                          m_domain_info.neighbor_right[1], i, &req_send[i],
                          &req_recv[i], &stat_recv[i]);
    }

    // Finally send z direction
    if constexpr (Conf::dim == 3) {
      // Send left in z
      int buf_send = 4;
      int buf_recv = 13;
      send_particle_array(buffers[buf_send], buffers[buf_recv],
                          m_domain_info.neighbor_right[2],
                          m_domain_info.neighbor_left[2], 0, &req_send[0],
                          &req_recv[0], &stat_recv[0]);
      // Send right in z
      buf_send = 22;
      send_particle_array(buffers[buf_send], buffers[buf_recv],
                          m_domain_info.neighbor_left[2],
                          m_domain_info.neighbor_right[2], 0, &req_send[0],
                          &req_recv[0], &stat_recv[0]);
    }
  }

  // Copy the central recv buffer into the main array
  ptc.copy_from(buffers[central], buffers[central].number(), 0, ptc.number());
  // Logger::print_debug(
  //     "Communication resulted in {} ptc in total, ptc has {} particles "
  //     "now",
  //     buffers[central].number(), ptc.number());
  buffers[central].set_num(0);
  timer::show_duration_since_stamp("Send particles", "ms", "send_ptc");
}

template <typename Conf>
void
domain_comm<Conf>::send_particles(photons_t &ptc,
                                  const grid_t<Conf> &grid) const {
  send_particles_impl(ptc, grid);
}

template <typename Conf>
void
domain_comm<Conf>::send_particles(particles_t &ptc,
                                  const grid_t<Conf> &grid) const {
  send_particles_impl(ptc, grid);
}

template <typename Conf>
void
domain_comm<Conf>::get_total_num_offset(uint64_t &num, uint64_t &total,
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

template <typename Conf>
template <typename T>
void
domain_comm<Conf>::gather_to_root(buffer<T>& buf) const {
  buffer<T> tmp_buf(buf.size(), buf.mem_type());
// #if CUDA_ENABLED && (MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
//   auto result =
//       MPI_Reduce(buf.dev_ptr(), tmp_buf.dev_ptr(), buf.size(),
//                  MPI_Helper::get_mpi_datatype(T{}), MPI_SUM, 0,
//                  MPI_COMM_WORLD);
// #else
  buf.copy_to_host();
  auto result =
      MPI_Reduce(buf.host_ptr(), tmp_buf.host_ptr(), buf.size(),
                 MPI_Helper::get_mpi_datatype(T{}), MPI_SUM, 0,
                 MPI_COMM_WORLD);
  // buf.copy_to_device();
// #endif
  if (is_root()) {
    buf.host_copy_from(tmp_buf, buf.size());
    // buf.copy_to_host();
  }
}

template <typename Conf>
std::vector<particles_t> &
domain_comm<Conf>::ptc_buffers(const particles_t &ptc) const {
  return m_ptc_buffers;
}

template <typename Conf>
std::vector<photons_t> &
domain_comm<Conf>::ptc_buffers(const photons_t &ptc) const {
  return m_ph_buffers;
}

template <typename Conf>
buffer<ptc_ptrs> &
domain_comm<Conf>::ptc_buffer_ptrs(const particles_t &ptc) const {
  return m_ptc_buffer_ptrs;
}

template <typename Conf>
buffer<ph_ptrs> &
domain_comm<Conf>::ptc_buffer_ptrs(const photons_t &ph) const {
  return m_ph_buffer_ptrs;
}

// Explicitly instantiate some of the configurations that may occur
template class domain_comm<Config<1>>;
template void domain_comm<Config<1>>::gather_to_root(buffer<float>& buf) const;
template void domain_comm<Config<1>>::gather_to_root(buffer<double>& buf) const;
template class domain_comm<Config<2>>;
template void domain_comm<Config<2>>::gather_to_root(buffer<float>& buf) const;
template void domain_comm<Config<2>>::gather_to_root(buffer<double>& buf) const;
template class domain_comm<Config<3>>;
template void domain_comm<Config<3>>::gather_to_root(buffer<float>& buf) const;
template void domain_comm<Config<3>>::gather_to_root(buffer<double>& buf) const;

}  // namespace Aperture
