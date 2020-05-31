#include "domain_comm_async.h"
#include "core/constant_mem_func.h"
#include "core/detail/multi_array_helpers.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "framework/params_store.h"
#include "utils/logger.h"
#include "utils/mpi_helper.h"

#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>  // Needed for CUDA-aware check
#endif

namespace Aperture {

template <typename Conf>
domain_comm_async<Conf>::domain_comm_async(sim_environment &env)
    : domain_comm<Conf>(env) {
  CudaSafeCall(cudaStreamCreate(&m_copy_stream));
}

template <typename Conf>
domain_comm_async<Conf>::~domain_comm_async() {
  CudaSafeCall(cudaStreamDestroy(m_copy_stream));
}

template <typename Conf>
void
domain_comm_async<Conf>::send_array_guard_cells_single_dir_async(
    typename Conf::multi_array_t &array, const Grid<Conf::dim> &grid, int dim,
    int dir) const {
  if (dim < 0 || dim >= Conf::dim) return;

  int dest, origin;
  MPI_Status status;

  dest = (dir == -1 ? this->m_domain_info.neighbor_left[dim]
                    : this->m_domain_info.neighbor_right[dim]);
  origin = (dir == -1 ? this->m_domain_info.neighbor_right[dim]
                      : this->m_domain_info.neighbor_left[dim]);

  // Index send_idx(0, 0, 0);
  auto send_idx = index_t<Conf::dim>{};
  send_idx[dim] =
      (dir == -1 ? grid.guard[dim] : grid.dims[dim] - 2 * grid.guard[dim]);

  if (array.mem_type() == MemType::host_only) {
    copy(this->m_send_buffers[dim], array, index_t<Conf::dim>{}, send_idx,
         this->m_send_buffers[dim].extent());
  } else {
    copy_dev(this->m_send_buffers[dim], array, index_t<Conf::dim>{}, send_idx,
             this->m_send_buffers[dim].extent(), &m_copy_stream);
  }

#if CUDA_ENABLED && (MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  auto send_ptr = this->m_send_buffers[dim].dev_ptr();
  auto recv_ptr = this->m_recv_buffers[dim].dev_ptr();
#else
  auto send_ptr = this->m_send_buffers[dim].host_ptr();
  auto recv_ptr = this->m_recv_buffers[dim].host_ptr();
  this->m_send_buffers[dim].copy_to_host(m_copy_stream);
#endif

  CudaSafeCall(cudaStreamSynchronize(m_copy_stream));

  MPI_Sendrecv(send_ptr, this->m_send_buffers[dim].size(), this->m_scalar_type,
               dest, 0, recv_ptr, this->m_recv_buffers[dim].size(),
               this->m_scalar_type, origin, 0, this->m_cart, &status);
  // MPI_Request req_send, req_recv;

  // MPI_Irecv(recv_ptr, this->m_recv_buffers[dim].size(), this->m_scalar_type,
  // origin,
  //           0, this->m_world, &req_recv);
  // MPI_Isend(send_ptr, this->m_send_buffers[dim].size(), this->m_scalar_type,
  // dest,
  //           0, this->m_world, &req_send);
  // MPI_Wait(&req_recv, &status);

  if (origin != MPI_PROC_NULL) {
    // Index recv_idx(0, 0, 0);
    auto recv_idx = index_t<Conf::dim>{};
    recv_idx[dim] = (dir == -1 ? grid.dims[dim] - grid.guard[dim] : 0);

    if (array.mem_type() == MemType::host_only) {
      copy(array, this->m_recv_buffers[dim], recv_idx, index_t<Conf::dim>{},
           this->m_recv_buffers[dim].extent());
    } else {
#if CUDA_ENABLED && !defined(MPIX_CUDA_AWARE_SUPPORT) || \
    !MPIX_CUDA_AWARE_SUPPORT
      this->m_recv_buffers[dim].copy_to_device(m_copy_stream);
#endif
      copy_dev(array, this->m_recv_buffers[dim], recv_idx, index_t<Conf::dim>{},
               this->m_recv_buffers[dim].extent(), &m_copy_stream);
      CudaSafeCall(cudaStreamSynchronize(m_copy_stream));
    }
  }
}

template <typename Conf>
void
domain_comm_async<Conf>::send_add_array_guard_cells_single_dir_async(
      typename Conf::multi_array_t& array, const Grid<Conf::dim>& grid, int dim,
      int dir) const {
  if (dim < 0 || dim >= Conf::dim) return;

  int dest, origin;
  MPI_Status status;

  dest = (dir == -1 ? this->m_domain_info.neighbor_left[dim]
                    : this->m_domain_info.neighbor_right[dim]);
  origin = (dir == -1 ? this->m_domain_info.neighbor_right[dim]
                      : this->m_domain_info.neighbor_left[dim]);

  // Index send_idx(0, 0, 0);
  auto send_idx = index_t<Conf::dim>{};
  send_idx[dim] = (dir == -1 ? 0 : grid.dims[dim] - grid.guard[dim]);

  if (array.mem_type() == MemType::host_only) {
    add(this->m_send_buffers[dim], array, index_t<Conf::dim>{}, send_idx,
        this->m_send_buffers[dim].extent());
  } else {
    add_dev(this->m_send_buffers[dim], array, index_t<Conf::dim>{}, send_idx,
            this->m_send_buffers[dim].extent(), Conf::value(1.0), &m_copy_stream);
  }

#if CUDA_ENABLED && (MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  auto send_ptr = this->m_send_buffers[dim].dev_ptr();
  auto recv_ptr = this->m_recv_buffers[dim].dev_ptr();
#else
  auto send_ptr = this->m_send_buffers[dim].host_ptr();
  auto recv_ptr = this->m_recv_buffers[dim].host_ptr();
  this->m_send_buffers[dim].copy_to_host();
#endif

  CudaSafeCall(cudaStreamSynchronize(m_copy_stream));

  MPI_Sendrecv(send_ptr, this->m_send_buffers[dim].size(), this->m_scalar_type, dest, 0,
               recv_ptr, this->m_recv_buffers[dim].size(), this->m_scalar_type, origin, 0,
               this->m_cart, &status);
  // MPI_Request req_send, req_recv;

  // MPI_Irecv(recv_ptr, this->m_recv_buffers[dim].size(), this->m_scalar_type, origin,
  //           0, this->m_world, &req_recv);
  // MPI_Isend(send_ptr, this->m_send_buffers[dim].size(), this->m_scalar_type, dest,
  //           0, this->m_world, &req_send);
  // MPI_Wait(&req_recv, &status);

  if (origin != MPI_PROC_NULL) {
    // Index recv_idx(0, 0, 0);
    auto recv_idx = index_t<Conf::dim>{};
    recv_idx[dim] =
        (dir == -1 ? grid.dims[dim] - 2 * grid.guard[dim] : grid.guard[dim]);

    if (array.mem_type() == MemType::host_only) {
      add(array, this->m_recv_buffers[dim], recv_idx, index_t<Conf::dim>{},
          this->m_recv_buffers[dim].extent());
    } else {
#if CUDA_ENABLED && !defined(MPIX_CUDA_AWARE_SUPPORT) || \
    !MPIX_CUDA_AWARE_SUPPORT
      this->m_recv_buffers[dim].copy_to_device(m_copy_stream);
#endif
      add_dev(array, this->m_recv_buffers[dim], recv_idx, index_t<Conf::dim>{},
              this->m_recv_buffers[dim].extent(), Conf::value(1.0), &m_copy_stream);
    }
  }
}

// template <typename Conf>
// void
// domain_comm_async<Conf>::send_guard_cells(vector_field<Conf> &field) const {
//   if (!this->m_buffers_ready) this->resize_buffers(field.grid());
//   send_guard_cells(field[0], field.grid());
//   send_guard_cells(field[1], field.grid());
//   send_guard_cells(field[2], field.grid());
// }

// template <typename Conf>
// void
// domain_comm_async<Conf>::send_guard_cells(scalar_field<Conf> &field) const {
//   if (!this->m_buffers_ready) this->resize_buffers(field.grid());
//   send_guard_cells(field[0], field.grid());
// }

template <typename Conf>
void
domain_comm_async<Conf>::send_add_guard_cells(typename Conf::multi_array_t &array,
                                              const Grid<Conf::dim> &grid) const {
  // Logger::print_debug("Send adding guard cells async!");
  send_add_array_guard_cells_single_dir_async(array, grid, 0, -1);
  send_add_array_guard_cells_single_dir_async(array, grid, 0, 1);
  send_add_array_guard_cells_single_dir_async(array, grid, 1, -1);
  send_add_array_guard_cells_single_dir_async(array, grid, 1, 1);
  send_add_array_guard_cells_single_dir_async(array, grid, 2, -1);
  send_add_array_guard_cells_single_dir_async(array, grid, 2, 1);
}

template <typename Conf>
void
domain_comm_async<Conf>::send_guard_cells(typename Conf::multi_array_t &array,
                                          const Grid<Conf::dim> &grid) const {
  // Logger::print_debug("Sending guard cells async!");
  send_array_guard_cells_single_dir_async(array, grid, 0, -1);
  send_array_guard_cells_single_dir_async(array, grid, 0, 1);
  send_array_guard_cells_single_dir_async(array, grid, 1, -1);
  send_array_guard_cells_single_dir_async(array, grid, 1, 1);
  send_array_guard_cells_single_dir_async(array, grid, 2, -1);
  send_array_guard_cells_single_dir_async(array, grid, 2, 1);
}

// Explicitly instantiate some of the configurations that may occur
INSTANTIATE_WITH_CONFIG(domain_comm_async);

}  // namespace Aperture
