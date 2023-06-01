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

#include "core/buffer.hpp"
#include "core/domain_info.h"
#include "core/particles.h"
#include "data/fields.h"
#include "data/phase_space.hpp"
#include "framework/system.h"
#include "utils/mpi_helper.h"
#include <mpi.h>
#include <vector>

namespace Aperture {

template <int Dim, typename value_t>
struct Grid;

template <typename Conf, template <class> class ExecPolicy>
class domain_comm : public system_t {
 public:
  static std::string name() { return "domain_comm"; }
  using exec_tag = typename ExecPolicy<Conf>::exec_tag;

  // domain_comm(sim_environment& env);
  domain_comm(int* argc = nullptr, char*** argv = nullptr);
  virtual ~domain_comm();

  void barrier() const { MPI_Barrier(m_world); }

  bool is_root() const { return m_rank == 0; }
  int rank() const { return m_rank; }
  int size() const { return m_size; }
  void resize_buffers(const typename Conf::grid_t& grid) const;
  void resize_phase_space_buffers(const typename Conf::grid_t& grid) const;

  void send_guard_cells(vector_field<Conf>& field) const;
  void send_guard_cells(scalar_field<Conf>& field) const;
  virtual void send_guard_cells(typename Conf::multi_array_t& array,
                                const typename Conf::grid_t& grid) const;
  void send_add_guard_cells(vector_field<Conf>& field) const;
  void send_add_guard_cells(scalar_field<Conf>& field) const;
  virtual void send_add_guard_cells(typename Conf::multi_array_t& array,
                                    const typename Conf::grid_t& grid) const;
  virtual void send_particles(particles_t& ptc, const grid_t<Conf>& grid) const;
  virtual void send_particles(photons_t& ptc, const grid_t<Conf>& grid) const;
  template <int Dim>
  void send_phase_space(phase_space<Conf, Dim>& data, const grid_t<Conf>& grid) const;
  void get_total_num_offset(const uint64_t& num, uint64_t& total,
                            uint64_t& offset) const;

  template <typename T>
  void gather_to_root(buffer<T>& buf) const {
    buffer<T> tmp_buf(buf.size(), MemType::host_only);
    buf.copy_to_host();
    auto result =
        MPI_Reduce(buf.host_ptr(), tmp_buf.host_ptr(), buf.size(),
                   MPI_Helper::get_mpi_datatype(T{}), MPI_SUM, 0, m_cart);
    if (is_root()) {
      buf.host_copy_from(tmp_buf, buf.size());
    }
  }

  const domain_info_t<Conf::dim>& domain_info() const { return m_domain_info; }

 protected:
  MPI_Comm m_world;
  MPI_Comm m_cart;
  MPI_Datatype m_scalar_type;

  int m_rank = 0;  ///< Rank of current process
  int m_size = 1;  ///< Size of MPI_COMM_WORLD
  domain_info_t<Conf::dim> m_domain_info;
  mutable bool m_buffers_ready = false;
  mutable bool m_phase_buffers_ready = false;
  static constexpr bool m_is_device =
      std::is_same<typename ExecPolicy<Conf>::exec_tag, exec_tags::device>::value;

  // Communication buffers. These buffers are declared mutable because we want
  // to use a const domain_comm reference to invoke communications, but
  // communication will necessarily need to modify these buffers.
  typedef typename Conf::multi_array_t multi_array_t;

  mutable std::vector<multi_array_t> m_send_buffers;
  mutable std::vector<multi_array_t> m_recv_buffers;
  mutable std::vector<multi_array_t> m_send_vec_buffers;
  mutable std::vector<multi_array_t> m_recv_vec_buffers;
  // mutable std::vector<multi_array_t> m_send_phase_space_buffers;
  // mutable std::vector<multi_array_t> m_recv_phase_space_buffers;
  // mutable std::vector<particles_t> m_ptc_buffers;
  // mutable std::vector<photons_t> m_ph_buffers;
  mutable std::vector<buffer<single_ptc_t>> m_ptc_buffers;
  mutable std::vector<buffer<single_ph_t>> m_ph_buffers;
  mutable buffer<int> m_ptc_buffer_num;
  mutable buffer<int> m_ph_buffer_num;
  mutable buffer<single_ptc_t*> m_ptc_buffer_ptrs;
  mutable buffer<single_ph_t*> m_ph_buffer_ptrs;

  void setup_domain();
  void setup_devices();
  void send_array_guard_cells_single_dir(typename Conf::multi_array_t& array,
                                         const typename Conf::grid_t& grid,
                                         int dim, int dir) const;
  void send_add_array_guard_cells_single_dir(
      typename Conf::multi_array_t& array, const typename Conf::grid_t& grid,
      int dim, int dir) const;
  void send_vector_field_guard_cells_single_dir(vector_field<Conf>& field,
                                                int dim, int dir) const;
  void send_add_vector_field_guard_cells_single_dir(vector_field<Conf>& field,
                                                    int dim, int dir) const;
  template <typename PtcType>
  void send_particles_impl(PtcType& ptc, const grid_t<Conf>& grid) const;

  template <typename T>
  void send_particle_array(T& send_buffer, int& send_num, T& recv_buffer,
                           int& recv_num, int src, int dst, int tag,
                           MPI_Request* send_req, MPI_Request* recv_req,
                           MPI_Status* recv_stat) const;
  template <typename SingleType>
  void send_particle_array(std::vector<buffer<SingleType>>& buffers,
                           buffer<int>& buf_nums,
                           const std::vector<int>& buf_send_idx,
                           const std::vector<int>& buf_recv_idx, int src,
                           int dst, std::vector<MPI_Request>& req_send,
                           std::vector<MPI_Request>& req_recv,
                           std::vector<MPI_Status>& stat_recv) const;
  // std::vector<particles_t>& ptc_buffers(const particles_t& ptc) const;
  // std::vector<photons_t>& ptc_buffers(const photons_t& ptc) const;
  std::vector<buffer<single_ptc_t>>& ptc_buffers(const particles_t& ptc) const;
  std::vector<buffer<single_ph_t>>& ptc_buffers(const photons_t& ptc) const;
  buffer<single_ptc_t*>& ptc_buffer_ptrs(const particles_t& ptc) const;
  buffer<single_ph_t*>& ptc_buffer_ptrs(const photons_t& ph) const;
  buffer<int>& ptc_buffer_nums(const particles_t& ptc) const {
    return m_ptc_buffer_num;
  }
  buffer<int>& ptc_buffer_nums(const photons_t& ph) const {
    return m_ph_buffer_num;
  }
};

}  // namespace Aperture
