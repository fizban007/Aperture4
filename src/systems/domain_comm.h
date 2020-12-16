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

#ifndef __DOMAIN_COMM_H_
#define __DOMAIN_COMM_H_

#include "core/buffer.hpp"
#include "core/domain_info.h"
#include "core/particles.h"
#include "data/fields.h"
#include "framework/system.h"
#include <mpi.h>
#include <vector>

namespace Aperture {

template <int Dim, typename value_t>
struct Grid;

template <typename Conf>
class domain_comm : public system_t {
 public:
  static std::string name() { return "domain_comm"; }

  domain_comm(sim_environment& env);
  virtual ~domain_comm();

  bool is_root() const { return m_rank == 0; }
  int rank() const { return m_rank; }
  int size() const { return m_size; }
  void resize_buffers(const typename Conf::grid_t& grid) const;

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
  void get_total_num_offset(uint64_t& num, uint64_t& total,
                            uint64_t& offset) const;

  template <typename T>
  void gather_to_root(buffer<T>& buf) const;

  const domain_info_t<Conf::dim>& domain_info() const { return m_domain_info; }

 protected:
  MPI_Comm m_world;
  MPI_Comm m_cart;
  MPI_Datatype m_scalar_type;

  int m_rank = 0;  ///< Rank of current process
  int m_size = 1;  ///< Size of MPI_COMM_WORLD
  domain_info_t<Conf::dim> m_domain_info;
  mutable bool m_buffers_ready = false;

  // Communication buffers. These buffers are declared mutable because we want
  // to use a const domain_comm reference to invoke communications, but
  // communication will necessarily need to modify these buffers.
  typedef typename Conf::multi_array_t multi_array_t;

  mutable std::vector<multi_array_t> m_send_buffers;
  mutable std::vector<multi_array_t> m_recv_buffers;
  mutable std::vector<particles_t> m_ptc_buffers;
  mutable std::vector<photons_t> m_ph_buffers;
  mutable buffer<ptc_ptrs> m_ptc_buffer_ptrs;
  mutable buffer<ph_ptrs> m_ph_buffer_ptrs;

  void setup_domain();
  void send_array_guard_cells_single_dir(
      typename Conf::multi_array_t& array, const typename Conf::grid_t& grid, int dim,
      int dir) const;
  void send_add_array_guard_cells_single_dir(
      typename Conf::multi_array_t& array, const typename Conf::grid_t& grid, int dim,
      int dir) const;
  template <typename PtcType>
  void send_particles_impl(PtcType& ptc, const grid_t<Conf>& grid) const;

  template <typename T>
  void send_particle_array(T& send_buffer, T& recv_buffer, int src, int dst,
                           int tag, MPI_Request* send_req,
                           MPI_Request* recv_req, MPI_Status* recv_stat) const;
  std::vector<particles_t>& ptc_buffers(const particles_t& ptc) const;
  std::vector<photons_t>& ptc_buffers(const photons_t& ptc) const;
  buffer<ptc_ptrs>& ptc_buffer_ptrs(const particles_t& ptc) const;
  buffer<ph_ptrs>& ptc_buffer_ptrs(const photons_t& ph) const;
};

}  // namespace Aperture

// #include "domain_comm.cpp"

#endif  // __DOMAIN_COMM_H_
