#ifndef __DOMAIN_COMM_H_
#define __DOMAIN_COMM_H_

#include "core/domain_info.h"
#include "core/particles.h"
#include "data/fields.h"
#include "framework/system.h"
#include <mpi.h>
#include <vector>

namespace Aperture {

template <int Dim>
struct Grid;

template <typename Conf>
class domain_comm : public system_t {
 private:
  MPI_Comm m_world;
  MPI_Comm m_cart;
  MPI_Datatype m_scalar_type;

  int m_rank = 0;  ///< Rank of current process
  int m_size = 1;  ///< Size of MPI_COMM_WORLD
  domain_info_t<Conf::dim> m_domain_info;

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

 public:
  static std::string name() { return "domain_comm"; }

  domain_comm(sim_environment& env);
  bool is_root() const { return m_rank == 0; }
  int rank() const { return m_rank; }
  int size() const { return m_size; }
  void resize_buffers(const Grid<Conf::dim>& grid);

  void send_guard_cells(vector_field<Conf>& field) const;
  void send_guard_cells(scalar_field<Conf>& field) const;
  void send_add_guard_cells(vector_field<Conf>& field) const;
  void send_add_guard_cells(scalar_field<Conf>& field) const;
  void send_particles(particles_t& ptc, const grid_t<Conf>& grid) const;
  void send_particles(photons_t& ptc, const grid_t<Conf>& grid) const;
  void get_total_num_offset(uint64_t& num, uint64_t& total,
                            uint64_t& offset) const;

  const domain_info_t<Conf::dim>& domain_info() const { return m_domain_info; }
};

}  // namespace Aperture

// #include "domain_comm.cpp"

#endif  // __DOMAIN_COMM_H_
