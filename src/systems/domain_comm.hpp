#ifndef __DOMAIN_COMM_H_
#define __DOMAIN_COMM_H_

#include "data/fields.hpp"
#include "framework/system.h"
#include "core/domain_info.h"
#include "core/particles.h"
#include <vector>
#include <mpi.h>

// #define NEIGHBOR_NULL

namespace Aperture {

template <typename Conf>
class domain_comm : public system_t {
 private:
  MPI_Comm m_world;
  MPI_Comm m_cart;
  MPI_Datatype m_scalar_type;

  int m_rank = 0;  ///< Rank of current process
  int m_size = 1;  ///< Size of MPI_COMM_WORLD
  domain_info_t<Conf::dim> m_domain_info;

  // Communication buffers
  std::vector<typename Conf::multi_array_t> m_send_buffers;
  std::vector<typename Conf::multi_array_t> m_recv_buffers;
  std::vector<typename Conf::ptc_t> m_ptc_buffers;
  std::vector<typename Conf::ph_t> m_ph_buffers;

 public:
  static std::string name() { return "communicator"; }

  domain_comm(sim_environment& env);

  void setup_domain();
  void send_guard_cell(vector_field<Conf>& field);
  void send_guard_cell(scalar_field<Conf>& field);
  void get_total_num_offset(uint64_t& num, uint64_t& total,
                            uint64_t& offset);

  const domain_info_t<Conf::dim>& domain_info() {
    return m_domain_info;
  }
};

}  // namespace Aperture

// #include "domain_comm.cpp"

#endif  // __DOMAIN_COMM_H_
