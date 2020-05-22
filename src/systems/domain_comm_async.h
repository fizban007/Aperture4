#ifndef _DOMAIN_COMM_ASYNC_H_
#define _DOMAIN_COMM_ASYNC_H_

#include "domain_comm.h"
#ifdef CUDA_ENABLED
#include <cuda_runtime_api.h>
#endif

namespace Aperture {

template <typename Conf>
class domain_comm_async : public domain_comm<Conf> {
 public:
  static std::string name() { return "domain_comm"; }

  domain_comm_async(sim_environment& env);
  virtual ~domain_comm_async();

  virtual void send_guard_cells(vector_field<Conf>& field) const override;
  virtual void send_guard_cells(scalar_field<Conf>& field) const override;
  virtual void send_guard_cells(typename Conf::multi_array_t& array,
                                const Grid<Conf::dim>& grid) const override;

 protected:
  cudaStream_t m_copy_stream;

  void send_array_guard_cells_single_dir_async(
      typename Conf::multi_array_t& array, const Grid<Conf::dim>& grid, int dim,
      int dir) const;
};

}  // namespace Aperture

#endif  // _DOMAIN_COMM_ASYNC_H_
