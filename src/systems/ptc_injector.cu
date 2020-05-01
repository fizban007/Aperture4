#include "ptc_injector.h"
#include "framework/environment.h"
#include "framework/config.h"
#include "utils/kernel_helper.hpp"

namespace Aperture {

template <typename Conf>
void count_ptc_injected(buffer<int>& num_per_cell,
                        buffer<int>& cum_num_per_cell,
                        const vector_field<Conf>& B) {
  auto ext = B.grid().extent();


}

template <typename Conf>
void
ptc_injector_cu<Conf>::init() {}

template <typename Conf>
void
ptc_injector_cu<Conf>::register_dependencies() {
  ptc_injector<Conf>::register_dependencies();

  this->m_env.get_data("rand_states", &m_rand_states);

  m_num_per_cell.set_memtype(MemType::host_device);
  m_cum_num_per_cell.set_memtype(MemType::host_device);
  // m_pos_in_array.set_memtype(MemType::host_device);

  m_num_per_cell.resize(this->m_grid.extent().size());
  m_cum_num_per_cell.resize(this->m_grid.extent().size());
  // m_posInBlock.resize()
}

template <typename Conf>
void
ptc_injector_cu<Conf>::update(double dt, uint32_t step) {
  count_ptc_injected<Conf>(m_num_per_cell, m_cum_num_per_cell, *(this->B));
}

template class ptc_injector_cu<Config<2>>;

}
