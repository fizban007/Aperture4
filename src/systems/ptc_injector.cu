#include "ptc_injector.h"
#include "framework/environment.hpp"
#include "framework/config.h"

namespace Aperture {

// template <typename Conf>
// void count_ptc_injected(Gu)

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

}

template class ptc_injector_cu<Config<2>>;

}
