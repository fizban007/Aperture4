#include "ptc_updater.h"
#include "framework/config.h"
#include "core/constant_mem_func.h"

namespace Aperture {

template <typename Conf>
void
ptc_updater<Conf>::init() {
  init_charge_mass();
  init_dev_charge_mass(m_charges, m_masses);
}

template <typename Conf>
void
ptc_updater<Conf>::update(double dt, uint32_t step) {

}


template class ptc_updater<Config<1>>;
template class ptc_updater<Config<2>>;
template class ptc_updater<Config<3>>;


}
