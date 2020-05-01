#include "ptc_injector.h"
#include "framework/environment.hpp"
#include "framework/config.h"

namespace Aperture {

template <typename Conf>
void
ptc_injector<Conf>::init() {}

template <typename Conf>
void
ptc_injector<Conf>::update(double dt, uint32_t step) {}

template <typename Conf>
void
ptc_injector<Conf>::register_dependencies() {
  m_env.get_data("particles", &ptc);
  m_env.get_data("B", &B);
}

template class ptc_injector<Config<2>>;

}
