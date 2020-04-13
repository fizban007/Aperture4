#include "fields.hpp"
#include "framework/config.h"
#include "framework/environment.hpp"
#include "systems/grid.hpp"
#include <exception>

namespace Aperture {

template <int N, typename Conf>
void
field_t<N, Conf>::init(const std::string& name,
                       const sim_environment& env) {
  auto grid = env.shared_data().get<const Grid<Conf::dim>>("grid");
  if (grid == nullptr)
    throw std::runtime_error("No grid system defined!");

  init(grid->extent());

  // Find the initial condition for this data component. If found, then use it
  // to initialize this class with the given values
  env.event_handler().invoke_callback("initial " + name, *this, *grid);
}

template <int N, typename Conf>
void
field_t<N, Conf>::init(const extent_t<Conf::dim>& ext) {
  for (int i = 0; i < N; i++) {
    m_data[i].resize(ext);
    m_data[i].assign(0.0);
  }
}

///////////////////////////////////////////////////////
// Explicitly instantiate some fields
template class field_t<3, Config<1, float>>;
template class field_t<3, Config<2, float>>;
template class field_t<3, Config<3, float>>;
template class field_t<3, Config<1, double>>;
template class field_t<3, Config<2, double>>;
template class field_t<3, Config<3, double>>;

template class field_t<1, Config<1, float>>;
template class field_t<1, Config<2, float>>;
template class field_t<1, Config<3, float>>;
template class field_t<1, Config<1, double>>;
template class field_t<1, Config<2, double>>;
template class field_t<1, Config<3, double>>;

}  // namespace Aperture
