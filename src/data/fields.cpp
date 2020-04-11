#include "fields.hpp"
#include "systems/grid.hpp"
#include "framework/environment.hpp"
#include "framework/config.h"
#include <exception>

namespace Aperture {

template <typename Conf>
void
vector_field<Conf>::init(const sim_environment& env) {
  auto grid = std::dynamic_pointer_cast<const grid_t<Conf>>(
      env.get_system("grid"));
  if (grid == nullptr)
    throw std::runtime_error("No grid system defined!");

  auto ext = grid->extent();
  v1.resize(ext);
  v2.resize(ext);
  v3.resize(ext);
}

template <typename Conf>
void
scalar_field<Conf>::init(const sim_environment& env) {
  auto grid = std::dynamic_pointer_cast<const grid_t<Conf>>(
      env.get_system("grid"));
  if (grid == nullptr)
    throw std::runtime_error("No grid system defined!");

  auto ext = grid->extent();
  v.resize(ext);
}

///////////////////////////////////////////////////////
// Explicitly instantiate some fields
template class vector_field<Config<1, float>>;
template class vector_field<Config<2, float>>;
template class vector_field<Config<3, float>>;
template class vector_field<Config<1, double>>;
template class vector_field<Config<2, double>>;
template class vector_field<Config<3, double>>;

template class scalar_field<Config<1, float>>;
template class scalar_field<Config<2, float>>;
template class scalar_field<Config<3, float>>;
template class scalar_field<Config<1, double>>;
template class scalar_field<Config<2, double>>;
template class scalar_field<Config<3, double>>;

}
