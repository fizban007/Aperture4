#ifndef __GRID_H_
#define __GRID_H_

#include "core/grid.hpp"
#include "framework/system.h"

namespace Aperture {

// The system that is responsible for setting up the computational grid
template <typename Conf>
class grid_t : public system_t, public Grid<Conf::dim> {
 public:
  static std::string name() { return "grid"; }

  typedef Grid<Conf::dim> base_type;

  grid_t(const Conf& conf) {}

  void init();
  void register_dependencies(sim_environment& env);

};

}  // namespace Aperture

#endif  // __GRID_H_
