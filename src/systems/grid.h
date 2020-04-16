#ifndef __GRID_H_
#define __GRID_H_

#include "core/domain_info.h"
#include "core/grid.hpp"
#include "framework/system.h"

namespace Aperture {

template <typename Conf>
class domain_comm;

// The system that is responsible for setting up the computational grid
template <typename Conf>
class grid_t : public system_t, public Grid<Conf::dim> {
 public:
  static std::string name() { return "grid"; }

  typedef Grid<Conf::dim> base_type;

  grid_t(sim_environment& env, const domain_info_t<Conf::dim>& domain_info =
                                   domain_info_t<Conf::dim>{});
  grid_t(sim_environment& env, const domain_comm<Conf>& comm);

  inline typename Conf::idx_t get_idx(const index_t<Conf::dim>& pos) const {
    return typename Conf::idx_t(pos, this->extent());
  }

  template <typename... Args>
  inline typename Conf::idx_t get_idx(Args... args) const {
    return typename Conf::idx_t(index_t<Conf::dim>(args...), this->extent());
  }

  inline typename Conf::idx_t idx_at(uint32_t lin) const {
    return typename Conf::idx_t(lin, this->extent());
  }

};

}  // namespace Aperture

#endif  // __GRID_H_
