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
  grid_t(const grid_t<Conf>& grid) = default;
  virtual ~grid_t();

  grid_t<Conf>& operator=(const grid_t<Conf>& grid) = default;

  // Coordinate for output position
  inline virtual vec_t<float, Conf::dim> cart_coord(const index_t<Conf::dim>& pos) const {
    vec_t<float, Conf::dim> result;
    for (int i = 0; i < Conf::dim; i++)
      result[i] = this->pos(i, pos[i], false);
    return result;
  }

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

  inline typename Conf::idx_t begin() const {
    return idx_at(0);
  }

  inline typename Conf::idx_t end() const {
    return idx_at(this->extent().size());
  }

  inline size_t size() const {
    return this->extent().size();
  }

};

}  // namespace Aperture

#endif  // __GRID_H_
