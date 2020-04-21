#ifndef __GRID_LOGSPH_H_
#define __GRID_LOGSPH_H_

#include "core/multi_array.hpp"
#include "grid.h"

namespace Aperture {

template <typename ValueT, int Rank, typename Idx_t>
struct grid_logsph_ptrs {
  vec_t<ndptr_const<ValueT, Rank, Idx_t>, 3> le;
  vec_t<ndptr_const<ValueT, Rank, Idx_t>, 3> lb;
  vec_t<ndptr_const<ValueT, Rank, Idx_t>, 3> Ae;
  vec_t<ndptr_const<ValueT, Rank, Idx_t>, 3> Ab;
  ndptr_const<ValueT, Rank, Idx_t> dV;
};

template <typename Conf>
class grid_logsph_t : public grid_t<Conf> {
 public:
  static std::string name() { return "grid"; }
  typedef typename Conf::value_t value_t;
  typedef grid_logsph_ptrs<value_t, Conf::dim, typename Conf::idx_t>
      grid_ptrs_t;

  grid_logsph_t(
      sim_environment& env,
      const domain_info_t<Conf::dim>& domain_info = domain_info_t<Conf::dim>{});

  grid_logsph_t(sim_environment& env, const domain_comm<Conf>& comm);
  ~grid_logsph_t();

  // Coordinate for output position
  inline vec_t<float, Conf::dim> cart_coord(
      const index_t<Conf::dim>& pos) const override {
    vec_t<float, Conf::dim> result;
    for (int i = 0; i < Conf::dim; i++) result[i] = this->pos(i, pos[i], false);
    float r = std::exp(this->pos(0, pos[0], false));
    float theta = this->pos(1, pos[1], false);
    result[0] = r * std::sin(theta);
    result[1] = r * std::cos(theta);
    return result;
  }

  void compute_coef();
  grid_ptrs_t get_grid_ptrs() const;

  std::array<multi_array<value_t, Conf::dim>, 3> m_le;
  std::array<multi_array<value_t, Conf::dim>, 3> m_lb;
  std::array<multi_array<value_t, Conf::dim>, 3> m_Ae;
  std::array<multi_array<value_t, Conf::dim>, 3> m_Ab;
  multi_array<value_t, Conf::dim> m_dV;
};

}  // namespace Aperture

#endif  // __GRID_LOGSPH_H_
