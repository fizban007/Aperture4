#ifndef _GRID_SPH_H_
#define _GRID_SPH_H_

#include "grid_curv.h"

namespace Aperture {

template <typename Conf>
class grid_sph_t : public grid_curv_t<Conf> {
 public:
  static std::string name() { return "grid"; }
  typedef typename Conf::value_t value_t;

  grid_sph_t(
      sim_environment &env,
      const domain_info_t<Conf::dim> &domain_info = domain_info_t<Conf::dim>{});

  grid_sph_t(sim_environment &env, const domain_comm<Conf> &comm);
  ~grid_sph_t();

  // Coordinate for output position
  inline vec_t<float, Conf::dim> cart_coord(
      const index_t<Conf::dim> &pos) const override {
    vec_t<float, Conf::dim> result;
    for (int i = 0; i < Conf::dim; i++) result[i] = this->pos(i, pos[i], false);
    float r = this->pos(0, pos[0], false);
    float theta = this->pos(1, pos[1], false);
    result[0] = r * std::sin(theta);
    result[1] = r * std::cos(theta);
    return result;
  }

  void compute_coef() override;
};

}  // namespace Aperture


#endif  // _GRID_SPH_H_
