#ifndef __GRID_LOGSPH_H_
#define __GRID_LOGSPH_H_

#include "grid_curv.h"

namespace Aperture {

template <typename Conf>
class grid_logsph_t : public grid_curv_t<Conf> {
 public:
  static std::string name() { return "grid"; }
  typedef typename Conf::value_t value_t;

  grid_logsph_t(
      sim_environment &env,
      const domain_info_t<Conf::dim> &domain_info = domain_info_t<Conf::dim>{});

  grid_logsph_t(sim_environment &env, const domain_comm<Conf> &comm);
  ~grid_logsph_t();

  // Coordinate for output position
  inline vec_t<float, Conf::dim> cart_coord(
      const index_t<Conf::dim> &pos) const override {
    vec_t<float, Conf::dim> result;
    for (int i = 0; i < Conf::dim; i++) result[i] = this->pos(i, pos[i], false);
    float r = std::exp(this->pos(0, pos[0], false));
    float theta = this->pos(1, pos[1], false);
    result[0] = r * std::sin(theta);
    result[1] = r * std::cos(theta);
    return result;
  }

  void compute_coef() override;
};

template <typename FloatT>
HD_INLINE void
cart2logsph(FloatT &v1, FloatT &v2, FloatT &v3, FloatT x1, FloatT x2,
            FloatT x3) {
  FloatT v1n = v1, v2n = v2, v3n = v3;
  FloatT c2 = cos(x2), s2 = sin(x2), c3 = cos(x3), s3 = sin(x3);
  v1 = v1n * s2 * c3 + v2n * s2 * s3 + v3n * c2;
  v2 = v1n * c2 * c3 + v2n * c2 * s3 - v3n * s2;
  v3 = -v1n * s3 + v2n * c3;
}

template <typename FloatT>
HD_INLINE void
logsph2cart(FloatT &v1, FloatT &v2, FloatT &v3, FloatT x1, FloatT x2,
            FloatT x3) {
  FloatT v1n = v1, v2n = v2, v3n = v3;
  FloatT c2 = cos(x2), s2 = sin(x2), c3 = cos(x3), s3 = sin(x3);
  v1 = v1n * s2 * c3 + v2n * c2 * c3 - v3n * s3;
  v2 = v1n * s2 * s3 + v2n * c2 * s3 + v3n * c3;
  v3 = v1n * c2 - v2n * s2;
}

template <typename FloatT>
HD_INLINE FloatT
beta_phi(FloatT r, FloatT theta, FloatT compactness, FloatT omega) {
  return -0.4f * compactness * omega * std::sin(theta) / (r * r);
}

template <typename FloatT>
HD_INLINE FloatT
alpha_gr(FloatT r, FloatT compactness) {
  return std::sqrt(1.0f - compactness / r);
}

}  // namespace Aperture

#endif  // __GRID_LOGSPH_H_
