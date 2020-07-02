#ifndef _GRID_SPH_H_
#define _GRID_SPH_H_

#include "core/math.hpp"
#include "grid_curv.h"

namespace Aperture {

////////////////////////////////////////////////////////////////////////////////
///  This is the general spherical grid class. The class implements two crucial
///  functions: radius and theta, and provides a way to use these to compute
///  area and length elements.
////////////////////////////////////////////////////////////////////////////////
template <typename Conf>
class grid_sph_t : public grid_curv_t<Conf> {
 public:
  static std::string name() { return "grid"; }
  typedef typename Conf::value_t value_t;

  using grid_curv_t<Conf>::grid_curv_t;
  ~grid_sph_t();

  // static HD_INLINE value_t radius(value_t x1) { return x1; }
  static HD_INLINE value_t radius(value_t x1) { return math::exp(x1); }
  static HD_INLINE value_t theta(value_t x2) { return x2; }
  // static HD_INLINE value_t from_radius(value_t r) { return r; }
  static HD_INLINE value_t from_radius(value_t r) { return math::log(r); }
  static HD_INLINE value_t from_theta(value_t theta) { return theta; }

  // Coordinate for output position
  inline vec_t<float, Conf::dim> cart_coord(
      const index_t<Conf::dim> &pos) const override {
    vec_t<float, Conf::dim> result;
    for (int i = 0; i < Conf::dim; i++) result[i] = this->pos(i, pos[i], false);
    float r = radius(this->pos(0, pos[0], false));
    float th = theta(this->pos(1, pos[1], false));
    result[0] = r * math::sin(th);
    result[1] = r * math::cos(th);
    return result;
  }

  void compute_coef() override;
};

template <typename FloatT>
HD_INLINE void
cart2sph(FloatT &v1, FloatT &v2, FloatT &v3, FloatT x1, FloatT x2, FloatT x3) {
  FloatT v1n = v1, v2n = v2, v3n = v3;
  FloatT c2 = math::cos(x2), s2 = math::sin(x2), c3 = math::cos(x3),
         s3 = math::sin(x3);
  v1 = v1n * s2 * c3 + v2n * s2 * s3 + v3n * c2;
  v2 = v1n * c2 * c3 + v2n * c2 * s3 - v3n * s2;
  v3 = -v1n * s3 + v2n * c3;
}

template <typename FloatT>
HD_INLINE void
sph2cart(FloatT &v1, FloatT &v2, FloatT &v3, FloatT x1, FloatT x2, FloatT x3) {
  FloatT v1n = v1, v2n = v2, v3n = v3;
  FloatT c2 = math::cos(x2), s2 = math::sin(x2), c3 = math::cos(x3),
         s3 = math::sin(x3);
  v1 = v1n * s2 * c3 + v2n * c2 * c3 - v3n * s3;
  v2 = v1n * s2 * s3 + v2n * c2 * s3 + v3n * c3;
  v3 = v1n * c2 - v2n * s2;
}

template <typename FloatT>
HD_INLINE FloatT
beta_phi(FloatT r, FloatT theta, FloatT compactness, FloatT omega) {
  return -0.4f * compactness * omega * math::sin(theta) / (r * r);
}

template <typename FloatT>
HD_INLINE FloatT
alpha_gr(FloatT r, FloatT compactness) {
  return math::sqrt(1.0f - compactness / r);
}

}  // namespace Aperture

#endif  // _GRID_SPH_H_
