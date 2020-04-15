#include "catch.hpp"
#include "utils/interpolation.hpp"
#include "utils/logger.h"
#include "core/multi_array.hpp"

using namespace Aperture;

TEST_CASE("1D linear interpolation", "[interp]") {
  auto v = make_multi_array<double, MemoryModel::host_only>(4);
  auto interp = interpolator<bspline<1>, 1>{};

  v.assign(1.0);
  auto idx = v.get_idx(1);
  auto pos = idx.get_pos();
  REQUIRE(interp(v, vec_t<float, 3>(0.1, 0.1, 0.1),
                 idx, pos) == Approx(1.0));
}

TEST_CASE("2D cubic interpolation", "[interp]") {
  auto v = make_multi_array<double, MemoryModel::host_only>(4, 4);
  auto interp = interpolator<bspline<3>, 2>{};

  v.assign(1.0);
  auto idx = v.get_idx(1, 1);
  double a = interp(v, vec_t<float, 3>(0.3, 0.4, 0.5),
                    idx, index(1, 1));
  REQUIRE(a == Approx(1.0));

  Logger::print_info("a is {}", a);

  v.emplace(0, {1.0, 2.0, 3.0, 4.0,
                2.0, 3.0, 4.0, 5.0,
                3.0, 4.0, 5.0, 6.0,
                4.0, 5.0, 6.0, 7.0});
  a = interp(v, vec_t<double, 3>(0.0, 1.0, 0.5),
             idx, index(1, 1));
  REQUIRE(a == Approx(4.0));
}
