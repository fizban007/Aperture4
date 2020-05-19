#include "catch.hpp"
#include "core/math.hpp"
#include "utils/buffer.h"
#include "utils/kernel_helper.hpp"

using namespace Aperture;

TEST_CASE("Checking double precision math functions", "[math]") {
  buffer<double> x(2, MemType::device_managed);
  x[0] = 3.0;
  kernel_launch({1, 1}, [] __device__(double* x) {
      x[1] = math::sin(x[0]);
    }, x.dev_ptr());
  CudaSafeCall(cudaDeviceSynchronize());

  REQUIRE(x[1] == std::sin(x[0]));
  REQUIRE(math::abs(x[0]) == std::abs(x[0]));
  REQUIRE(math::sqrt(x[0]) == std::sqrt(x[0]));
  REQUIRE(math::cos(3.0f) == std::cos(3.0f));
}
