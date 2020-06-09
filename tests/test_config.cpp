#include "catch.hpp"
#include "framework/config.h"

using namespace Aperture;

TEST_CASE("Making multi_array", "[config]") {
  // Since Config is a purely compile-time configurator, this test file is meant
  // to contain no output. The test passes as long as this can compile.
  typedef Config<3> Conf;
  auto array = Conf::make_multi_array({32, 32, 32});
}
