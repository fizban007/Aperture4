#include "catch.hpp"
#include "utils/singleton_holder.h"
#include "systems/policies/exec_policy_host.hpp"

using namespace Aperture;

struct Atest {
  int n = 3;
};

TEST_CASE("Using Singleton", "[singleton]") {
  REQUIRE(singleton_holder<Atest>::instance().n == 3);
}

