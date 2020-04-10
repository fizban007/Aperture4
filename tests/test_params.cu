#include "catch.hpp"
#include "framework/param_store.hpp"

using namespace Aperture;

TEST_CASE("Using the parameter store in cu", "[param_store]") {
  param_store store;

  store.add("p", 4.0);
  store.add("n", 42l);

  int n = store.get<int64_t>("n", 1);
  REQUIRE(n == 42);
  // REQUIRE(store.get<int>("n", 1) == 4);
  // REQUIRE(store.get<std::string>("name", "") == "");
  // REQUIRE(store.get<std::string>("p", "") == "");
}
