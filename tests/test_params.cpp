#include "catch.hpp"
#include "framework/param_store.hpp"

using namespace Aperture;

TEST_CASE("Using the parameter store", "[param_store]") {
  param_store store;

  store.add("p", 4.0);
  store.add("n", 42l);
  store.add("flag", false);
  store.add("name", std::string("Alex"));

  REQUIRE(store.get<double>("p", 0.0) == 4.0);
  REQUIRE(store.get<int64_t>("n", 0) == 42);
  REQUIRE(store.get<bool>("flag", true) == false);
  REQUIRE(store.get<std::string>("name") == "Alex");
  REQUIRE(store.get<std::string>("p") == "");
}

TEST_CASE("Parsing toml into our params store", "[param_store]") {
  Logger::init(0, LogLevel::debug);
  param_store store;

  store.parse("test_parsing.toml");

  REQUIRE(store.get<double>("dt", 0.0) == 0.01);
  REQUIRE(store.get<int64_t>("steps", 0) == 10000);
  REQUIRE(store.get<bool>("enable_cooling", false) == true);
  REQUIRE(store.get<std::string>("coordinate") == "Cartesian");
  REQUIRE(store.get<std::vector<bool>>("periodic_boundary") ==
          std::vector<bool>({true, true, true}));
  REQUIRE(store.get<std::vector<int64_t>>("N") ==
          std::vector<int64_t>({24, 24, 24}));
  REQUIRE(store.get<std::vector<double>>("size") ==
          std::vector<double>({100.0, 10.0, 2.0}));
  REQUIRE(store.get<std::vector<std::string>>("names") ==
          std::vector<std::string>({"Alex", "Yajie"}));
}
