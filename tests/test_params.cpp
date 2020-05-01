#include "catch.hpp"
#include "framework/params_store.h"
// #include "visit_struct/visit_struct_intrusive.hpp"
#include "visit_struct/visit_struct.hpp"

using namespace Aperture;

TEST_CASE("Using the parameter store", "[param_store]") {
  params_store store;

  store.add("p", 4.0);
  store.add("n", 42l);
  store.add("flag", false);
  store.add("name", std::string("Alex"));

  REQUIRE(store.get_as<double>("p", 0.0) == 4.0);
  REQUIRE(store.get_as<int64_t>("n", 0) == 42);
  REQUIRE(store.get_as<bool>("flag", true) == false);
  REQUIRE(store.get_as<std::string>("name") == "Alex");
  REQUIRE(store.get_as<std::string>("p") == "");
  REQUIRE(store.get_as<std::vector<int64_t>>("something") ==
          std::vector<int64_t>{});
}

TEST_CASE("Parsing toml into our params store", "[param_store]") {
  Logger::init(0, LogLevel::debug);
  params_store store;

  store.parse("test_parsing.toml");

  REQUIRE(store.get_as<double>("dt", 0.0) == 0.01);
  REQUIRE(store.get_as<int64_t>("steps", 0) == 10000);
  REQUIRE(store.get_as<bool>("enable_cooling", false) == true);
  REQUIRE(store.get_as<std::string>("coordinate") == "Cartesian");
  REQUIRE(store.get_as<std::vector<bool>>("periodic_boundary") ==
          std::vector<bool>({true, true, true}));
  REQUIRE(store.get_as<std::vector<int64_t>>("N") ==
          std::vector<int64_t>({24, 24, 24}));
  REQUIRE(store.get_as<std::vector<double>>("size") ==
          std::vector<double>({100.0, 10.0, 2.0}));
  REQUIRE(store.get_as<std::vector<std::string>>("names") ==
          std::vector<std::string>({"Alex", "Yajie"}));
}

struct my_struct {
  float dt = 0.1f;
  uint32_t max_steps = 10000;
  bool b[4] = {};
};

VISITABLE_STRUCT(my_struct, dt, max_steps, b);

TEST_CASE("Parsing into a struct directly", "[param_store]") {
  my_struct my_param;

  REQUIRE(my_param.dt == 0.1f);
  params_store store;
  store.add("dt", 0.4);
  store.add("max_steps", 500l);
  store.add("b", std::vector<bool>{true, false, true, false});

  store.parse_struct(my_param);

  REQUIRE(my_param.dt == 0.4f);
  REQUIRE(my_param.max_steps == 500);
  REQUIRE(my_param.b[0] == true);
  REQUIRE(my_param.b[2] == true);

  // Parsing a single array
  float sizes[3];
  store.add("sizes", std::vector<double>{1.0, 2.0, 3.0, 4.0});

  store.get_array("sizes", sizes);

  REQUIRE(sizes[0] == 1.0f);
  REQUIRE(sizes[1] == 2.0f);
  REQUIRE(sizes[2] == 3.0f);

  // Parsing single data
  uint32_t N;
  store.add("N", 300l);
  store.get_value("N", N);
  REQUIRE(N == 300);
}
