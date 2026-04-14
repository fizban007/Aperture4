#include "catch2/catch_all.hpp"
#include "utils/util_functions.h"
#include "core/enum_types.h"

using namespace Aperture;

TEST_CASE("sgn function", "[util_funcs]") {
  REQUIRE(sgn(5) == 1);
  REQUIRE(sgn(-3) == -1);
  REQUIRE(sgn(0) == 1);  // Note: sgn(0) returns 1 in this implementation
  REQUIRE(sgn(0.0) == 1);
  REQUIRE(sgn(-2.5) == -1);
}

TEST_CASE("clamp function", "[util_funcs]") {
  REQUIRE(clamp(5, 0, 10) == 5);
  REQUIRE(clamp(-1, 0, 10) == 0);
  REQUIRE(clamp(15, 0, 10) == 10);
  REQUIRE(clamp(0, 0, 10) == 0);
  REQUIRE(clamp(10, 0, 10) == 10);
  REQUIRE(clamp(0.5, 0.0, 1.0) == Catch::Approx(0.5));
}

TEST_CASE("symlog function", "[util_funcs]") {
  REQUIRE(symlog(0.0) == Catch::Approx(0.0));
  // symlog(x) = sgn(x) * log(1 + |x|)
  REQUIRE(symlog(1.0) == Catch::Approx(std::log(2.0)));
  REQUIRE(symlog(-1.0) == Catch::Approx(-std::log(2.0)));
  // Symmetry: symlog(-x) == -symlog(x)
  REQUIRE(symlog(5.0) == Catch::Approx(-symlog(-5.0)));
}

TEST_CASE("swap_values", "[util_funcs]") {
  int a = 3, b = 7;
  swap_values(a, b);
  REQUIRE(a == 7);
  REQUIRE(b == 3);
}

TEST_CASE("check_flag, set_flag, clear_flag, toggle_flag", "[util_funcs]") {
  uint32_t flag = 0;

  // Set tracked
  set_flag(flag, PtcFlag::tracked);
  REQUIRE(check_flag(flag, PtcFlag::tracked));
  REQUIRE_FALSE(check_flag(flag, PtcFlag::ignore_force));

  // Set multiple
  set_flag(flag, PtcFlag::ignore_force, PtcFlag::primary);
  REQUIRE(check_flag(flag, PtcFlag::tracked));
  REQUIRE(check_flag(flag, PtcFlag::ignore_force));
  REQUIRE(check_flag(flag, PtcFlag::primary));

  // Clear one
  clear_flag(flag, PtcFlag::tracked);
  REQUIRE_FALSE(check_flag(flag, PtcFlag::tracked));
  REQUIRE(check_flag(flag, PtcFlag::ignore_force));

  // Toggle
  toggle_flag(flag, PtcFlag::tracked);
  REQUIRE(check_flag(flag, PtcFlag::tracked));
  toggle_flag(flag, PtcFlag::tracked);
  REQUIRE_FALSE(check_flag(flag, PtcFlag::tracked));
}

TEST_CASE("flag_or combines bits", "[util_funcs]") {
  auto combined = flag_or(PtcFlag::tracked, PtcFlag::primary);
  REQUIRE((combined & (1 << static_cast<int>(PtcFlag::tracked))) != 0);
  REQUIRE((combined & (1 << static_cast<int>(PtcFlag::primary))) != 0);
  REQUIRE((combined & (1 << static_cast<int>(PtcFlag::secondary))) == 0);
}

TEST_CASE("get_ptc_type / gen_ptc_type_flag round-trip", "[util_funcs]") {
  // Electron
  auto flag_e = gen_ptc_type_flag(PtcType::electron);
  REQUIRE(get_ptc_type(flag_e) == static_cast<uint32_t>(PtcType::electron));

  // Positron
  auto flag_p = gen_ptc_type_flag(PtcType::positron);
  REQUIRE(get_ptc_type(flag_p) == static_cast<uint32_t>(PtcType::positron));

  // Ion
  auto flag_i = gen_ptc_type_flag(PtcType::ion);
  REQUIRE(get_ptc_type(flag_i) == static_cast<uint32_t>(PtcType::ion));
}

TEST_CASE("set_ptc_type_flag preserves lower bits", "[util_funcs]") {
  uint32_t flag = 0;
  set_flag(flag, PtcFlag::tracked, PtcFlag::primary);
  flag = set_ptc_type_flag(flag, PtcType::positron);

  // Type is positron
  REQUIRE(get_ptc_type(flag) == static_cast<uint32_t>(PtcType::positron));
  // Lower bits preserved
  REQUIRE(check_flag(flag, PtcFlag::tracked));
  REQUIRE(check_flag(flag, PtcFlag::primary));

  // Change type, lower bits still there
  flag = set_ptc_type_flag(flag, PtcType::electron);
  REQUIRE(get_ptc_type(flag) == static_cast<uint32_t>(PtcType::electron));
  REQUIRE(check_flag(flag, PtcFlag::tracked));
}

TEST_CASE("to_float, to_double, roundi", "[util_funcs]") {
  REQUIRE(to_float(7) == 7.0f);
  REQUIRE(to_double(7) == 7.0);
  REQUIRE(roundi(2.7f) == 3);
  REQUIRE(roundi(2.3f) == 2);
  REQUIRE(roundi(-0.5f) == -1);  // std::round rounds away from zero
}
