#include "catch.hpp"
#include "components/particles.h"
#include "utils/util_functions.h"

using namespace Aperture;

TEST_CASE("Initializing particles", "[particles]") {
  size_t N = 10000;
  particles_t<MemoryModel::host_device> ptc(N);
  photons_t<MemoryModel::host_device> ph(N);

  REQUIRE(ptc.size() == N);
  REQUIRE(ph.size() == N);
  REQUIRE(ptc.x1.size() == N);
  REQUIRE(ptc.flag.size() == N);
  REQUIRE(ptc.number() == 0);
  REQUIRE(ph.number() == 0);

  SECTION("Move assignment and constructor") {
    ptc.x1[10] = 0.1f;

    particles_t<MemoryModel::host_device> ptc1 = std::move(ptc);
    REQUIRE(ptc1.size() == N);
    REQUIRE(ptc1.x1[10] == 0.1f);
    REQUIRE(ptc.x1.host_allocated() == false);
  }
}

TEST_CASE("Particle flag manipulation", "[particles]") {
  uint32_t flag = set_ptc_type_flag(
      bit_or(PtcFlag::primary, PtcFlag::tracked),
      PtcType::electron);

  REQUIRE(get_ptc_type(flag) == (int)PtcType::electron);
  REQUIRE(check_flag(flag, PtcFlag::primary) == true);
  REQUIRE(check_flag(flag, PtcFlag::tracked) == true);
  REQUIRE(check_flag(flag, PtcFlag::ignore_EM) == false);
  REQUIRE(check_flag(flag, PtcFlag::ignore_current) == false);
  REQUIRE(check_flag(flag, PtcFlag::emit_photon) == false);
}
