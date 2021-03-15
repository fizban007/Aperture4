/*
 * Copyright (c) 2020 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "catch.hpp"
#include "core/particles.h"
#include "utils/util_functions.h"

using namespace Aperture;

#ifdef CUDA_ENABLED
constexpr MemType mem_type = MemType::host_device;
#else
constexpr MemType mem_type = MemType::host_only;
#endif

TEST_CASE("Initializing particles", "[particles]") {
  size_t N = 10000;
  particles_t ptc(N, mem_type);
  photons_t ph(N, mem_type);

  REQUIRE(ptc.mem_type() == mem_type);
  REQUIRE(ph.mem_type() == mem_type);
  REQUIRE(ptc.size() == N);
  REQUIRE(ph.size() == N);
  REQUIRE(ptc.x1.size() == N);
  REQUIRE(ptc.flag.size() == N);
  REQUIRE(ptc.number() == 0);
  REQUIRE(ph.number() == 0);

  SECTION("Move assignment and constructor") {
    ptc.x1[10] = 0.1f;

    particles_t ptc1 = std::move(ptc);
    REQUIRE(ptc1.mem_type() == mem_type);
    REQUIRE(ptc1.size() == N);
    REQUIRE(ptc1.x1[10] == 0.1f);
    REQUIRE(ptc.x1.host_allocated() == false);
  }
}

TEST_CASE("Particle flag manipulation", "[particles]") {
  uint32_t flag = set_ptc_type_flag(
      flag_or(PtcFlag::primary, PtcFlag::tracked),
      PtcType::electron);

  REQUIRE(get_ptc_type(flag) == (int)PtcType::electron);
  REQUIRE(check_flag(flag, PtcFlag::primary) == true);
  REQUIRE(check_flag(flag, PtcFlag::tracked) == true);
  REQUIRE(check_flag(flag, PtcFlag::ignore_EM) == false);
  REQUIRE(check_flag(flag, PtcFlag::ignore_current) == false);
  // REQUIRE(check_flag(flag, PtcFlag::emit_photon) == false);
}

TEST_CASE("Init, copy and assign particles", "[particles]") {
  size_t N = 100;
  particles_t ptc(N, mem_type);
  particles_t ptc2(N, mem_type);
  ptc.init();
  ptc.copy_to_host();
  for (int i = 0; i < N; i++)
    REQUIRE(ptc.cell[i] == empty_cell);

  ptc2.cell.assign(10);
  ptc2.copy_to_host();
  for (int i = 0; i < N; i++)
    REQUIRE(ptc2.cell[i] == 10);

  ptc2.copy_from(ptc, N, 0, 0);
  ptc2.copy_to_host();
  for (int i = 0; i < N; i++)
    REQUIRE(ptc2.cell[i] == empty_cell);
}

#ifdef CUDA_ENABLED
TEST_CASE("Particle pointers", "[particles]") {
  particles_t ptc(100, MemType::host_device);
  auto ptrs = ptc.get_dev_ptrs();
  REQUIRE(ptrs.x1 == ptc.x1.dev_ptr());
  REQUIRE(ptrs.x2 == ptc.x2.dev_ptr());
  REQUIRE(ptrs.x3 == ptc.x3.dev_ptr());
  REQUIRE(ptrs.cell == ptc.cell.dev_ptr());
  REQUIRE(ptrs.flag == ptc.flag.dev_ptr());
}
#endif

TEST_CASE("Sorting particles by cell", "[particles]") {
  size_t N = 100;

  particles_t ptc(N, mem_type);
  ptc.set_num(3);
  ptc.x1.emplace(0, {0.1, 0.2, 0.3});
  ptc.cell.emplace(0, {34, 24, 14});

#ifdef CUDA_ENABLED
  ptc.copy_to_device();
  ptc.sort_by_cell_dev(100);
  ptc.copy_to_host();
#else
  ptc.sort_by_cell_host(100);
#endif
  REQUIRE(ptc.x1[0] == Approx(0.3f));
  REQUIRE(ptc.x1[1] == Approx(0.2f));
  REQUIRE(ptc.x1[2] == Approx(0.1f));
  REQUIRE(ptc.cell[0] == 14);
  REQUIRE(ptc.cell[1] == 24);
  REQUIRE(ptc.cell[2] == 34);
}
