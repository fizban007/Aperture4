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
#include "core/buffer.hpp"
#include "utils/range.hpp"

using namespace Aperture;

TEST_CASE("Host only buffer", "[buffer]") {
  uint32_t N = 1000;

  buffer<double> buf(N, MemType::host_only);

  for (int i = 0; i < buf.size(); i++) {
    buf[i] = i;
  }
  for (int i = 0; i < buf.size(); i++) {
    REQUIRE(buf[i] == Approx(i));
  }

  SECTION("Move assignment and constructor") {
    buffer<double> buf1 = std::move(buf);

    REQUIRE(buf.host_allocated() == false);
    REQUIRE(buf.host_ptr() == nullptr);
    REQUIRE(buf1.size() == N);

    buffer<double> buf2(std::move(buf1));

    REQUIRE(buf1.host_allocated() == false);
    REQUIRE(buf1.host_ptr() == nullptr);
    REQUIRE(buf2.size() == N);

    for (int i = 0; i < buf2.size(); i++) {
      REQUIRE(buf2[i] == Approx(i));
    }
  }
}

TEST_CASE("Host device buffer", "[buffer]") {
  uint32_t N = 1000;

  buffer<double> buf(N, MemType::host_device);

  REQUIRE(buf.host_allocated() == true);
#ifdef GPU_ENABLED
  REQUIRE(buf.dev_allocated() == true);
#endif

  auto ptr = buf.data();
  ptr[300] = 3.0;
  REQUIRE(ptr[300] == 3.0);

#ifdef GPU_ENABLED
  // Test coping to device and back
  buf.copy_to_device();
  ptr[300] = 6.0;
  REQUIRE(ptr[300] == 6.0);
  buf.copy_to_host();
  REQUIRE(ptr[300] == 3.0);
#endif

  SECTION("Move assignment and constructor") {
    auto buf1 = std::move(buf);

    REQUIRE(buf.host_allocated() == false);
#ifdef GPU_ENABLED
    REQUIRE(buf.dev_allocated() == false);
#endif
    REQUIRE(buf.host_ptr() == nullptr);
    REQUIRE(buf.dev_ptr() == nullptr);
    REQUIRE(buf1.size() == N);
    REQUIRE(buf1.host_allocated() == true);

    auto buf2(std::move(buf1));

    REQUIRE(buf1.host_allocated() == false);
    REQUIRE(buf1.host_ptr() == nullptr);
#ifdef GPU_ENABLED
    REQUIRE(buf1.dev_ptr() == nullptr);
#endif
    REQUIRE(buf2.size() == N);
  }
}

TEST_CASE("Managed buffer", "[buffer]") {
  uint32_t N = 1000;

  buffer<double> buf(N, MemType::device_managed);

  REQUIRE(buf.host_allocated() == false);
#ifdef GPU_ENABLED
  REQUIRE(buf.dev_allocated() == true);

  for (int i = 0; i < buf.size(); i++) {
    buf[i] = i;
  }
  for (int i = 0; i < buf.size(); i++) {
    REQUIRE(buf[i] == Approx(i));
  }
#endif
}

#ifdef GPU_ENABLED
TEST_CASE("Device only buffer", "[buffer]") {
  uint32_t N = 1000;

  buffer<double> buf(N, MemType::device_only);

  REQUIRE(buf.host_allocated() == false);
  REQUIRE(buf.dev_allocated() == true);
}
#endif

TEST_CASE("Assign and copy buffer", "[buffer]") {
  uint32_t N = 1000;

  SECTION("Host only") {
    buffer<float> buf(N, MemType::host_only);

    buf.assign(5.0f);
    for (auto i : range(0ul, buf.size())) {
      REQUIRE(buf[i] == 5.0f);
    }

    buffer<float> buf2(N, MemType::host_only);
    buf2.copy_from(buf);
    for (auto i : range(0ul, buf.size())) {
      REQUIRE(buf2[i] == 5.0f);
    }
  }

#ifdef GPU_ENABLED
  SECTION("Host and device") {
    buffer<float> buf(N, MemType::host_device);
    buf.assign(5.0f);
    buf.copy_to_host();
    for (auto i : range(0ul, buf.size())) {
      REQUIRE(buf[i] == 5.0f);
    }

    buffer<float> buf2(N, MemType::host_device);
    buf2.copy_from(buf);
    buf2.copy_to_host();
    for (auto i : range(0ul, buf.size())) {
      REQUIRE(buf2[i] == 5.0f);
    }
  }
#endif
}
