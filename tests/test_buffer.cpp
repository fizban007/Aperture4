#include "catch.hpp"
#include "core/buffer.h"

using namespace Aperture;

TEST_CASE("Host only buffer", "[buffer]") {
  uint32_t N = 1000;

  buffer_t<double> buf(N);

  for (int i = 0; i < buf.size(); i++) {
    buf[i] = i;
  }
  for (int i = 0; i < buf.size(); i++) {
    REQUIRE(buf[i] == Approx(i));
  }
}

TEST_CASE("Host device buffer", "[buffer]") {
  uint32_t N = 1000;

  buffer_t<double, MemoryModel::host_device> buf(N);

  REQUIRE(buf.host_allocated() == true);
#ifdef CUDA_ENABLED
  REQUIRE(buf.dev_allocated() == true);
#endif

  auto ptr = buf.data();
  ptr[300] = 3.0;
  REQUIRE(ptr[300] == 3.0);

#ifdef CUDA_ENABLED
  // Test coping to device and back
  buf.copy_to_device();
  ptr[300] = 6.0;
  REQUIRE(ptr[300] == 6.0);
  buf.copy_to_host();
  REQUIRE(ptr[300] == 3.0);
#endif
}

TEST_CASE("Managed buffer", "[buffer]") {
  uint32_t N = 1000;

  buffer_t<double, MemoryModel::device_managed> buf(N);

  REQUIRE(buf.host_allocated() == false);
#ifdef CUDA_ENABLED
  REQUIRE(buf.dev_allocated() == true);

  for (int i = 0; i < buf.size(); i++) {
    buf[i] = i;
  }
  for (int i = 0; i < buf.size(); i++) {
    REQUIRE(buf[i] == Approx(i));
  }
#endif
}

#ifdef CUDA_ENABLED
TEST_CASE("Device only buffer", "[buffer]") {
  uint32_t N = 1000;

  buffer_t<double, MemoryModel::device_only> buf(N);

  REQUIRE(buf.host_allocated() == false);
  REQUIRE(buf.dev_allocated() == true);
}
#endif
