#include "catch2/catch_all.hpp"
#include "utils/stagger.h"

using namespace Aperture;

TEST_CASE("stagger_t default construction", "[stagger]") {
  stagger_t st;
  REQUIRE(st[0] == 0);
  REQUIRE(st[1] == 0);
  REQUIRE(st[2] == 0);
}

TEST_CASE("stagger_t construction from bitmask", "[stagger]") {
  stagger_t st(0b101);
  REQUIRE(st[0] == 1);
  REQUIRE(st[1] == 0);
  REQUIRE(st[2] == 1);

  stagger_t st2(0b110);
  REQUIRE(st2[0] == 0);
  REQUIRE(st2[1] == 1);
  REQUIRE(st2[2] == 1);
}

TEST_CASE("stagger_t set_bit and read round-trip", "[stagger]") {
  stagger_t st;
  st.set_bit(1, true);
  REQUIRE(st[0] == 0);
  REQUIRE(st[1] == 1);
  REQUIRE(st[2] == 0);

  st.set_bit(2, true);
  REQUIRE(st[2] == 1);

  st.set_bit(1, false);
  REQUIRE(st[1] == 0);
  REQUIRE(st[2] == 1);
}

TEST_CASE("stagger_t flip", "[stagger]") {
  stagger_t st(0b010);
  REQUIRE(st[1] == 1);

  st.flip(1);
  REQUIRE(st[1] == 0);

  st.flip(1);
  REQUIRE(st[1] == 1);

  st.flip(0);
  REQUIRE(st[0] == 1);
  REQUIRE(st[1] == 1);
}

TEST_CASE("stagger_t complement", "[stagger]") {
  stagger_t st(0b101);
  auto comp = st.complement();

  // Complement inverts the low 3 bits
  REQUIRE(comp[0] == 0);
  REQUIRE(comp[1] == 1);
  REQUIRE(comp[2] == 0);
}

TEST_CASE("stagger_t copy and assignment", "[stagger]") {
  stagger_t st(0b110);
  stagger_t copy(st);
  REQUIRE(copy[0] == st[0]);
  REQUIRE(copy[1] == st[1]);
  REQUIRE(copy[2] == st[2]);

  stagger_t assigned;
  assigned = st;
  REQUIRE(assigned[1] == 1);

  // Assign from unsigned char
  assigned = 0b001;
  REQUIRE(assigned[0] == 1);
  REQUIRE(assigned[1] == 0);
}
