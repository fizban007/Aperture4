#include "catch2/catch_all.hpp"
#include "utils/nonown_ptr.hpp"

using namespace Aperture;

TEST_CASE("nonown_ptr default and nullptr construction", "[nonown_ptr]") {
  nonown_ptr<int> p;
  REQUIRE(p == nullptr);
  REQUIRE_FALSE(p != nullptr);
  REQUIRE(p.get() == nullptr);

  nonown_ptr<int> p2(nullptr);
  REQUIRE(p2 == nullptr);
}

TEST_CASE("nonown_ptr from raw pointer", "[nonown_ptr]") {
  int val = 42;
  nonown_ptr<int> p(&val);

  REQUIRE(p != nullptr);
  REQUIRE(*p == 42);
  REQUIRE(p.get() == &val);

  // Modify through pointer
  *p = 99;
  REQUIRE(val == 99);
}

TEST_CASE("nonown_ptr arrow operator", "[nonown_ptr]") {
  struct Point {
    int x, y;
  };

  Point pt{3, 7};
  nonown_ptr<Point> p(&pt);
  REQUIRE(p->x == 3);
  REQUIRE(p->y == 7);

  p->x = 10;
  REQUIRE(pt.x == 10);

  // const arrow
  const nonown_ptr<Point> cp(&pt);
  REQUIRE(cp->x == 10);
}

TEST_CASE("nonown_ptr reset and release", "[nonown_ptr]") {
  int a = 1, b = 2;
  nonown_ptr<int> p(&a);
  REQUIRE(*p == 1);

  p.reset(&b);
  REQUIRE(*p == 2);

  p.release();
  REQUIRE(p == nullptr);
}

TEST_CASE("nonown_ptr copy and move", "[nonown_ptr]") {
  int val = 55;
  nonown_ptr<int> p(&val);

  // Copy
  nonown_ptr<int> copy(p);
  REQUIRE(*copy == 55);
  REQUIRE(copy.get() == p.get());

  // Copy assignment
  nonown_ptr<int> assigned;
  assigned = p;
  REQUIRE(*assigned == 55);

  // Move
  nonown_ptr<int> moved(std::move(p));
  REQUIRE(*moved == 55);

  // Move assignment
  nonown_ptr<int> move_assigned;
  move_assigned = std::move(copy);
  REQUIRE(*move_assigned == 55);
}

TEST_CASE("nonown_ptr does not own memory", "[nonown_ptr]") {
  int* raw = new int(100);
  {
    nonown_ptr<int> p(raw);
    REQUIRE(*p == 100);
  }
  // p is destroyed but raw is still valid
  REQUIRE(*raw == 100);
  delete raw;
}
