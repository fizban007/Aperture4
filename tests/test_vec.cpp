#include "catch2/catch_all.hpp"
#include "utils/vec.hpp"

using namespace Aperture;

TEST_CASE("vec_t construction and element access", "[vec]") {
  vec_t<double, 3> v(1.0, 2.0, 3.0);
  REQUIRE(v[0] == 1.0);
  REQUIRE(v[1] == 2.0);
  REQUIRE(v[2] == 3.0);
  REQUIRE(v.at(1) == 2.0);

  // Default construction zeroes elements
  vec_t<int, 4> z;
  REQUIRE(z[0] == 0);
  REQUIRE(z[3] == 0);

  // Construct from C array
  float arr[2] = {5.0f, 6.0f};
  vec_t<float, 2> from_arr(arr);
  REQUIRE(from_arr[0] == 5.0f);
  REQUIRE(from_arr[1] == 6.0f);

  // Factory function
  auto w = vec<double>(4.0, 5.0, 6.0);
  REQUIRE(w[0] == 4.0);
  REQUIRE(w[2] == 6.0);
}

TEST_CASE("vec_t assignment", "[vec]") {
  vec_t<int, 3> v(1, 2, 3);

  // Assign scalar fills all elements
  v = 7;
  REQUIRE(v[0] == 7);
  REQUIRE(v[1] == 7);
  REQUIRE(v[2] == 7);

  // set() does the same
  v.set(42);
  REQUIRE(v[0] == 42);
  REQUIRE(v[2] == 42);

  // Assign from C array
  int arr[3] = {10, 20, 30};
  v = arr;
  REQUIRE(v[0] == 10);
  REQUIRE(v[1] == 20);
  REQUIRE(v[2] == 30);
}

TEST_CASE("vec_t arithmetic", "[vec]") {
  vec_t<double, 3> a(1.0, 2.0, 3.0);
  vec_t<double, 3> b(4.0, 5.0, 6.0);

  SECTION("addition") {
    auto c = a + b;
    REQUIRE(c[0] == 5.0);
    REQUIRE(c[1] == 7.0);
    REQUIRE(c[2] == 9.0);
  }

  SECTION("subtraction") {
    auto c = b - a;
    REQUIRE(c[0] == 3.0);
    REQUIRE(c[1] == 3.0);
    REQUIRE(c[2] == 3.0);
  }

  SECTION("element-wise multiplication") {
    auto c = a * b;
    REQUIRE(c[0] == 4.0);
    REQUIRE(c[1] == 10.0);
    REQUIRE(c[2] == 18.0);
  }

  SECTION("element-wise division") {
    auto c = b / a;
    REQUIRE(c[0] == 4.0);
    REQUIRE(c[1] == 2.5);
    REQUIRE(c[2] == 2.0);
  }

  SECTION("scalar multiplication") {
    auto c = a * 3.0;
    REQUIRE(c[0] == 3.0);
    REQUIRE(c[1] == 6.0);
    REQUIRE(c[2] == 9.0);

    // Left multiply
    auto d = 2.0 * a;
    REQUIRE(d[0] == 2.0);
    REQUIRE(d[1] == 4.0);
  }

  SECTION("scalar division") {
    auto c = b / 2.0;
    REQUIRE(c[0] == 2.0);
    REQUIRE(c[1] == 2.5);
    REQUIRE(c[2] == 3.0);
  }

  SECTION("in-place operators") {
    auto c = a;
    c += b;
    REQUIRE(c[0] == 5.0);

    c = a;
    c -= b;
    REQUIRE(c[0] == -3.0);

    c = a;
    c *= 2.0;
    REQUIRE(c[1] == 4.0);
  }
}

TEST_CASE("vec_t dot, cross, norm, product", "[vec]") {
  vec_t<double, 3> a(1.0, 2.0, 3.0);
  vec_t<double, 3> b(4.0, 5.0, 6.0);

  REQUIRE(a.dot(b) == Catch::Approx(32.0));  // 1*4 + 2*5 + 3*6
  REQUIRE(a.norm() == Catch::Approx(std::sqrt(14.0)));
  REQUIRE(a.product() == 6.0);  // 1 * 2 * 3

  // Cross product
  vec_t<double, 3> x(1.0, 0.0, 0.0);
  vec_t<double, 3> y(0.0, 1.0, 0.0);
  auto z = cross(x, y);
  REQUIRE(z[0] == Catch::Approx(0.0));
  REQUIRE(z[1] == Catch::Approx(0.0));
  REQUIRE(z[2] == Catch::Approx(1.0));

  // Cross product anti-commutativity
  auto z2 = cross(y, x);
  REQUIRE(z2[2] == Catch::Approx(-1.0));

  // a x a == 0
  auto self_cross = cross(a, a);
  REQUIRE(self_cross[0] == Catch::Approx(0.0));
  REQUIRE(self_cross[1] == Catch::Approx(0.0));
  REQUIRE(self_cross[2] == Catch::Approx(0.0));
}

TEST_CASE("vec_t comparison", "[vec]") {
  vec_t<int, 3> a(1, 2, 3);
  vec_t<int, 3> b(1, 2, 3);
  vec_t<int, 3> c(4, 5, 6);

  REQUIRE(a == b);
  REQUIRE(a != c);
  REQUIRE(a < c);
  REQUIRE(a <= b);
  REQUIRE(a <= c);
}

TEST_CASE("vec_t subset extraction", "[vec]") {
  vec_t<int, 4> v(10, 20, 30, 40);

  auto sub = v.subset<1, 3>();
  REQUIRE(sub.rank() == 2);
  REQUIRE(sub[0] == 20);
  REQUIRE(sub[1] == 30);

  auto first = v.subset<0, 1>();
  REQUIRE(first[0] == 10);
}
