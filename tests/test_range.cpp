#include "catch2/catch_all.hpp"
#include "utils/range.hpp"
#include <vector>

using namespace Aperture;

TEST_CASE("range(n) iterates [0, n)", "[range]") {
  std::vector<int> vals;
  for (auto i : range(5)) {
    vals.push_back(i);
  }
  REQUIRE(vals.size() == 5);
  REQUIRE(vals[0] == 0);
  REQUIRE(vals[4] == 4);
}

TEST_CASE("range(a, b) iterates [a, b)", "[range]") {
  std::vector<int> vals;
  for (auto i : range(3, 8)) {
    vals.push_back(i);
  }
  REQUIRE(vals.size() == 5);
  REQUIRE(vals[0] == 3);
  REQUIRE(vals[4] == 7);
}

TEST_CASE("range with step", "[range]") {
  std::vector<int> vals;
  for (auto i : range(0, 10).step(3)) {
    vals.push_back(i);
  }
  // 0, 3, 6, 9
  REQUIRE(vals.size() == 4);
  REQUIRE(vals[0] == 0);
  REQUIRE(vals[1] == 3);
  REQUIRE(vals[2] == 6);
  REQUIRE(vals[3] == 9);
}

TEST_CASE("range with step 2", "[range]") {
  std::vector<int> vals;
  for (auto i : range(1, 10).step(2)) {
    vals.push_back(i);
  }
  // 1, 3, 5, 7, 9
  REQUIRE(vals.size() == 5);
  REQUIRE(vals[0] == 1);
  REQUIRE(vals[4] == 9);
}

TEST_CASE("range empty when begin >= end", "[range]") {
  std::vector<int> vals;
  for (auto i : range(5, 5)) {
    vals.push_back(i);
  }
  REQUIRE(vals.empty());

  for (auto i : range(0)) {
    vals.push_back(i);
  }
  REQUIRE(vals.empty());
}

TEST_CASE("range iterator difference", "[range]") {
  auto r = range(10, 20);
  auto it_begin = r.begin();
  auto it_end = r.end();
  REQUIRE(it_end - it_begin == 10);
}

TEST_CASE("indices over container", "[range]") {
  std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
  std::vector<size_t> idx;
  for (auto i : indices(v)) {
    idx.push_back(i);
  }
  REQUIRE(idx.size() == 4);
  REQUIRE(idx[0] == 0);
  REQUIRE(idx[3] == 3);
}

TEST_CASE("indices over C array", "[range]") {
  int arr[6] = {};
  std::vector<size_t> idx;
  for (auto i : indices(arr)) {
    idx.push_back(i);
  }
  REQUIRE(idx.size() == 6);
  REQUIRE(idx[5] == 5);
}
