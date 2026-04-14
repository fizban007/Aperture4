#include "catch2/catch_all.hpp"
#include "core/random.h"
#include <cmath>
#include <vector>

using namespace Aperture;

TEST_CASE("rand_state determinism: same seed gives same sequence", "[rng]") {
  rand_state s1, s2;
  s1.init(12345);
  s2.init(12345);

  for (int i = 0; i < 100; i++) {
    REQUIRE(s1.next() == s2.next());
  }
}

TEST_CASE("rand_state different seeds give different sequences", "[rng]") {
  rand_state s1, s2;
  s1.init(111);
  s2.init(222);

  bool any_differ = false;
  for (int i = 0; i < 10; i++) {
    if (s1.next() != s2.next()) {
      any_differ = true;
      break;
    }
  }
  REQUIRE(any_differ);
}

TEST_CASE("rng_uniform values are in [0, 1)", "[rng]") {
  rand_state state;
  state.init(42);

  int N = 10000;
  for (int i = 0; i < N; i++) {
    double u = rng_uniform<double>(state);
    REQUIRE(u >= 0.0);
    REQUIRE(u < 1.0);
  }
}

TEST_CASE("rng_uniform distribution is approximately uniform", "[rng]") {
  rand_state state;
  state.init(42);

  int N = 100000;
  int bins = 10;
  std::vector<int> hist(bins, 0);

  for (int i = 0; i < N; i++) {
    double u = rng_uniform<double>(state);
    int bin = static_cast<int>(u * bins);
    if (bin >= bins) bin = bins - 1;
    hist[bin]++;
  }

  double expected = static_cast<double>(N) / bins;
  for (int b = 0; b < bins; b++) {
    // Each bin should be within 5% of expected
    REQUIRE(hist[b] > expected * 0.90);
    REQUIRE(hist[b] < expected * 1.10);
  }
}

TEST_CASE("rng_gaussian mean and variance", "[rng]") {
  rand_state state;
  state.init(42);

  int N = 50000;
  double sigma = 2.0;
  double sum = 0.0;
  double sum_sq = 0.0;

  for (int i = 0; i < N; i++) {
    double x = rng_gaussian<double>(state, sigma);
    sum += x;
    sum_sq += x * x;
  }

  double mean = sum / N;
  double variance = sum_sq / N - mean * mean;

  REQUIRE(mean == Catch::Approx(0.0).margin(0.1));
  REQUIRE(variance == Catch::Approx(sigma * sigma).epsilon(0.1));
}

TEST_CASE("jump produces a different sequence", "[rng]") {
  rand_state s1, s2;
  s1.init(42);
  s2.init(42);

  s2.jump();

  bool any_differ = false;
  for (int i = 0; i < 10; i++) {
    if (s1.next() != s2.next()) {
      any_differ = true;
      break;
    }
  }
  REQUIRE(any_differ);
}

TEST_CASE("rng_t host wrapper matches free functions", "[rng]") {
  rand_state s1, s2;
  s1.init(99);
  s2.init(99);

  rng_t<exec_tags::host> rng(&s1);

  for (int i = 0; i < 50; i++) {
    double from_wrapper = rng.uniform<double>();
    double from_free = rng_uniform<double>(s2);
    REQUIRE(from_wrapper == from_free);
  }
}

TEST_CASE("rng_poisson basic correctness", "[rng]") {
  rand_state state;
  state.init(42);

  double lambda = 5.0;
  int N = 20000;
  double sum = 0;

  for (int i = 0; i < N; i++) {
    int k = rng_poisson<double>(state, lambda);
    REQUIRE(k >= 0);
    sum += k;
  }

  double mean = sum / N;
  // Poisson mean should be lambda
  REQUIRE(mean == Catch::Approx(lambda).epsilon(0.1));
}
