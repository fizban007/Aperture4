#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "systems/prismatic/prismatic_mesh_metric.h"
#include <cmath>
#include <memory>

using namespace Aperture;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

static constexpr int TEST_L = 2;
static constexpr int TEST_NR = 5;
static constexpr double TEST_RMIN = 1.0;
static constexpr double TEST_RMAX = 5.0;

TEST_CASE("Metric mesh: flat Hodge stars match base class",
          "[prismatic_metric]") {
  // Build base mesh (flat Euclidean Hodge stars)
  prismatic_mesh base;
  base.build(TEST_L, TEST_NR, TEST_RMIN, TEST_RMAX);

  // Build metric mesh with flat metric
  prismatic_mesh_metric met_mesh;
  met_mesh.build(TEST_L, TEST_NR, TEST_RMIN, TEST_RMAX);
  met_mesh.compute_metric(flat_spherical_metric{});

  REQUIRE(met_mesh.m_N_edges == base.m_N_edges);
  REQUIRE(met_mesh.m_N_faces == base.m_N_faces);

  // With the Jacobian-based metric distance, flat-space metric distances
  // are exactly Euclidean.  Face areas scale by sqrt(gamma)/sqrt(gamma_flat)
  // which is exactly 1 for flat space.  So the metric Hodge stars should
  // closely match the base class values.
  double max_h2_err = 0;
  for (int f = 0; f < base.m_N_faces; f++) {
    if (base.hodge2[f] > 1e-10) {
      double err = std::abs(met_mesh.hodge2[f] - base.hodge2[f]) /
                   base.hodge2[f];
      if (err > max_h2_err) max_h2_err = err;
    }
  }
  INFO("Max hodge2 relative error: " << max_h2_err);
  CHECK(max_h2_err < 0.05);

  double max_h1_err = 0;
  for (int e = 0; e < base.m_N_edges; e++) {
    if (base.hodge1_inv[e] > 1e-10) {
      double err = std::abs(met_mesh.hodge1_inv[e] - base.hodge1_inv[e]) /
                   base.hodge1_inv[e];
      if (err > max_h1_err) max_h1_err = err;
    }
  }
  INFO("Max hodge1_inv relative error: " << max_h1_err);
  CHECK(max_h1_err < 0.05);
}

TEST_CASE("Metric mesh: per-element lapse is 1 for flat space",
          "[prismatic_metric]") {
  prismatic_mesh_metric mesh;
  mesh.build(TEST_L, TEST_NR, TEST_RMIN, TEST_RMAX);
  mesh.compute_metric(flat_spherical_metric{});

  for (int e = 0; e < mesh.m_N_edges; e++) {
    CHECK_THAT(mesh.edge_alpha[e], WithinAbs(Scalar(1.0), Scalar(1e-6)));
    CHECK_THAT((double)mesh.edge_sq_gamma_beta_r[e],
               WithinAbs(0.0, 1e-6));
  }
  for (int f = 0; f < mesh.m_N_faces; f++) {
    CHECK_THAT(mesh.face_alpha[f], WithinAbs(Scalar(1.0), Scalar(1e-6)));
  }
}

TEST_CASE("Metric mesh: KS lapse matches metric_kerr_schild",
          "[prismatic_metric]") {
  Scalar a = 0.9;
  prismatic_mesh_metric mesh;
  mesh.build(TEST_L, TEST_NR, TEST_RMIN, TEST_RMAX);
  mesh.compute_metric(ks_spherical_metric{a});

  // Spot-check some edges
  for (int e = 0; e < mesh.m_N_edges; e += mesh.m_N_edges / 10) {
    Scalar r = mesh.edge_r_coord[e];
    Scalar sth = mesh.edge_sth[e];
    Scalar cth = mesh.edge_cth[e];
    Scalar alpha_exp = Metric_KS::alpha(a, r, sth, cth);
    CHECK_THAT(mesh.edge_alpha[e], WithinRel(alpha_exp, Scalar(1e-5)));
  }
}

TEST_CASE("Metric mesh: KS Hodge stars differ from flat",
          "[prismatic_metric]") {
  prismatic_mesh_metric flat_mesh, ks_mesh;
  flat_mesh.build(TEST_L, TEST_NR, TEST_RMIN, TEST_RMAX);
  flat_mesh.compute_metric(flat_spherical_metric{});
  ks_mesh.build(TEST_L, TEST_NR, TEST_RMIN, TEST_RMAX);
  ks_mesh.compute_metric(ks_spherical_metric{0.9});

  // KS and flat Hodge stars should differ
  int n_diff = 0;
  for (int f = 0; f < flat_mesh.m_N_faces; f++) {
    if (std::abs(ks_mesh.hodge2[f] - flat_mesh.hodge2[f]) > 1e-6)
      n_diff++;
  }
  CHECK(n_diff > flat_mesh.m_N_faces / 2);
}

TEST_CASE("Metric mesh: Schwarzschild (a=0) is close to flat at large r",
          "[prismatic_metric]") {
  prismatic_mesh_metric flat_mesh, schw_mesh;
  // Use large r_min so we're far from the BH
  flat_mesh.build(TEST_L, TEST_NR, 10.0, 50.0);
  flat_mesh.compute_metric(flat_spherical_metric{});
  schw_mesh.build(TEST_L, TEST_NR, 10.0, 50.0);
  schw_mesh.compute_metric(ks_spherical_metric{0.0});

  // At r >> 2M, KS ≈ flat, so Hodge stars should be close
  double max_err = 0;
  for (int f = 0; f < flat_mesh.m_N_faces; f++) {
    if (flat_mesh.hodge2[f] > 1e-10) {
      double err = std::abs(schw_mesh.hodge2[f] - flat_mesh.hodge2[f]) /
                   flat_mesh.hodge2[f];
      if (err > max_err) max_err = err;
    }
  }
  // At r=10M, 2M/r = 0.2, so γ_rr = 1.2 — metric distances differ by
  // up to ~10% from flat.  The Hodge star ratio compounds radial and
  // angular corrections, so ~8% max error is expected.
  INFO("Max hodge2 error (Schwarzschild vs flat at large r): " << max_err);
  CHECK(max_err < 0.10);
}
