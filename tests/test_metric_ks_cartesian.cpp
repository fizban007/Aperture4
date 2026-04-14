#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "systems/physics/metric_ks_cartesian.hpp"
#include "systems/physics/metric_kerr_schild.hpp"
#include <cmath>

using namespace Aperture;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// Convert spherical KS coordinates to Cartesian
static void sph_to_cart(Scalar r, Scalar th, Scalar phi, Scalar a,
                        Scalar& x, Scalar& y, Scalar& z) {
  Scalar phi_ks = phi + std::atan2(a, r);
  Scalar ra = std::sqrt(r * r + a * a);
  Scalar sth = std::sin(th);
  x = ra * sth * std::cos(phi_ks);
  y = ra * sth * std::sin(phi_ks);
  z = r * std::cos(th);
}

TEST_CASE("KS Cartesian: radius recovery", "[metric_ks_cart]") {
  Scalar a = 0.9;

  SECTION("equatorial plane") {
    Scalar r = 5.0, th = M_PI / 2.0, phi = 0.7;
    Scalar x, y, z;
    sph_to_cart(r, th, phi, a, x, y, z);
    CHECK_THAT(Metric_KS_Cart::radius(x, y, z, a),
               WithinRel(r, Scalar(1e-6)));
  }

  SECTION("polar axis") {
    Scalar r = 3.0;
    Scalar x = 0.0, y = 0.0, z = r;
    CHECK_THAT(Metric_KS_Cart::radius(x, y, z, a),
               WithinRel(r, Scalar(1e-6)));
  }

  SECTION("general point") {
    Scalar r = 8.0, th = M_PI / 3.0, phi = 1.2;
    Scalar x, y, z;
    sph_to_cart(r, th, phi, a, x, y, z);
    CHECK_THAT(Metric_KS_Cart::radius(x, y, z, a),
               WithinRel(r, Scalar(1e-6)));
  }

  SECTION("Schwarzschild limit a=0") {
    Scalar x = 3.0, y = 4.0, z = 0.0;
    Scalar r_expect = std::sqrt(x * x + y * y + z * z);
    CHECK_THAT(Metric_KS_Cart::radius(x, y, z, Scalar(0.0)),
               WithinRel(r_expect, Scalar(1e-6)));
  }
}

TEST_CASE("KS Cartesian: null vector unit norm", "[metric_ks_cart]") {
  Scalar a = 0.9;
  Scalar r = 5.0, th = M_PI / 3.0, phi = M_PI / 4.0;
  Scalar x, y, z;
  sph_to_cart(r, th, phi, a, x, y, z);

  Scalar r_rec = Metric_KS_Cart::radius(x, y, z, a);
  auto l = Metric_KS_Cart::null_covector(x, y, z, a, r_rec);
  Scalar norm2 = l[0] * l[0] + l[1] * l[1] + l[2] * l[2];
  CHECK_THAT(norm2, WithinAbs(Scalar(1.0), Scalar(1e-6)));
}

TEST_CASE("KS Cartesian: lapse matches spherical KS", "[metric_ks_cart]") {
  Scalar a = 0.9;
  Scalar r = 5.0, th = M_PI / 3.0, phi = M_PI / 4.0;
  Scalar sth = std::sin(th), cth = std::cos(th);
  Scalar x, y, z;
  sph_to_cart(r, th, phi, a, x, y, z);

  Scalar alpha_sph = Metric_KS::alpha(a, r, sth, cth);
  Scalar alpha_cart = Metric_KS_Cart::alpha(x, y, z, a);
  CHECK_THAT(alpha_cart, WithinRel(alpha_sph, Scalar(1e-6)));
}

TEST_CASE("KS Cartesian: f matches Z from spherical KS", "[metric_ks_cart]") {
  Scalar a = 0.9;
  Scalar r = 5.0, th = M_PI / 3.0, phi = 0.0;
  Scalar sth = std::sin(th), cth = std::cos(th);
  Scalar x, y, z;
  sph_to_cart(r, th, phi, a, x, y, z);

  Scalar r_rec = Metric_KS_Cart::radius(x, y, z, a);
  Scalar f_cart = Metric_KS_Cart::f_ks(r_rec, z, a);
  Scalar Z_sph = Metric_KS::Z(a, r, sth, cth);
  CHECK_THAT(f_cart, WithinRel(Z_sph, Scalar(1e-6)));
}

TEST_CASE("KS Cartesian: raise/lower roundtrip", "[metric_ks_cart]") {
  Scalar a = 0.9;
  Scalar r = 5.0, th = M_PI / 3.0, phi = M_PI / 4.0;
  Scalar x, y, z;
  sph_to_cart(r, th, phi, a, x, y, z);

  Scalar r_rec = Metric_KS_Cart::radius(x, y, z, a);
  auto l = Metric_KS_Cart::null_covector(x, y, z, a, r_rec);
  Scalar fv = Metric_KS_Cart::f_ks(r_rec, z, a);

  vec_t<Scalar, 3> v;
  v[0] = 1.3; v[1] = -0.7; v[2] = 2.1;

  auto v_low = Metric_KS_Cart::lower(v, fv, l);
  auto v_rec = Metric_KS_Cart::raise(v_low, fv, l);

  CHECK_THAT(v_rec[0], WithinRel(v[0], Scalar(1e-5)));
  CHECK_THAT(v_rec[1], WithinRel(v[1], Scalar(1e-5)));
  CHECK_THAT(v_rec[2], WithinRel(v[2], Scalar(1e-5)));
}

TEST_CASE("KS Cartesian: Schwarzschild lapse", "[metric_ks_cart]") {
  Scalar a = 0.0;
  Scalar r = 3.0;
  Scalar x = r, y = 0.0, z = 0.0;
  Scalar alpha_expect = Scalar(1.0) / std::sqrt(Scalar(1.0) + Scalar(2.0) / r);
  CHECK_THAT(Metric_KS_Cart::alpha(x, y, z, a),
             WithinRel(alpha_expect, Scalar(1e-6)));
}

TEST_CASE("KS Cartesian: compute_all consistency", "[metric_ks_cart]") {
  Scalar a = 0.9;
  Scalar r = 5.0, th = M_PI / 3.0, phi = M_PI / 4.0;
  Scalar x, y, z;
  sph_to_cart(r, th, phi, a, x, y, z);

  Scalar r_all, fv_all, alp_all, sg_all;
  vec_t<Scalar, 3> l_all, bu_all, sgb_all;
  Metric_KS_Cart::compute_all(x, y, z, a, r_all, l_all, fv_all,
                               alp_all, bu_all, sgb_all, sg_all);

  // Check individual functions agree
  CHECK_THAT(r_all, WithinRel(Metric_KS_Cart::radius(x, y, z, a),
                               Scalar(1e-6)));
  CHECK_THAT(alp_all, WithinRel(Metric_KS_Cart::alpha(x, y, z, a),
                                 Scalar(1e-6)));

  // Check sqrt(gamma) * beta^i = sg * bu
  for (int i = 0; i < 3; i++) {
    CHECK_THAT(sgb_all[i], WithinRel(sg_all * bu_all[i], Scalar(1e-5)));
  }

  // Check that alpha * sqrt(gamma) = 1  (since alpha = 1/sqrt(1+f) and sg = sqrt(1+f))
  CHECK_THAT(alp_all * sg_all, WithinAbs(Scalar(1.0), Scalar(1e-6)));
}

TEST_CASE("KS Cartesian: horizon radius", "[metric_ks_cart]") {
  CHECK_THAT(Metric_KS_Cart::rH(Scalar(0.0)),
             WithinAbs(Scalar(2.0), Scalar(1e-6)));
  CHECK_THAT(Metric_KS_Cart::rH(Scalar(1.0)),
             WithinAbs(Scalar(1.0), Scalar(1e-6)));
  Scalar a = 0.9;
  Scalar rh = Scalar(1.0) + std::sqrt(Scalar(1.0) - a * a);
  CHECK_THAT(Metric_KS_Cart::rH(a), WithinRel(rh, Scalar(1e-6)));
}

TEST_CASE("KS Cartesian: dot products", "[metric_ks_cart]") {
  Scalar a = 0.9;
  Scalar r = 5.0, th = M_PI / 3.0, phi = M_PI / 4.0;
  Scalar x, y, z;
  sph_to_cart(r, th, phi, a, x, y, z);

  Scalar r_rec = Metric_KS_Cart::radius(x, y, z, a);
  auto l = Metric_KS_Cart::null_covector(x, y, z, a, r_rec);
  Scalar fv = Metric_KS_Cart::f_ks(r_rec, z, a);

  vec_t<Scalar, 3> u, v;
  u[0] = 1.0; u[1] = 0.5; u[2] = -0.3;
  v[0] = 0.2; v[1] = -1.0; v[2] = 0.8;

  // dot_product_u(u, v) should equal u_i v^i = gamma_ij u^i v^j
  Scalar dot_uv = Metric_KS_Cart::dot_product_u(u, v, fv, l);

  // Compute manually: delta_ij u^i v^j + f (l.u)(l.v)
  Scalar flat = u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
  Scalar lu = l[0]*u[0] + l[1]*u[1] + l[2]*u[2];
  Scalar lv = l[0]*v[0] + l[1]*v[1] + l[2]*v[2];
  CHECK_THAT(dot_uv, WithinRel(flat + fv * lu * lv, Scalar(1e-5)));

  // dot_product_u(u, u) should be positive (metric is positive definite)
  CHECK(Metric_KS_Cart::dot_product_u(u, u, fv, l) > 0.0);

  // lower then dot_product_l should give same result
  auto u_low = Metric_KS_Cart::lower(u, fv, l);
  auto v_low = Metric_KS_Cart::lower(v, fv, l);
  Scalar dot_low = Metric_KS_Cart::dot_product_l(u_low, v_low, fv, l);
  CHECK_THAT(dot_low, WithinRel(dot_uv, Scalar(1e-4)));
}
