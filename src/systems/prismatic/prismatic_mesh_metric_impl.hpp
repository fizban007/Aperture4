#pragma once

#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_mesh_metric.h"
#include "utils/gauss_quadrature.h"
#include "utils/logger.h"

namespace Aperture {

namespace prismatic_metric_helpers {

// -------------------------------------------------------------------------
// Everything in (r, θ, φ) coordinate basis.
//
// Primal elements are parameterized coord-linearly:
//
//   horizontal edge (fixed r):
//       (r, θ, φ)(s) = (r, θ_a + s·Δθ, φ_a + s·Δφ),   s ∈ [0, 1]
//
//   vertical edge   (fixed θ, φ):
//       (r, θ, φ)(s) = (r_0 + s·Δr, θ, φ),            s ∈ [0, 1]
//
//   triangular face (fixed r, barycentric in (θ, φ)):
//       x(u, v) = vertex_a + u·(vertex_b − a) + v·(vertex_c − a)
//
//   rectangular face (bilinear in r × arc-parameter):
//       (r, θ, φ)(u, v) = ((1-v)·r_0 + v·r_1,
//                          θ_a + u·Δθ,
//                          φ_a + u·Δφ)
//
// All integrals use the spherical metric components γ_rr, γ_θθ, γ_φφ, γ_rφ
// directly — no Cartesian inner products.
// -------------------------------------------------------------------------

// Angular difference with wrap handling: returns φ_b − φ_a in (−π, π].
HD_INLINE double angular_diff(double phi_a, double phi_b) {
  double d = phi_b - phi_a;
  const double pi = 3.14159265358979323846;
  if (d > pi) d -= 2.0 * pi;
  if (d < -pi) d += 2.0 * pi;
  return d;
}

// Spherical coordinates of a mesh vertex.  vidx = k · N_vert_s + s.
//
// Derive θ in double via acos on the stored float sphere_vz rather than
// reading the stored float sphere_theta: float(acos(-1)) ≈ 3.1415927 is
// slightly > π in double precision, so sin(sphere_theta[polar]) returns
// a tiny negative value (-8.7e-8) which cascades into negative √γ on
// polar-adjacent horizontal edges and blows up the 1/√γ factor in the
// shift cross term.  Clamping cos θ to [-1, 1] and using acos gives
// θ ∈ [0, π] with sin θ ≥ 0 to double precision.
template <typename Mesh>
HD_INLINE void vertex_sph(const Mesh& mp, int vidx, double& r, double& sth,
                          double& cth, double& phi) {
  int k = vidx / mp.N_vert_s;
  int sphv = vidx % mp.N_vert_s;
  r = mp.radii[k];
  double ct = (double)mp.sphere_vz[sphv];
  if (ct > 1.0) ct = 1.0;
  if (ct < -1.0) ct = -1.0;
  double theta = math::acos(ct);
  sth = math::sin(theta);
  cth = math::cos(theta);
  phi = mp.sphere_phi[sphv];
}

// Spherical circumcenter (θ, φ) of triangle with sphere-vertex indices
// a, b, c.  The circumcenter is the unit normal to the chord plane, i.e.
// the point on the sphere equidistant (in great-circle arc) from all
// three vertices.  Matches the convention in prismatic_mesh::compute_-
// geometric_dual so the metric and base-class Hodge stars agree in flat
// space.
template <typename Mesh>
HD_INLINE void tri_circumcenter_sph(const Mesh& mp, int a, int b, int c,
                                    double& th_cc, double& ph_cc) {
  double e1x = mp.sphere_vx[b] - mp.sphere_vx[a];
  double e1y = mp.sphere_vy[b] - mp.sphere_vy[a];
  double e1z = mp.sphere_vz[b] - mp.sphere_vz[a];
  double e2x = mp.sphere_vx[c] - mp.sphere_vx[a];
  double e2y = mp.sphere_vy[c] - mp.sphere_vy[a];
  double e2z = mp.sphere_vz[c] - mp.sphere_vz[a];
  double nx = e1y * e2z - e1z * e2y;
  double ny = e1z * e2x - e1x * e2z;
  double nz = e1x * e2y - e1y * e2x;
  double nlen = math::sqrt(nx * nx + ny * ny + nz * nz);
  if (nlen > 0) {
    nx /= nlen; ny /= nlen; nz /= nlen;
    // Orient outward (same hemisphere as the triangle centroid).
    double mx = (mp.sphere_vx[a] + mp.sphere_vx[b] + mp.sphere_vx[c]);
    double my = (mp.sphere_vy[a] + mp.sphere_vy[b] + mp.sphere_vy[c]);
    double mz = (mp.sphere_vz[a] + mp.sphere_vz[b] + mp.sphere_vz[c]);
    if (nx * mx + ny * my + nz * mz < 0) {
      nx = -nx; ny = -ny; nz = -nz;
    }
  }
  if (nz >  1.0) nz =  1.0;
  if (nz < -1.0) nz = -1.0;
  th_cc = math::acos(nz);
  ph_cc = math::atan2(ny, nx);
}

// Horizontal edge length on shell r, along the great-circle arc from
// û_a to û_b.  The slerp parametrization matches the primal edge geometry.
template <typename Metric>
HD_INLINE double
horizontal_edge_length(const Metric& met, double r,
                       double ux_a, double uy_a, double uz_a,
                       double ux_b, double uy_b, double uz_b) {
  double dot = ux_a*ux_b + uy_a*uy_b + uz_a*uz_b;
  if (dot >  1.0) dot =  1.0;
  if (dot < -1.0) dot = -1.0;
  double alpha = math::acos(dot);
  double sa = math::sin(alpha);

  return gauss_quad(
      [&](double s) {
        double hx, hy, hz, dhx, dhy, dhz;
        if (sa < 1e-12) {
          hx = ux_a; hy = uy_a; hz = uz_a;
          dhx = ux_b - ux_a; dhy = uy_b - uy_a; dhz = uz_b - uz_a;
        } else {
          double w0 = math::sin((1.0 - s) * alpha) / sa;
          double w1 = math::sin(s * alpha) / sa;
          hx = w0*ux_a + w1*ux_b;
          hy = w0*uy_a + w1*uy_b;
          hz = w0*uz_a + w1*uz_b;
          double dw0 = -alpha * math::cos((1.0 - s) * alpha) / sa;
          double dw1 =  alpha * math::cos(s * alpha) / sa;
          dhx = dw0*ux_a + dw1*ux_b;
          dhy = dw0*uy_a + dw1*uy_b;
          dhz = dw0*uz_a + dw1*uz_b;
        }
        double cth = hz;
        double sth2 = hx*hx + hy*hy;
        double sth = math::sqrt(sth2 > 0 ? sth2 : 0.0);
        if (sth < 1e-30) return 0.0;
        double inv_sth = 1.0 / sth;
        double inv_sth2 = inv_sth * inv_sth;
        double dth_ds = -dhz * inv_sth;
        double dph_ds = (hx*dhy - hy*dhx) * inv_sth2;
        double g22 = met.g_thth(r, sth, cth);
        double g33 = met.g_phph(r, sth, cth);
        double g23 = met.g_thph(r, sth, cth);
        double q = g22*dth_ds*dth_ds + g33*dph_ds*dph_ds
                 + 2.0*g23*dth_ds*dph_ds;
        return math::sqrt(q > 0 ? q : 0.0);
      },
      0.0, 1.0);
}

// Vertical (radial) edge length at fixed (θ, φ).
template <typename Metric>
HD_INLINE double
radial_edge_length(const Metric& met, double r_0, double r_1, double theta) {
  double dr = r_1 - r_0;
  double sth = math::sin(theta), cth = math::cos(theta);
  return gauss_quad(
      [&](double s) {
        double r = r_0 + s * dr;
        double g11 = met.g_rr(r, sth, cth);
        return math::sqrt(g11) * math::abs(dr);
      },
      0.0, 1.0);
}

// Generic triangle area in (r, θ, φ) coord-linear parametrization.
// Works for any three vertices (possibly at different r).
template <typename Metric>
HD_INLINE double
coord_triangle_area(const Metric& met, double r_a, double th_a, double ph_a,
                    double r_b, double th_b, double ph_b, double r_c,
                    double th_c, double ph_c) {
  double drB = r_b - r_a, dthB = th_b - th_a, dphB = angular_diff(ph_a, ph_b);
  double drC = r_c - r_a, dthC = th_c - th_a, dphC = angular_diff(ph_a, ph_c);
  return gauss_quad(
      [&](double u) {
        return gauss_quad(
            [&](double v) {
              double r = r_a + u * drB + v * drC;
              double th = th_a + u * dthB + v * dthC;
              if (r < 1e-12) return 0.0;
              double sth = math::sin(th), cth = math::cos(th);
              double g11 = met.g_rr(r, sth, cth);
              double g22 = met.g_thth(r, sth, cth);
              double g33 = met.g_phph(r, sth, cth);
              double g13 = met.g_rph(r, sth, cth);
              double g23 = met.g_thph(r, sth, cth);
              double huu = g11 * drB * drB + g22 * dthB * dthB +
                           g33 * dphB * dphB + 2.0 * g13 * drB * dphB +
                           2.0 * g23 * dthB * dphB;
              double hvv = g11 * drC * drC + g22 * dthC * dthC +
                           g33 * dphC * dphC + 2.0 * g13 * drC * dphC +
                           2.0 * g23 * dthC * dphC;
              double huv = g11 * drB * drC + g22 * dthB * dthC +
                           g33 * dphB * dphC + g13 * (drB * dphC + drC * dphB) +
                           g23 * (dthB * dphC + dthC * dphB);
              double det = huu * hvv - huv * huv;
              return math::sqrt(det > 0 ? det : 0.0);
            },
            0.0, 1.0 - u);
      },
      0.0, 1.0);
}

// Shell-restricted triangle area: all three vertices at the same r.
//
// The face is the spherical triangle bounded by great-circle arcs, NOT a
// coordinate-linear (θ,φ) triangle.  We parametrize via normalized
// barycentric interpolation on the unit sphere:
//     û(u,v) = normalize(λ_a·û_a + λ_b·û_b + λ_c·û_c)
// with λ_a = 1-u-v, λ_b = u, λ_c = v, and integrate the metric area
// element over the reference triangle {u,v ≥ 0, u+v ≤ 1}.
//
// The angular derivatives ∂θ/∂u, ∂φ/∂u are extracted from ∂û/∂u via the
// chain rule on û → (θ,φ).  For flat space this reproduces the exact
// Girard area; for curved metrics the Gauss quadrature accounts for the
// metric distortion point-by-point.
template <typename Metric>
HD_INLINE double
shell_triangle_area(const Metric& met, double r,
                    double ux_a, double uy_a, double uz_a,
                    double ux_b, double uy_b, double uz_b,
                    double ux_c, double uy_c, double uz_c) {
  // Constant derivatives of the unnormalized linear combination:
  //   ∂Q/∂u = û_b - û_a,   ∂Q/∂v = û_c - û_a
  double dqu_x = ux_b - ux_a, dqu_y = uy_b - uy_a, dqu_z = uz_b - uz_a;
  double dqv_x = ux_c - ux_a, dqv_y = uy_c - uy_a, dqv_z = uz_c - uz_a;

  return gauss_quad(
      [&](double u) {
        return gauss_quad(
            [&](double v) {
              // Normalized barycentric position on the unit sphere.
              double la = 1.0 - u - v, lb = u, lc = v;
              double qx = la*ux_a + lb*ux_b + lc*ux_c;
              double qy = la*uy_a + lb*uy_b + lc*uy_c;
              double qz = la*uz_a + lb*uz_b + lc*uz_c;
              double qn = math::sqrt(qx*qx + qy*qy + qz*qz);
              if (qn < 1e-30) return 0.0;
              double inv_qn = 1.0 / qn;
              double hx = qx*inv_qn, hy = qy*inv_qn, hz = qz*inv_qn;

              // ∂û/∂ξ = (∂Q/∂ξ − (û·∂Q/∂ξ)·û) / |Q|
              double pu = hx*dqu_x + hy*dqu_y + hz*dqu_z;
              double duhdx = (dqu_x - pu*hx)*inv_qn;
              double duhdy = (dqu_y - pu*hy)*inv_qn;
              double duhdz = (dqu_z - pu*hz)*inv_qn;
              double pv = hx*dqv_x + hy*dqv_y + hz*dqv_z;
              double dvhdx = (dqv_x - pv*hx)*inv_qn;
              double dvhdy = (dqv_y - pv*hy)*inv_qn;
              double dvhdz = (dqv_z - pv*hz)*inv_qn;

              // Extract (θ,φ) and angular derivatives.
              double cth = hz;
              double sth2 = hx*hx + hy*hy;
              double sth = math::sqrt(sth2 > 0 ? sth2 : 0.0);
              if (sth < 1e-30) return 0.0;  // at pole, area element vanishes
              double inv_sth = 1.0 / sth;
              double inv_sth2 = inv_sth * inv_sth;

              // ∂θ/∂ξ = −(∂û/∂ξ)_z / sinθ
              double dth_du = -duhdz * inv_sth;
              double dth_dv = -dvhdz * inv_sth;
              // ∂φ/∂ξ = (û_x·(∂û/∂ξ)_y − û_y·(∂û/∂ξ)_x) / sin²θ
              double dph_du = (hx*duhdy - hy*duhdx) * inv_sth2;
              double dph_dv = (hx*dvhdy - hy*dvhdx) * inv_sth2;

              // Metric-weighted induced area element.
              double g22 = met.g_thth(r, sth, cth);
              double g33 = met.g_phph(r, sth, cth);
              double g23 = met.g_thph(r, sth, cth);
              double huu = g22*dth_du*dth_du + g33*dph_du*dph_du
                         + 2.0*g23*dth_du*dph_du;
              double hvv = g22*dth_dv*dth_dv + g33*dph_dv*dph_dv
                         + 2.0*g23*dth_dv*dph_dv;
              double huv = g22*dth_du*dth_dv + g33*dph_du*dph_dv
                         + g23*(dth_du*dph_dv + dth_dv*dph_du);
              double det = huu*hvv - huv*huv;
              return math::sqrt(det > 0 ? det : 0.0);
            },
            0.0, 1.0 - u);
      },
      0.0, 1.0);
}

// Rectangular face area bounded by great-circle arcs at r_0 and r_1 with
// angular endpoints given by unit vectors û_a and û_b.  The angular path
// is a slerp (great circle), matching the primal face geometry.
template <typename Metric>
HD_INLINE double
rectangular_face_area(const Metric& met, double r_0, double r_1,
                      double ux_a, double uy_a, double uz_a,
                      double ux_b, double uy_b, double uz_b) {
  double dr = r_1 - r_0;
  // Slerp setup: angle α between û_a and û_b.
  double dot = ux_a*ux_b + uy_a*uy_b + uz_a*uz_b;
  if (dot >  1.0) dot =  1.0;
  if (dot < -1.0) dot = -1.0;
  double alpha = math::acos(dot);
  double sa = math::sin(alpha);

  return gauss_quad(
      [&](double u) {
        // Slerp position and derivative on the unit sphere.
        double hx, hy, hz, dhx, dhy, dhz;
        if (sa < 1e-12) {
          hx = ux_a; hy = uy_a; hz = uz_a;
          dhx = ux_b - ux_a; dhy = uy_b - uy_a; dhz = uz_b - uz_a;
        } else {
          double w0 = math::sin((1.0 - u) * alpha) / sa;
          double w1 = math::sin(u * alpha) / sa;
          hx = w0*ux_a + w1*ux_b;
          hy = w0*uy_a + w1*uy_b;
          hz = w0*uz_a + w1*uz_b;
          double dw0 = -alpha * math::cos((1.0 - u) * alpha) / sa;
          double dw1 =  alpha * math::cos(u * alpha) / sa;
          dhx = dw0*ux_a + dw1*ux_b;
          dhy = dw0*uy_a + dw1*uy_b;
          dhz = dw0*uz_a + dw1*uz_b;
        }
        // Extract (θ, φ) and angular derivatives.
        double cth = hz;
        double sth2 = hx*hx + hy*hy;
        double sth = math::sqrt(sth2 > 0 ? sth2 : 0.0);
        if (sth < 1e-30) return 0.0;
        double inv_sth = 1.0 / sth;
        double inv_sth2 = inv_sth * inv_sth;
        double dth_du = -dhz * inv_sth;
        double dph_du = (hx*dhy - hy*dhx) * inv_sth2;

        return gauss_quad(
            [&](double v) {
              double r = r_0 + v * dr;
              double g11 = met.g_rr(r, sth, cth);
              double g22 = met.g_thth(r, sth, cth);
              double g33 = met.g_phph(r, sth, cth);
              double g13 = met.g_rph(r, sth, cth);
              double g23 = met.g_thph(r, sth, cth);
              // ∂/∂u = (0, dth_du, dph_du); ∂/∂v = (dr, 0, 0).
              double huu = g22*dth_du*dth_du + g33*dph_du*dph_du
                         + 2.0*g23*dth_du*dph_du;
              double hvv = g11 * dr * dr;
              double huv = g13 * dr * dph_du;
              double det = huu * hvv - huv * huv;
              return math::sqrt(det > 0 ? det : 0.0);
            },
            0.0, 1.0);
      },
      0.0, 1.0);
}

// Polygon area via fan from an explicit center point.  Uses
// coord_triangle_area for each sub-triangle so works with arbitrary vertex
// r-values (needed for horizontal-edge dual faces, which span two shells).
template <typename Metric>
HD_INLINE double
polygon_area_about(const Metric& met, double r_c, double th_c, double ph_c,
                   const double* rr, const double* th, const double* ph,
                   int n) {
  if (n < 3) return 0.0;
  double area = 0.0;
  for (int i = 0; i < n; i++) {
    int j = (i + 1) % n;
    area += coord_triangle_area(met, r_c, th_c, ph_c, rr[i], th[i], ph[i],
                                rr[j], th[j], ph[j]);
  }
  return area;
}

// Polygon area via fan from an explicit center point, assuming the fan
// center and every polygon vertex lie on the same shell at radius r.
// Each sub-triangle is a shell triangle, so we only need the angular
// metric sub-block.  Used by vertical-edge dual polygons (Hodge1_inv on
// vertical edges), where the dual face is a tangential polygon at
// r = ½(r_0 + r_1).
template <typename Metric>
HD_INLINE double
shell_polygon_area_about(const Metric& met, double r, double th_c, double ph_c,
                         const double* th, const double* ph, int n) {
  if (n < 3) return 0.0;
  // Convert center and vertices from (θ,φ) to unit vectors for the
  // great-circle-bounded shell_triangle_area.
  double cx = math::sin(th_c) * math::cos(ph_c);
  double cy = math::sin(th_c) * math::sin(ph_c);
  double cz = math::cos(th_c);
  double area = 0.0;
  for (int i = 0; i < n; i++) {
    int j = (i + 1) % n;
    double ui_x = math::sin(th[i])*math::cos(ph[i]);
    double ui_y = math::sin(th[i])*math::sin(ph[i]);
    double ui_z = math::cos(th[i]);
    double uj_x = math::sin(th[j])*math::cos(ph[j]);
    double uj_y = math::sin(th[j])*math::sin(ph[j]);
    double uj_z = math::cos(th[j]);
    area += shell_triangle_area(met, r, cx, cy, cz,
                                ui_x, ui_y, ui_z, uj_x, uj_y, uj_z);
  }
  return area;
}

}  // namespace prismatic_metric_helpers

// =========================================================================
// Template compute_metric<Metric>.
// All per-element metric buffers + Hodge stars computed in (r, θ, φ) coord
// basis; no Cartesian chord lengths or flat triangle areas anywhere.
// =========================================================================
template <typename Metric>
void prismatic_mesh_metric::compute_metric(const Metric& met) {
  using ExecPolicy = prismatic_exec_policy_dynamic;
  using namespace prismatic_metric_helpers;
  auto mem = ExecPolicy::data_mem_type();

  auto alloc = [mem](auto& buf, size_t n) {
    buf.set_memtype(mem);
    buf.resize(n);
  };

  // ----- Allocate per-element metric buffers -----
  alloc(edge_r_coord, m_N_edges);
  alloc(edge_sth, m_N_edges);
  alloc(edge_cth, m_N_edges);
  alloc(edge_alpha, m_N_edges);
  alloc(edge_sq_gamma_beta_r, m_N_edges);
  alloc(edge_sqrt_gamma, m_N_edges);

  alloc(face_r_coord, m_N_faces);
  alloc(face_sth, m_N_faces);
  alloc(face_cth, m_N_faces);
  alloc(face_alpha, m_N_faces);
  alloc(face_sq_gamma_beta_r, m_N_faces);
  alloc(face_sqrt_gamma, m_N_faces);

  // ----- Build edge-to-triangle and vertex-to-triangle adjacency on host -----
  alloc(edge_tris, 2 * m_N_edge_s);
  for (int e = 0; e < m_N_edge_s; e++) {
    edge_tris[2 * e + 0] = -1;
    edge_tris[2 * e + 1] = -1;
  }
  const int* tri_edges_host = tri_edges_s.host_ptr();
  for (int t = 0; t < m_N_tri; t++) {
    for (int j = 0; j < 3; j++) {
      int e = tri_edges_host[t * 3 + j];
      if (edge_tris[2 * e + 0] == -1)
        edge_tris[2 * e + 0] = t;
      else
        edge_tris[2 * e + 1] = t;
    }
  }

  alloc(vert_tri_count, m_N_vert_s);
  alloc(vert_tris, max_vert_valence * m_N_vert_s);
  for (int s = 0; s < m_N_vert_s; s++) vert_tri_count[s] = 0;
  const int* tv_host = tri_verts.host_ptr();
  for (int t = 0; t < m_N_tri; t++) {
    for (int j = 0; j < 3; j++) {
      int s = tv_host[t * 3 + j];
      int k = vert_tri_count[s];
      if (k >= max_vert_valence) {
        Logger::err(
            "prismatic_mesh_metric: vertex valence exceeds max_vert_valence "
            "(= {}).  Increase the constant and rebuild.",
            max_vert_valence);
      }
      vert_tris[s * max_vert_valence + k] = t;
      vert_tri_count[s] = k + 1;
    }
  }

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  if (mem == MemType::host_device) {
    // Base mesh topology + geometry must be on device before the GPU
    // kernels below can read it (radii, sphere angles, edge endpoints,
    // triangle connectivity, etc.).
    prismatic_mesh::copy_to_device();

    edge_tris.copy_to_device();
    vert_tri_count.copy_to_device();
    vert_tris.copy_to_device();
  }
#endif

  // ----- Per-edge metric eval at edge midpoints (spherical coords) -----
  //
  // θ is re-derived by acos on the stored Cartesian z-component cast to
  // double, rather than via atan2(sin θ, cos θ) of the float-stored
  // sphere_theta.  The atan2 round-trip is unstable for polar vertices:
  // float(acos(-1)) ≈ 3.1415927 is slightly > π in double precision, so
  // sin evaluated in double returns a tiny *negative* value, and atan2 of
  // (-tiny, -1) returns -π instead of +π — which then averages with the
  // first-ring θ to a spurious near-zero midpoint, giving a negative
  // sth_m and a negative edge_sqrt_gamma.
  ExecPolicy::launch(
      [met, Ne = m_N_edges, Nh = (m_N_r + 1) * m_N_edge_s]
      LAMBDA(auto mp, auto er, auto esth, auto ecth, auto ealpha, auto esgb,
             auto esg) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) {
          int v0 = mp.edge_v0[e], v1 = mp.edge_v1[e];
          int k0 = v0 / mp.N_vert_s, k1 = v1 / mp.N_vert_s;
          int s0 = v0 % mp.N_vert_s, s1 = v1 % mp.N_vert_s;
          // Read cos θ as float → cast to double losslessly → clamp to
          // [-1, 1] → acos gives θ ∈ [0, π] with sin(θ) ≥ 0.
          double cth0 = (double)mp.sphere_vz[s0];
          double cth1 = (double)mp.sphere_vz[s1];
          if (cth0 > 1.0) cth0 = 1.0;
          if (cth0 < -1.0) cth0 = -1.0;
          if (cth1 > 1.0) cth1 = 1.0;
          if (cth1 < -1.0) cth1 = -1.0;
          double th0 = math::acos(cth0);
          double th1 = math::acos(cth1);
          double r_m = 0.5 * (mp.radii[k0] + mp.radii[k1]);
          double th_m = 0.5 * (th0 + th1);
          double sth_m = math::sin(th_m);
          double cth_m = math::cos(th_m);
          er[e] = r_m;
          esth[e] = sth_m;
          ecth[e] = cth_m;
          ealpha[e] = met.alpha(r_m, sth_m, cth_m);
          esgb[e] = met.sq_gamma_beta_r(r_m, sth_m, cth_m);
          esg[e] = met.sqrt_gamma(r_m, sth_m, cth_m);
          (void)Nh;
        });
      },
      prismatic_mesh::get_ptrs(typename ExecPolicy::exec_tag{}), edge_r_coord,
      edge_sth, edge_cth, edge_alpha, edge_sq_gamma_beta_r, edge_sqrt_gamma);

  // ----- Per-tri-face metric eval at spherical circumcenter -----
  //
  // The dual 1-cell of a tri face is a radial segment through the tri's
  // spherical CIRCUMCENTER (the dual 0-cell on this shell).  For
  // geometric consistency with the Hodge1_inv / rect-face Hodge2
  // kernels below — which all place dual polygon vertices / dual-edge
  // endpoints at tri circumcenters — sample the pointwise 3+1 scalars
  // here too, not at the centroid.  On non-equilateral triangles the
  // two points differ by O(Δ²), and on an anisotropic metric like KS
  // where g_rr depends on θ the mismatch feeds into the constitutive
  // relations.
  int N_tri_all = (m_N_r + 1) * m_N_tri;
  ExecPolicy::launch(
      [met, N_tri_all]
      LAMBDA(auto mp, auto fr, auto fsth, auto fcth, auto falpha, auto fsgb,
             auto fsg) {
        ExecPolicy::loop(0, N_tri_all, [&] LAMBDA(int fi) {
          int k = fi / mp.N_tri;
          int t = fi % mp.N_tri;
          int a = mp.tri_verts[t * 3 + 0];
          int b = mp.tri_verts[t * 3 + 1];
          int c = mp.tri_verts[t * 3 + 2];
          double th_cc, ph_cc;
          tri_circumcenter_sph(mp, a, b, c, th_cc, ph_cc);
          (void)ph_cc;  // axisymmetric metrics depend only on (r, θ).
          double r = mp.radii[k];
          double sth = math::sin(th_cc);
          double cth = math::cos(th_cc);
          fr[fi] = r;
          fsth[fi] = sth;
          fcth[fi] = cth;
          falpha[fi] = met.alpha(r, sth, cth);
          fsgb[fi] = met.sq_gamma_beta_r(r, sth, cth);
          fsg[fi] = met.sqrt_gamma(r, sth, cth);
        });
      },
      prismatic_mesh::get_ptrs(typename ExecPolicy::exec_tag{}), face_r_coord,
      face_sth, face_cth, face_alpha, face_sq_gamma_beta_r, face_sqrt_gamma);

  // ----- Per-rect-face metric eval at (r_mid, arc-mid) -----
  int N_rect = m_N_r * m_N_edge_s;
  ExecPolicy::launch(
      [met, N_rect, N_tri_all]
      LAMBDA(auto mp, auto fr, auto fsth, auto fcth, auto falpha, auto fsgb,
             auto fsg) {
        ExecPolicy::loop(0, N_rect, [&] LAMBDA(int ri) {
          int fi = N_tri_all + ri;
          int k = ri / mp.N_edge_s;
          int va = mp.rect_face_v0[ri], vb = mp.rect_face_v1[ri];
          double r_m = 0.5 * (mp.radii[k] + mp.radii[k + 1]);
          int sa = va % mp.N_vert_s;
          int sb = vb % mp.N_vert_s;
          double sx = 0.5 * (mp.sphere_vx[sa] + mp.sphere_vx[sb]);
          double sy = 0.5 * (mp.sphere_vy[sa] + mp.sphere_vy[sb]);
          double sz = 0.5 * (mp.sphere_vz[sa] + mp.sphere_vz[sb]);
          double norm = math::sqrt(sx * sx + sy * sy + sz * sz);
          if (norm > 0) {
            sx /= norm;
            sy /= norm;
            sz /= norm;
          }
          double cth = sz;
          double sth = math::sqrt(((0.0) > (1.0 - cth * cth) ? (0.0) : (1.0 - cth * cth)));
          fr[fi] = r_m;
          fsth[fi] = sth;
          fcth[fi] = cth;
          falpha[fi] = met.alpha(r_m, sth, cth);
          fsgb[fi] = met.sq_gamma_beta_r(r_m, sth, cth);
          fsg[fi] = met.sqrt_gamma(r_m, sth, cth);
        });
      },
      prismatic_mesh::get_ptrs(typename ExecPolicy::exec_tag{}), face_r_coord,
      face_sth, face_cth, face_alpha, face_sq_gamma_beta_r, face_sqrt_gamma);

  ExecPolicy::sync();

  int N_tri_local = m_N_tri, N_r_local = m_N_r;

  // ----- Hodge2 on tri faces: |dual edge|_metric / |tri face|_metric -----
  ExecPolicy::launch(
      [met, N_tri_local, N_r_local]
      LAMBDA(auto mp, auto h2_out, auto fa_out) {
        int N_tri_all_ = (N_r_local + 1) * N_tri_local;
        ExecPolicy::loop(0, N_tri_all_, [&] LAMBDA(int fi) {
          int k = fi / N_tri_local;
          int t = fi % N_tri_local;

          int a = mp.tri_verts[t * 3 + 0];
          int b = mp.tri_verts[t * 3 + 1];
          int c = mp.tri_verts[t * 3 + 2];

          double r = mp.radii[k];

          double face_area = shell_triangle_area(
              met, r,
              mp.sphere_vx[a], mp.sphere_vy[a], mp.sphere_vz[a],
              mp.sphere_vx[b], mp.sphere_vy[b], mp.sphere_vz[b],
              mp.sphere_vx[c], mp.sphere_vy[c], mp.sphere_vz[c]);

          // Dual edge: radial segment through the tri CIRCUMCENTER (the
          // dual 0-cell for this shell).  For g_rr depending on θ (KS),
          // evaluating radial_edge_length at the circumcenter θ — not
          // the centroid — keeps the Hodge2 consistent with the rest of
          // the dual-mesh geometry (polygon vertices and rect-face dual
          // endpoints are all at tri circumcenters).
          double theta_f, phi_f;
          tri_circumcenter_sph(mp, a, b, c, theta_f, phi_f);
          (void)phi_f;

          // Boundary shells use the TRUNCATED dual (the half-segment
          // that actually exists), matching the base class — the old
          // x2 "ghost" extension was removed there in the A2 boundary
          // hygiene pass, and the two must agree for the flat metric.
          double dual_len;
          if (k == 0) {
            double r_above = 0.5 * (mp.radii[0] + mp.radii[1]);
            dual_len = radial_edge_length(met, r, r_above, theta_f);
          } else if (k == N_r_local) {
            double r_below =
                0.5 * (mp.radii[N_r_local - 1] + mp.radii[N_r_local]);
            dual_len = radial_edge_length(met, r_below, r, theta_f);
          } else {
            double r_below = 0.5 * (mp.radii[k - 1] + mp.radii[k]);
            double r_above = 0.5 * (mp.radii[k] + mp.radii[k + 1]);
            dual_len = radial_edge_length(met, r_below, r_above, theta_f);
          }
          h2_out[fi] = (Scalar)((face_area > 0) ? dual_len / face_area : 0.0);
          fa_out[fi] = (Scalar)face_area;
        });
      },
      this->get_ptrs(typename ExecPolicy::exec_tag{}), hodge2, face_area);

  // ----- Hodge2 on rect faces -----
  int N_edge_s_local = m_N_edge_s;
  ExecPolicy::launch(
      [met, N_tri_local, N_r_local, N_edge_s_local]
      LAMBDA(auto mp, auto h2_out, auto fa_out) {
        int N_rect_ = N_r_local * N_edge_s_local;
        int N_tri_all_ = (N_r_local + 1) * N_tri_local;
        ExecPolicy::loop(0, N_rect_, [&] LAMBDA(int ri) {
          int k = ri / N_edge_s_local;
          int e_s = ri % N_edge_s_local;
          int t0 = mp.edge_tris[2 * e_s + 0];
          int t1 = mp.edge_tris[2 * e_s + 1];
          int fi = N_tri_all_ + ri;
          if (t0 < 0 || t1 < 0) {
            h2_out[fi] = 0;
            fa_out[fi] = 0;
            return;
          }

          // Face: between shells (k, k+1) along sphere edge (a, b).
          int va = mp.rect_face_v0[ri], vb = mp.rect_face_v1[ri];
          int sa = va % mp.N_vert_s;
          int sb = vb % mp.N_vert_s;
          double r0 = mp.radii[k], r1 = mp.radii[k + 1];

          double face_area = rectangular_face_area(
              met, r0, r1,
              mp.sphere_vx[sa], mp.sphere_vy[sa], mp.sphere_vz[sa],
              mp.sphere_vx[sb], mp.sphere_vy[sb], mp.sphere_vz[sb]);

          // Dual edge: from circumcenter of t0 (on slab k) to t1 (on slab k).
          double r_mid = 0.5 * (r0 + r1);
          // Circumcenter t0 on slab k: same r_mid; angular = sphere
          // barycenter of t0.
          double th0, ph0, th1, ph1;
          {
            int va0 = mp.tri_verts[t0*3+0], vb0 = mp.tri_verts[t0*3+1], vc0 = mp.tri_verts[t0*3+2];
            tri_circumcenter_sph(mp, va0, vb0, vc0, th0, ph0);
            int va1 = mp.tri_verts[t1*3+0], vb1 = mp.tri_verts[t1*3+1], vc1 = mp.tri_verts[t1*3+2];
            tri_circumcenter_sph(mp, va1, vb1, vc1, th1, ph1);
          }
          double c0x = math::sin(th0)*math::cos(ph0);
          double c0y = math::sin(th0)*math::sin(ph0);
          double c0z = math::cos(th0);
          double c1x = math::sin(th1)*math::cos(ph1);
          double c1y = math::sin(th1)*math::sin(ph1);
          double c1z = math::cos(th1);
          double dual_len =
              horizontal_edge_length(met, r_mid, c0x, c0y, c0z, c1x, c1y, c1z);

          h2_out[fi] = (Scalar)((face_area > 0) ? dual_len / face_area : 0.0);
          fa_out[fi] = (Scalar)face_area;
        });
      },
      this->get_ptrs(typename ExecPolicy::exec_tag{}), hodge2, face_area);

  // ----- Hodge1_inv on horizontal edges: primal arc length / dual polygon area
  ExecPolicy::launch(
      [met, N_tri_local, N_r_local, N_edge_s_local]
      LAMBDA(auto mp, auto h1inv_out) {
        int N_h_ = (N_r_local + 1) * N_edge_s_local;
        ExecPolicy::loop(0, N_h_, [&] LAMBDA(int ei) {
          int k = ei / N_edge_s_local;
          int e_s = ei % N_edge_s_local;
          int v0 = mp.edge_v0[ei], v1 = mp.edge_v1[ei];

          double r = mp.radii[k];
          int s0 = v0 % mp.N_vert_s, s1 = v1 % mp.N_vert_s;

          double m_len = horizontal_edge_length(
              met, r,
              mp.sphere_vx[s0], mp.sphere_vy[s0], mp.sphere_vz[s0],
              mp.sphere_vx[s1], mp.sphere_vy[s1], mp.sphere_vz[s1]);

          int t0 = mp.edge_tris[2 * e_s + 0];
          int t1 = mp.edge_tris[2 * e_s + 1];

          // Dual face is always a rectangular face in (r, θ, φ):
          // spans [r_lo, r_hi] radially between the two adjacent triangle
          // circumcenters, where the r-range is the interior dual-cell
          // span r_{k-½}..r_{k+½} clipped to the domain walls at k=0 or
          // k=N_r.  No fan, no polygon — one rectangular_face_area call.
          double th_t0, ph_t0, th_t1, ph_t1;
          {
            int va0 = mp.tri_verts[t0*3+0], vb0 = mp.tri_verts[t0*3+1], vc0 = mp.tri_verts[t0*3+2];
            tri_circumcenter_sph(mp, va0, vb0, vc0, th_t0, ph_t0);
            int va1 = mp.tri_verts[t1*3+0], vb1 = mp.tri_verts[t1*3+1], vc1 = mp.tri_verts[t1*3+2];
            tri_circumcenter_sph(mp, va1, vb1, vc1, th_t1, ph_t1);
          }
          double r_lo = (k > 0) ? 0.5 * (mp.radii[k - 1] + mp.radii[k])
                                : mp.radii[k];
          double r_hi = (k < N_r_local)
                            ? 0.5 * (mp.radii[k] + mp.radii[k + 1])
                            : mp.radii[k];
          double u0x = math::sin(th_t0)*math::cos(ph_t0);
          double u0y = math::sin(th_t0)*math::sin(ph_t0);
          double u0z = math::cos(th_t0);
          double u1x = math::sin(th_t1)*math::cos(ph_t1);
          double u1y = math::sin(th_t1)*math::sin(ph_t1);
          double u1z = math::cos(th_t1);
          double m_area = rectangular_face_area(met, r_lo, r_hi,
                                                u0x, u0y, u0z, u1x, u1y, u1z);

          h1inv_out[ei] = (Scalar)((m_area > 0) ? m_len / m_area : 0.0);
        });
      },
      this->get_ptrs(typename ExecPolicy::exec_tag{}), hodge1_inv);

  // ----- Hodge1_inv on vertical edges: primal radial length / dual polygon area
  int N_vert_s_local = m_N_vert_s;
  int max_valence_local = max_vert_valence;
  ExecPolicy::launch(
      [met, N_tri_local, N_r_local, N_edge_s_local, N_vert_s_local,
       max_valence_local]
      LAMBDA(auto mp, auto h1inv_out) {
        int v_edge_off = (N_r_local + 1) * N_edge_s_local;
        int N_v_ = N_r_local * N_vert_s_local;
        ExecPolicy::loop(0, N_v_, [&] LAMBDA(int vi) {
          int k = vi / N_vert_s_local;
          int s = vi % N_vert_s_local;
          int ei = v_edge_off + k * N_vert_s_local + s;

          double r0 = mp.radii[k], r1 = mp.radii[k + 1];
          // Derive θ in double from cos θ (see vertex_sph comment) to
          // avoid the float-π precision loss at polar vertices.
          double cth_v = (double)mp.sphere_vz[s];
          if (cth_v > 1.0) cth_v = 1.0;
          if (cth_v < -1.0) cth_v = -1.0;
          double th_v = math::acos(cth_v);
          double ph_v = mp.sphere_phi[s];
          double m_len = radial_edge_length(met, r0, r1, th_v);

          int np = mp.vert_tri_count[s];
          double rr[6], th[6], ph[6], angles[6];
          double r_mid = 0.5 * (r0 + r1);
          for (int i = 0; i < np; i++) {
            int t = mp.vert_tris[s * max_valence_local + i];
            int va = mp.tri_verts[t * 3 + 0];
            int vb = mp.tri_verts[t * 3 + 1];
            int vc = mp.tri_verts[t * 3 + 2];
            double th_cc, ph_cc;
            tri_circumcenter_sph(mp, va, vb, vc, th_cc, ph_cc);
            rr[i] = r_mid;
            th[i] = th_cc;
            ph[i] = ph_cc;
            // Angle around the vertex for sort.
            angles[i] =
                math::atan2(th[i] - th_v, angular_diff(ph_v, ph[i]));
          }
          // Selection sort by angle.
          int order[6];
          for (int i = 0; i < np; i++) order[i] = i;
          for (int i = 0; i < np - 1; i++) {
            int mi = i;
            for (int j = i + 1; j < np; j++) {
              if (angles[order[j]] < angles[order[mi]]) mi = j;
            }
            int tmp = order[i];
            order[i] = order[mi];
            order[mi] = tmp;
          }
          double srr[6], sth[6], sph[6];
          for (int i = 0; i < np; i++) {
            srr[i] = rr[order[i]];
            sth[i] = th[order[i]];
            sph[i] = ph[order[i]];
          }
          // All polygon vertices and the fan center lie at r_mid, so this
          // is a pure shell polygon — use the angular-only helper.
          (void)srr;  // radii are all r_mid; kept only for angle-sort input.
          double m_area = shell_polygon_area_about(met, r_mid, th_v, ph_v, sth,
                                                   sph, np);
          h1inv_out[ei] = (Scalar)((m_area > 0) ? m_len / m_area : 0.0);
        });
      },
      this->get_ptrs(typename ExecPolicy::exec_tag{}), hodge1_inv);

  ExecPolicy::sync();

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  if (mem == MemType::host_device) {
    copy_metric_to_host();
  }
#endif

  Logger::print_info("prismatic_mesh_metric::compute_metric done (spherical)");
}

}  // namespace Aperture
