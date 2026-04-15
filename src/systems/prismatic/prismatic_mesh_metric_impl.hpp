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
template <typename Mesh>
HD_INLINE void vertex_sph(const Mesh& mp, int vidx, double& r, double& sth,
                          double& cth, double& phi) {
  int k = vidx / mp.N_vert_s;
  int sphv = vidx % mp.N_vert_s;
  r = mp.radii[k];
  cth = mp.sphere_vz[sphv];
  double s2 = 1.0 - cth * cth;
  sth = math::sqrt(s2 > 0 ? s2 : 0.0);
  phi = math::atan2((double)mp.sphere_vy[sphv], (double)mp.sphere_vx[sphv]);
}

// Horizontal edge length (on shell r, coord-linear in θ, φ).
template <typename Metric>
HD_INLINE double
horizontal_edge_length(const Metric& met, double r, double th_a, double ph_a,
                       double th_b, double ph_b) {
  double dth = th_b - th_a;
  double dph = angular_diff(ph_a, ph_b);
  return gauss_quad(
      [&](double s) {
        double th = th_a + s * dth;
        double sth = math::sin(th), cth = math::cos(th);
        double g22 = met.g_thth(r, sth, cth);
        double g33 = met.g_phph(r, sth, cth);
        double g23 = met.g_thph(r, sth, cth);
        double q = g22 * dth * dth + g33 * dph * dph + 2.0 * g23 * dth * dph;
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
// The patch stays on the sphere of radius r, so only the angular sub-block
// of the metric (g_θθ, g_φφ, g_θφ) contributes.  Specialized to avoid the
// wasted multiplies by zero that the generic coord_triangle_area incurs
// when all Δr vanish, and to make the contract explicit (no dependence on
// g_rr or g_rφ).
template <typename Metric>
HD_INLINE double
shell_triangle_area(const Metric& met, double r, double th_a, double ph_a,
                    double th_b, double ph_b, double th_c, double ph_c) {
  double dthB = th_b - th_a, dphB = angular_diff(ph_a, ph_b);
  double dthC = th_c - th_a, dphC = angular_diff(ph_a, ph_c);
  return gauss_quad(
      [&](double u) {
        return gauss_quad(
            [&](double v) {
              double th = th_a + u * dthB + v * dthC;
              double sth = math::sin(th), cth = math::cos(th);
              double g22 = met.g_thth(r, sth, cth);
              double g33 = met.g_phph(r, sth, cth);
              double g23 = met.g_thph(r, sth, cth);
              double huu = g22 * dthB * dthB + g33 * dphB * dphB +
                           2.0 * g23 * dthB * dphB;
              double hvv = g22 * dthC * dthC + g33 * dphC * dphC +
                           2.0 * g23 * dthC * dphC;
              double huv = g22 * dthB * dthC + g33 * dphB * dphC +
                           g23 * (dthB * dphC + dthC * dphB);
              double det = huu * hvv - huv * huv;
              return math::sqrt(det > 0 ? det : 0.0);
            },
            0.0, 1.0 - u);
      },
      0.0, 1.0);
}

// Rectangular face area bounded by arcs at r_0 and r_1 with the same angular
// endpoints (θ_a, φ_a) and (θ_b, φ_b).
template <typename Metric>
HD_INLINE double
rectangular_face_area(const Metric& met, double r_0, double r_1, double th_a,
                      double ph_a, double th_b, double ph_b) {
  double dr = r_1 - r_0;
  double dth = th_b - th_a;
  double dph = angular_diff(ph_a, ph_b);
  return gauss_quad(
      [&](double u) {
        double th = th_a + u * dth;
        double sth = math::sin(th), cth = math::cos(th);
        return gauss_quad(
            [&](double v) {
              double r = r_0 + v * dr;
              double g11 = met.g_rr(r, sth, cth);
              double g22 = met.g_thth(r, sth, cth);
              double g33 = met.g_phph(r, sth, cth);
              double g13 = met.g_rph(r, sth, cth);
              double g23 = met.g_thph(r, sth, cth);
              // ∂/∂u = (0, dth, dph); ∂/∂v = (dr, 0, 0).
              double huu = g22 * dth * dth + g33 * dph * dph +
                           2.0 * g23 * dth * dph;
              double hvv = g11 * dr * dr;
              double huv = g13 * dr * dph;
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
  double area = 0.0;
  for (int i = 0; i < n; i++) {
    int j = (i + 1) % n;
    area += shell_triangle_area(met, r, th_c, ph_c, th[i], ph[i],
                                th[j], ph[j]);
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

  int N_prisms = m_N_tri * m_N_r;
  alloc(cc_x, N_prisms);
  alloc(cc_y, N_prisms);
  alloc(cc_z, N_prisms);

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
    edge_tris.copy_to_device();
    vert_tri_count.copy_to_device();
    vert_tris.copy_to_device();
  }
#endif

  // ----- Per-edge metric eval at edge midpoints (spherical coords) -----
  ExecPolicy::launch(
      [met, Ne = m_N_edges, Nh = (m_N_r + 1) * m_N_edge_s]
      LAMBDA(auto mp, auto er, auto esth, auto ecth, auto ealpha, auto esgb,
             auto esg) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) {
          int v0 = mp.edge_v0[e], v1 = mp.edge_v1[e];
          double r0, sth0, cth0, ph0;
          double r1, sth1, cth1, ph1;
          vertex_sph(mp, v0, r0, sth0, cth0, ph0);
          vertex_sph(mp, v1, r1, sth1, cth1, ph1);
          double r_m = 0.5 * (r0 + r1);
          double th0 = math::atan2(sth0, cth0);
          double th1 = math::atan2(sth1, cth1);
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

  // ----- Per-tri-face metric eval at spherical centroid -----
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
          double sx = (mp.sphere_vx[a] + mp.sphere_vx[b] + mp.sphere_vx[c]) /
                      3.0;
          double sy = (mp.sphere_vy[a] + mp.sphere_vy[b] + mp.sphere_vy[c]) /
                      3.0;
          double sz = (mp.sphere_vz[a] + mp.sphere_vz[b] + mp.sphere_vz[c]) /
                      3.0;
          double norm = math::sqrt(sx * sx + sy * sy + sz * sz);
          if (norm > 0) {
            sx /= norm;
            sy /= norm;
            sz /= norm;
          }
          double r = mp.radii[k];
          double cth = sz;
          double sth = math::sqrt(((0.0) > (1.0 - cth * cth) ? (0.0) : (1.0 - cth * cth)));
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

  // ----- Circumcenter kernel (angular barycenter on the sphere at r_mid) --
  int N_tri_local = m_N_tri, N_r_local = m_N_r;
  ExecPolicy::launch(
      [N_tri_local, N_r_local]
      LAMBDA(auto mp, auto ccx_out, auto ccy_out, auto ccz_out) {
        ExecPolicy::loop(0, N_tri_local * N_r_local, [&] LAMBDA(int pid) {
          int k = pid / N_tri_local;
          int t = pid % N_tri_local;
          double r_mid = 0.5 * (mp.radii[k] + mp.radii[k + 1]);

          int a = mp.tri_verts[t * 3 + 0];
          int b = mp.tri_verts[t * 3 + 1];
          int c = mp.tri_verts[t * 3 + 2];
          double sx = mp.sphere_vx[a] + mp.sphere_vx[b] + mp.sphere_vx[c];
          double sy = mp.sphere_vy[a] + mp.sphere_vy[b] + mp.sphere_vy[c];
          double sz = mp.sphere_vz[a] + mp.sphere_vz[b] + mp.sphere_vz[c];
          double norm = math::sqrt(sx * sx + sy * sy + sz * sz);
          if (norm > 0) {
            sx /= norm;
            sy /= norm;
            sz /= norm;
          }
          ccx_out[pid] = (Scalar)(r_mid * sx);
          ccy_out[pid] = (Scalar)(r_mid * sy);
          ccz_out[pid] = (Scalar)(r_mid * sz);
        });
      },
      prismatic_mesh::get_ptrs(typename ExecPolicy::exec_tag{}), cc_x, cc_y,
      cc_z);

  ExecPolicy::sync();

  // ----- Hodge2 on tri faces: |dual edge|_metric / |tri face|_metric -----
  ExecPolicy::launch(
      [met, N_tri_local, N_r_local]
      LAMBDA(auto mp, auto h2_out) {
        int N_tri_all_ = (N_r_local + 1) * N_tri_local;
        ExecPolicy::loop(0, N_tri_all_, [&] LAMBDA(int fi) {
          int k = fi / N_tri_local;
          int t = fi % N_tri_local;

          int a = mp.tri_verts[t * 3 + 0];
          int b = mp.tri_verts[t * 3 + 1];
          int c = mp.tri_verts[t * 3 + 2];

          double r = mp.radii[k];
          double cth_a = mp.sphere_vz[a];
          double th_a = math::acos(cth_a);
          double ph_a = math::atan2((double)mp.sphere_vy[a],
                                    (double)mp.sphere_vx[a]);
          double th_b = math::acos((double)mp.sphere_vz[b]);
          double ph_b = math::atan2((double)mp.sphere_vy[b],
                                    (double)mp.sphere_vx[b]);
          double th_c = math::acos((double)mp.sphere_vz[c]);
          double ph_c = math::atan2((double)mp.sphere_vy[c],
                                    (double)mp.sphere_vx[c]);

          double face_area =
              shell_triangle_area(met, r, th_a, ph_a, th_b, ph_b, th_c, ph_c);

          // Dual edge (radial segment through the face centroid).  Use the
          // sphere-centroid (angular average, renormalized) as the θ_center.
          double sx = mp.sphere_vx[a] + mp.sphere_vx[b] + mp.sphere_vx[c];
          double sy = mp.sphere_vy[a] + mp.sphere_vy[b] + mp.sphere_vy[c];
          double sz = mp.sphere_vz[a] + mp.sphere_vz[b] + mp.sphere_vz[c];
          double norm = math::sqrt(sx * sx + sy * sy + sz * sz);
          if (norm > 0) {
            sx /= norm;
            sy /= norm;
            sz /= norm;
          }
          double cth_f = sz;
          double theta_f = math::acos(cth_f);

          double dual_len;
          if (k == 0) {
            double r_above = 0.5 * (mp.radii[0] + mp.radii[1]);
            dual_len = 2.0 * radial_edge_length(met, r, r_above, theta_f);
          } else if (k == N_r_local) {
            double r_below =
                0.5 * (mp.radii[N_r_local - 1] + mp.radii[N_r_local]);
            dual_len = 2.0 * radial_edge_length(met, r_below, r, theta_f);
          } else {
            double r_below = 0.5 * (mp.radii[k - 1] + mp.radii[k]);
            double r_above = 0.5 * (mp.radii[k] + mp.radii[k + 1]);
            dual_len = radial_edge_length(met, r_below, r_above, theta_f);
          }
          h2_out[fi] = (Scalar)((face_area > 0) ? dual_len / face_area : 0.0);
        });
      },
      this->get_ptrs(typename ExecPolicy::exec_tag{}), hodge2);

  // ----- Hodge2 on rect faces -----
  int N_edge_s_local = m_N_edge_s;
  ExecPolicy::launch(
      [met, N_tri_local, N_r_local, N_edge_s_local]
      LAMBDA(auto mp, auto h2_out) {
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
            return;
          }

          // Face: between shells (k, k+1) along sphere edge (a, b).
          int va = mp.rect_face_v0[ri], vb = mp.rect_face_v1[ri];
          int sa = va % mp.N_vert_s;
          int sb = vb % mp.N_vert_s;
          double r0 = mp.radii[k], r1 = mp.radii[k + 1];
          double th_a = math::acos((double)mp.sphere_vz[sa]);
          double ph_a = math::atan2((double)mp.sphere_vy[sa],
                                    (double)mp.sphere_vx[sa]);
          double th_b = math::acos((double)mp.sphere_vz[sb]);
          double ph_b = math::atan2((double)mp.sphere_vy[sb],
                                    (double)mp.sphere_vx[sb]);

          double face_area = rectangular_face_area(met, r0, r1, th_a, ph_a,
                                                   th_b, ph_b);

          // Dual edge: from circumcenter of t0 (on slab k) to t1 (on slab k).
          double r_mid = 0.5 * (r0 + r1);
          // Circumcenter t0 on slab k: same r_mid; angular = sphere
          // barycenter of t0.
          auto tri_center_theta_phi =
              [&](int t, double& th, double& ph) {
                int va_ = mp.tri_verts[t * 3 + 0];
                int vb_ = mp.tri_verts[t * 3 + 1];
                int vc_ = mp.tri_verts[t * 3 + 2];
                double sx =
                    mp.sphere_vx[va_] + mp.sphere_vx[vb_] + mp.sphere_vx[vc_];
                double sy =
                    mp.sphere_vy[va_] + mp.sphere_vy[vb_] + mp.sphere_vy[vc_];
                double sz =
                    mp.sphere_vz[va_] + mp.sphere_vz[vb_] + mp.sphere_vz[vc_];
                double nrm = math::sqrt(sx * sx + sy * sy + sz * sz);
                if (nrm > 0) {
                  sx /= nrm;
                  sy /= nrm;
                  sz /= nrm;
                }
                th = math::acos(sz);
                ph = math::atan2(sy, sx);
              };
          double th0, ph0, th1, ph1;
          tri_center_theta_phi(t0, th0, ph0);
          tri_center_theta_phi(t1, th1, ph1);
          double dual_len =
              horizontal_edge_length(met, r_mid, th0, ph0, th1, ph1);

          h2_out[fi] = (Scalar)((face_area > 0) ? dual_len / face_area : 0.0);
        });
      },
      this->get_ptrs(typename ExecPolicy::exec_tag{}), hodge2);

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
          double th0 = math::acos((double)mp.sphere_vz[s0]);
          double ph0 = math::atan2((double)mp.sphere_vy[s0],
                                   (double)mp.sphere_vx[s0]);
          double th1 = math::acos((double)mp.sphere_vz[s1]);
          double ph1 = math::atan2((double)mp.sphere_vy[s1],
                                   (double)mp.sphere_vx[s1]);

          double m_len = horizontal_edge_length(met, r, th0, ph0, th1, ph1);

          int t0 = mp.edge_tris[2 * e_s + 0];
          int t1 = mp.edge_tris[2 * e_s + 1];

          // Up to 4 dual polygon vertices (circumcenters), at r_{k±½}.
          auto tri_center_th_ph =
              [&](int t, double& th, double& ph) {
                int va = mp.tri_verts[t * 3 + 0];
                int vb = mp.tri_verts[t * 3 + 1];
                int vc = mp.tri_verts[t * 3 + 2];
                double sx =
                    mp.sphere_vx[va] + mp.sphere_vx[vb] + mp.sphere_vx[vc];
                double sy =
                    mp.sphere_vy[va] + mp.sphere_vy[vb] + mp.sphere_vy[vc];
                double sz =
                    mp.sphere_vz[va] + mp.sphere_vz[vb] + mp.sphere_vz[vc];
                double nrm = math::sqrt(sx * sx + sy * sy + sz * sz);
                if (nrm > 0) {
                  sx /= nrm;
                  sy /= nrm;
                  sz /= nrm;
                }
                th = math::acos(sz);
                ph = math::atan2(sy, sx);
              };

          double rr[4], th[4], ph[4];
          int np = 0;
          if (k > 0 && t0 >= 0) {
            rr[np] = 0.5 * (mp.radii[k - 1] + mp.radii[k]);
            tri_center_th_ph(t0, th[np], ph[np]);
            np++;
          }
          if (k > 0 && t1 >= 0) {
            rr[np] = 0.5 * (mp.radii[k - 1] + mp.radii[k]);
            tri_center_th_ph(t1, th[np], ph[np]);
            np++;
          }
          if (k < N_r_local && t1 >= 0) {
            rr[np] = 0.5 * (mp.radii[k] + mp.radii[k + 1]);
            tri_center_th_ph(t1, th[np], ph[np]);
            np++;
          }
          if (k < N_r_local && t0 >= 0) {
            rr[np] = 0.5 * (mp.radii[k] + mp.radii[k + 1]);
            tri_center_th_ph(t0, th[np], ph[np]);
            np++;
          }

          // Fan center: primal edge midpoint (at r_k, angular average of
          // edge endpoints).
          double th_mid = 0.5 * (th0 + th1);
          double ph_mid =
              ph0 + 0.5 * angular_diff(ph0, ph1);
          double m_area = polygon_area_about(met, r, th_mid, ph_mid, rr, th,
                                             ph, np);

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
          double th_v = math::acos((double)mp.sphere_vz[s]);
          double ph_v = math::atan2((double)mp.sphere_vy[s],
                                    (double)mp.sphere_vx[s]);
          double m_len = radial_edge_length(met, r0, r1, th_v);

          int np = mp.vert_tri_count[s];
          double rr[6], th[6], ph[6], angles[6];
          double r_mid = 0.5 * (r0 + r1);
          for (int i = 0; i < np; i++) {
            int t = mp.vert_tris[s * max_valence_local + i];
            int va = mp.tri_verts[t * 3 + 0];
            int vb = mp.tri_verts[t * 3 + 1];
            int vc = mp.tri_verts[t * 3 + 2];
            double sx =
                mp.sphere_vx[va] + mp.sphere_vx[vb] + mp.sphere_vx[vc];
            double sy =
                mp.sphere_vy[va] + mp.sphere_vy[vb] + mp.sphere_vy[vc];
            double sz =
                mp.sphere_vz[va] + mp.sphere_vz[vb] + mp.sphere_vz[vc];
            double nrm = math::sqrt(sx * sx + sy * sy + sz * sz);
            if (nrm > 0) {
              sx /= nrm;
              sy /= nrm;
              sz /= nrm;
            }
            rr[i] = r_mid;
            th[i] = math::acos(sz);
            ph[i] = math::atan2(sy, sx);
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
