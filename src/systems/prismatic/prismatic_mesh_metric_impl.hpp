#pragma once

#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_mesh_metric.h"
#include "utils/gauss_quadrature.h"
#include "utils/logger.h"

namespace Aperture {

namespace prismatic_metric_helpers {

// -------------------------------------------------------------------------
// HD_INLINE device-callable helpers.  All use double precision
// internally; only the final stored per-element buffers are Scalar.
// -------------------------------------------------------------------------

HD_INLINE void cart_to_sph(double x, double y, double z, double& r,
                           double& sth, double& cth, double& phi) {
  r = math::sqrt(x * x + y * y + z * z);
  cth = (r > 0) ? z / r : 1.0;
  double s2 = 1.0 - cth * cth;
  sth = math::sqrt(s2 > 0 ? s2 : 0.0);
  phi = math::atan2(y, x);
}

// Metric inner product γ_{ij} V^i W^j at Cartesian point P.
template <typename Metric>
HD_INLINE double
metric_inner_product(const Metric& met, double x, double y, double z,
                     double vx, double vy, double vz, double wx, double wy,
                     double wz) {
  double r, sth, cth, phi;
  cart_to_sph(x, y, z, r, sth, cth, phi);
  if (r < 1e-30) return vx * wx + vy * wy + vz * wz;

  double cph = math::cos(phi), sph = math::sin(phi);

  double v_r = sth * cph * vx + sth * sph * vy + cth * vz;
  double v_th = (cth * cph * vx + cth * sph * vy - sth * vz) / r;
  double v_ph = (sth > 1e-12) ? (-sph * vx + cph * vy) / (r * sth) : 0.0;

  double w_r = sth * cph * wx + sth * sph * wy + cth * wz;
  double w_th = (cth * cph * wx + cth * sph * wy - sth * wz) / r;
  double w_ph = (sth > 1e-12) ? (-sph * wx + cph * wy) / (r * sth) : 0.0;

  double g11 = met.g_rr(r, sth, cth);
  double g22 = met.g_thth(r, sth, cth);
  double g33 = met.g_phph(r, sth, cth);
  double g13 = met.g_rph(r, sth, cth);

  return g11 * v_r * w_r + g22 * v_th * w_th + g33 * v_ph * w_ph +
         g13 * (v_r * w_ph + v_ph * w_r);
}

template <typename Metric>
HD_INLINE double
segment_length(const Metric& met, double x0, double y0, double z0, double x1,
               double y1, double z1) {
  double vx = x1 - x0, vy = y1 - y0, vz = z1 - z0;
  return gauss_quad(
      [&](double t) {
        double x = x0 + t * vx, y = y0 + t * vy, z = z0 + t * vz;
        double ds2 =
            metric_inner_product(met, x, y, z, vx, vy, vz, vx, vy, vz);
        return math::sqrt(ds2 > 0 ? ds2 : 0.0);
      },
      0.0, 1.0);
}

template <typename Metric>
HD_INLINE double
arc_length(const Metric& met, double x0, double y0, double z0, double x1,
           double y1, double z1) {
  double r = math::sqrt(x0 * x0 + y0 * y0 + z0 * z0);
  double dot = (x0 * x1 + y0 * y1 + z0 * z1) / (r * r);
  if (dot > 1.0) dot = 1.0;
  if (dot < -1.0) dot = -1.0;
  double Omega = math::acos(dot);
  if (Omega < 1e-12) return 0.0;

  double sO = math::sin(Omega);
  double sax = x0 / r, say = y0 / r, saz = z0 / r;
  double sbx = x1 / r, sby = y1 / r, sbz = z1 / r;
  double etx = (sbx - dot * sax) / sO;
  double ety = (sby - dot * say) / sO;
  double etz = (sbz - dot * saz) / sO;

  return gauss_quad(
      [&](double t) {
        double s = t * Omega;
        double cs = math::cos(s), ss = math::sin(s);
        double x = r * (cs * sax + ss * etx);
        double y = r * (cs * say + ss * ety);
        double z = r * (cs * saz + ss * etz);
        double tx = r * Omega * (-ss * sax + cs * etx);
        double ty = r * Omega * (-ss * say + cs * ety);
        double tz = r * Omega * (-ss * saz + cs * etz);
        double ds2 =
            metric_inner_product(met, x, y, z, tx, ty, tz, tx, ty, tz);
        return math::sqrt(ds2 > 0 ? ds2 : 0.0);
      },
      0.0, 1.0);
}

template <typename Metric>
HD_INLINE double
triangle_area(const Metric& met, double x0, double y0, double z0, double x1,
              double y1, double z1, double x2, double y2, double z2) {
  double e1x = x1 - x0, e1y = y1 - y0, e1z = z1 - z0;
  double e2x = x2 - x0, e2y = y2 - y0, e2z = z2 - z0;

  return gauss_quad(
      [&](double u) {
        return gauss_quad(
            [&](double v) {
              double x = x0 + u * e1x + v * e2x;
              double y = y0 + u * e1y + v * e2y;
              double z = z0 + u * e1z + v * e2z;
              double h11 = metric_inner_product(met, x, y, z, e1x, e1y, e1z,
                                                e1x, e1y, e1z);
              double h12 = metric_inner_product(met, x, y, z, e1x, e1y, e1z,
                                                e2x, e2y, e2z);
              double h22 = metric_inner_product(met, x, y, z, e2x, e2y, e2z,
                                                e2x, e2y, e2z);
              double det = h11 * h22 - h12 * h12;
              return math::sqrt(det > 0 ? det : 0.0);
            },
            0.0, 1.0 - u);
      },
      0.0, 1.0);
}

template <typename Metric>
HD_INLINE double
rect_face_area(const Metric& met, double x0, double y0, double z0, double x1,
               double y1, double z1, double /*x2*/, double /*y2*/,
               double /*z2*/, double x3, double y3, double z3) {
  double r0 = math::sqrt(x0 * x0 + y0 * y0 + z0 * z0);
  double r1 = math::sqrt(x3 * x3 + y3 * y3 + z3 * z3);
  double sax = x0 / r0, say = y0 / r0, saz = z0 / r0;
  double sbx = x1 / r0, sby = y1 / r0, sbz = z1 / r0;
  double dot = sax * sbx + say * sby + saz * sbz;
  if (dot > 1.0) dot = 1.0;
  if (dot < -1.0) dot = -1.0;
  double Omega = math::acos(dot);
  if (Omega < 1e-12 || r1 - r0 < 1e-30) return 0.0;

  double sO = math::sin(Omega);
  double etx = (sbx - dot * sax) / sO;
  double ety = (sby - dot * say) / sO;
  double etz = (sbz - dot * saz) / sO;

  return gauss_quad(
      [&](double u) {
        double cs = math::cos(u), ss = math::sin(u);
        double ux = cs * sax + ss * etx;
        double uy = cs * say + ss * ety;
        double uz = cs * saz + ss * etz;
        double tax = -ss * sax + cs * etx;
        double tay = -ss * say + cs * ety;
        double taz = -ss * saz + cs * etz;

        return gauss_quad(
            [&](double v) {
              double x = v * ux, y = v * uy, z = v * uz;
              double du_x = v * tax, du_y = v * tay, du_z = v * taz;
              double dv_x = ux, dv_y = uy, dv_z = uz;
              double h11 = metric_inner_product(met, x, y, z, du_x, du_y,
                                                du_z, du_x, du_y, du_z);
              double h12 = metric_inner_product(met, x, y, z, du_x, du_y,
                                                du_z, dv_x, dv_y, dv_z);
              double h22 = metric_inner_product(met, x, y, z, dv_x, dv_y,
                                                dv_z, dv_x, dv_y, dv_z);
              double det = h11 * h22 - h12 * h12;
              return math::sqrt(det > 0 ? det : 0.0);
            },
            r0, r1);
      },
      0.0, Omega);
}

// Polygon area with explicit fan center, fixed-size array input.
template <typename Metric>
HD_INLINE double
polygon_area_about_fixed(const Metric& met, double cx, double cy, double cz,
                         const double* px, const double* py, const double* pz,
                         int np) {
  if (np < 3) return 0.0;
  double area = 0.0;
  for (int i = 0; i < np; i++) {
    int j = (i + 1) % np;
    area += triangle_area(met, cx, cy, cz, px[i], py[i], pz[i], px[j], py[j],
                          pz[j]);
  }
  return area;
}

// Polygon area with fan from vertex average.
template <typename Metric>
HD_INLINE double
polygon_area_fixed(const Metric& met, const double* px, const double* py,
                   const double* pz, int np) {
  if (np < 3) return 0.0;
  double mx = 0, my = 0, mz = 0;
  for (int i = 0; i < np; i++) {
    mx += px[i];
    my += py[i];
    mz += pz[i];
  }
  mx /= np;
  my /= np;
  mz /= np;
  return polygon_area_about_fixed(met, mx, my, mz, px, py, pz, np);
}

}  // namespace prismatic_metric_helpers

// =========================================================================
// Template compute_metric<Metric>
//
// Entry point.  Must be called after build() and, on GPU builds, after
// copy_to_device().  Fills all per-element metric buffers + Hodge stars,
// with kernels running on the execution policy dispatched at compile
// time (prismatic_exec_policy_dynamic = GPU when available, otherwise
// host).  On GPU builds the updated hodge1_inv / hodge2 buffers are
// also copied back to host so the data exporter can write them.
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
  // (cheap topological scan, just indices)
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

  // ----- Launch per-edge metric eval -----
  ExecPolicy::launch(
      [met, Ne = m_N_edges]
      LAMBDA(auto mp, auto er, auto esth, auto ecth, auto ealpha, auto esgb,
             auto esg) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) {
          int v0 = mp.edge_v0[e], v1 = mp.edge_v1[e];
          double mx = 0.5 * (mp.vert_x[v0] + mp.vert_x[v1]);
          double my = 0.5 * (mp.vert_y[v0] + mp.vert_y[v1]);
          double mz = 0.5 * (mp.vert_z[v0] + mp.vert_z[v1]);
          double r, sth, cth, phi;
          cart_to_sph(mx, my, mz, r, sth, cth, phi);
          er[e] = r;
          esth[e] = sth;
          ecth[e] = cth;
          ealpha[e] = met.alpha(r, sth, cth);
          esgb[e] = met.sq_gamma_beta_r(r, sth, cth);
          esg[e] = met.sqrt_gamma(r, sth, cth);
        });
      },
      prismatic_mesh::get_ptrs(typename ExecPolicy::exec_tag{}), edge_r_coord,
      edge_sth, edge_cth, edge_alpha, edge_sq_gamma_beta_r, edge_sqrt_gamma);

  // ----- Launch per-tri-face metric eval -----
  int N_tri_all = (m_N_r + 1) * m_N_tri;
  ExecPolicy::launch(
      [met, N_tri_all]
      LAMBDA(auto mp, auto fr, auto fsth, auto fcth, auto falpha, auto fsgb,
             auto fsg) {
        ExecPolicy::loop(0, N_tri_all, [&] LAMBDA(int fi) {
          int va = mp.tri_face_v0[fi], vb = mp.tri_face_v1[fi],
              vc = mp.tri_face_v2[fi];
          double cx_ =
              (mp.vert_x[va] + mp.vert_x[vb] + mp.vert_x[vc]) / 3.0;
          double cy_ =
              (mp.vert_y[va] + mp.vert_y[vb] + mp.vert_y[vc]) / 3.0;
          double cz_ =
              (mp.vert_z[va] + mp.vert_z[vb] + mp.vert_z[vc]) / 3.0;
          double r, sth, cth, phi;
          cart_to_sph(cx_, cy_, cz_, r, sth, cth, phi);
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

  // ----- Launch per-rect-face metric eval -----
  int N_rect = m_N_r * m_N_edge_s;
  ExecPolicy::launch(
      [met, N_rect, N_tri_all]
      LAMBDA(auto mp, auto fr, auto fsth, auto fcth, auto falpha, auto fsgb,
             auto fsg) {
        ExecPolicy::loop(0, N_rect, [&] LAMBDA(int ri) {
          int fi = N_tri_all + ri;
          int va = mp.rect_face_v0[ri], vb = mp.rect_face_v1[ri];
          int vc = mp.rect_face_v2[ri], vd = mp.rect_face_v3[ri];
          double cx_ = 0.25 * (mp.vert_x[va] + mp.vert_x[vb] +
                               mp.vert_x[vc] + mp.vert_x[vd]);
          double cy_ = 0.25 * (mp.vert_y[va] + mp.vert_y[vb] +
                               mp.vert_y[vc] + mp.vert_y[vd]);
          double cz_ = 0.25 * (mp.vert_z[va] + mp.vert_z[vb] +
                               mp.vert_z[vc] + mp.vert_z[vd]);
          double r, sth, cth, phi;
          cart_to_sph(cx_, cy_, cz_, r, sth, cth, phi);
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

  // ----- Launch circumcenter kernel -----
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

          double ux = 0, uy = 0, uz = 0;
          if (nlen > 0) {
            ux = nx / nlen;
            uy = ny / nlen;
            uz = nz / nlen;
            double mx_ =
                (mp.sphere_vx[a] + mp.sphere_vx[b] + mp.sphere_vx[c]) / 3.0;
            double my_ =
                (mp.sphere_vy[a] + mp.sphere_vy[b] + mp.sphere_vy[c]) / 3.0;
            double mz_ =
                (mp.sphere_vz[a] + mp.sphere_vz[b] + mp.sphere_vz[c]) / 3.0;
            if (ux * mx_ + uy * my_ + uz * mz_ < 0) {
              ux = -ux;
              uy = -uy;
              uz = -uz;
            }
          }
          ccx_out[pid] = r_mid * ux;
          ccy_out[pid] = r_mid * uy;
          ccz_out[pid] = r_mid * uz;
        });
      },
      prismatic_mesh::get_ptrs(typename ExecPolicy::exec_tag{}), cc_x, cc_y,
      cc_z);

  ExecPolicy::sync();

  // ----- Hodge stars -----
  // hodge2[tri f on shell k]: metric dual-edge length (radial through f)
  //                            divided by metric face area.
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
          int va = k * mp.N_vert_s + a;
          int vb = k * mp.N_vert_s + b;
          int vc = k * mp.N_vert_s + c;

          double m_area = triangle_area(
              met, mp.vert_x[va], mp.vert_y[va], mp.vert_z[va], mp.vert_x[vb],
              mp.vert_y[vb], mp.vert_z[vb], mp.vert_x[vc], mp.vert_y[vc],
              mp.vert_z[vc]);

          double dist = 0;
          if (k == 0) {
            int pid = 0 * N_tri_local + t;
            double fx =
                (mp.vert_x[va] + mp.vert_x[vb] + mp.vert_x[vc]) / 3.0;
            double fy =
                (mp.vert_y[va] + mp.vert_y[vb] + mp.vert_y[vc]) / 3.0;
            double fz =
                (mp.vert_z[va] + mp.vert_z[vb] + mp.vert_z[vc]) / 3.0;
            dist = segment_length(met, fx, fy, fz, mp.cc_x[pid],
                                  mp.cc_y[pid], mp.cc_z[pid]) *
                   2.0;
          } else if (k == N_r_local) {
            int pid = (N_r_local - 1) * N_tri_local + t;
            double fx =
                (mp.vert_x[va] + mp.vert_x[vb] + mp.vert_x[vc]) / 3.0;
            double fy =
                (mp.vert_y[va] + mp.vert_y[vb] + mp.vert_y[vc]) / 3.0;
            double fz =
                (mp.vert_z[va] + mp.vert_z[vb] + mp.vert_z[vc]) / 3.0;
            dist = segment_length(met, fx, fy, fz, mp.cc_x[pid],
                                  mp.cc_y[pid], mp.cc_z[pid]) *
                   2.0;
          } else {
            int pb = (k - 1) * N_tri_local + t;
            int pa = k * N_tri_local + t;
            dist = segment_length(met, mp.cc_x[pb], mp.cc_y[pb], mp.cc_z[pb],
                                  mp.cc_x[pa], mp.cc_y[pa], mp.cc_z[pa]);
          }
          h2_out[fi] = (m_area > 0) ? dist / m_area : 0;
        });
      },
      this->get_ptrs(typename ExecPolicy::exec_tag{}), hodge2);

  // hodge2[rect f in slab k]: metric dual-edge (connects circumcenters
  //                           of two adjacent prisms) / metric face area.
  int N_edge_s_local = m_N_edge_s;
  ExecPolicy::launch(
      [met, N_tri_local, N_r_local, N_edge_s_local]
      LAMBDA(auto mp, auto h2_out) {
        int N_rect_ = N_r_local * N_edge_s_local;
        int N_tri_all_ = (N_r_local + 1) * N_tri_local;
        ExecPolicy::loop(0, N_rect_, [&] LAMBDA(int ri) {
          int k = ri / N_edge_s_local;
          int e = ri % N_edge_s_local;
          int t0 = mp.edge_tris[2 * e + 0];
          int t1 = mp.edge_tris[2 * e + 1];
          int fi = N_tri_all_ + ri;
          if (t0 < 0 || t1 < 0) {
            h2_out[fi] = 0;
            return;
          }

          int pid0 = k * N_tri_local + t0;
          int pid1 = k * N_tri_local + t1;
          double dist = segment_length(met, mp.cc_x[pid0], mp.cc_y[pid0],
                                       mp.cc_z[pid0], mp.cc_x[pid1],
                                       mp.cc_y[pid1], mp.cc_z[pid1]);

          int va = mp.rect_face_v0[ri], vb = mp.rect_face_v1[ri];
          int vc = mp.rect_face_v2[ri], vd = mp.rect_face_v3[ri];
          double m_area = rect_face_area(
              met, mp.vert_x[va], mp.vert_y[va], mp.vert_z[va],
              mp.vert_x[vb], mp.vert_y[vb], mp.vert_z[vb], mp.vert_x[vc],
              mp.vert_y[vc], mp.vert_z[vc], mp.vert_x[vd], mp.vert_y[vd],
              mp.vert_z[vd]);

          h2_out[fi] = (m_area > 0) ? dist / m_area : 0;
        });
      },
      this->get_ptrs(typename ExecPolicy::exec_tag{}), hodge2);

  // hodge1_inv[horizontal edge on shell k]: metric arc length / metric
  //                                         dual face (polygon) area.
  ExecPolicy::launch(
      [met, N_tri_local, N_r_local, N_edge_s_local]
      LAMBDA(auto mp, auto h1inv_out) {
        int N_h_ = (N_r_local + 1) * N_edge_s_local;
        ExecPolicy::loop(0, N_h_, [&] LAMBDA(int ei) {
          int k = ei / N_edge_s_local;
          int e = ei % N_edge_s_local;
          int v0 = mp.edge_v0[ei], v1 = mp.edge_v1[ei];
          double m_len = arc_length(met, mp.vert_x[v0], mp.vert_y[v0],
                                    mp.vert_z[v0], mp.vert_x[v1],
                                    mp.vert_y[v1], mp.vert_z[v1]);

          int t0 = mp.edge_tris[2 * e + 0];
          int t1 = mp.edge_tris[2 * e + 1];

          // Collect up to 4 dual-polygon vertices (circumcenters).
          double px[4], py[4], pz[4];
          int np = 0;
          if (k > 0 && t0 >= 0) {
            int pid = (k - 1) * N_tri_local + t0;
            px[np] = mp.cc_x[pid];
            py[np] = mp.cc_y[pid];
            pz[np] = mp.cc_z[pid];
            np++;
          }
          if (k > 0 && t1 >= 0) {
            int pid = (k - 1) * N_tri_local + t1;
            px[np] = mp.cc_x[pid];
            py[np] = mp.cc_y[pid];
            pz[np] = mp.cc_z[pid];
            np++;
          }
          if (k < N_r_local && t1 >= 0) {
            int pid = k * N_tri_local + t1;
            px[np] = mp.cc_x[pid];
            py[np] = mp.cc_y[pid];
            pz[np] = mp.cc_z[pid];
            np++;
          }
          if (k < N_r_local && t0 >= 0) {
            int pid = k * N_tri_local + t0;
            px[np] = mp.cc_x[pid];
            py[np] = mp.cc_y[pid];
            pz[np] = mp.cc_z[pid];
            np++;
          }

          double m_area = polygon_area_fixed(met, px, py, pz, np);

          // At boundary shells with only 2 dual vertices, the polygon
          // collapses.  Fall back to a triangular thin strip using
          // half the slab height as the second dimension.
          if (np == 2 && m_area < 1e-30) {
            double d = segment_length(met, px[0], py[0], pz[0], px[1], py[1],
                                      px[1]);
            double half_dr = (k == 0) ? mp.radii[1] - mp.radii[0]
                                      : mp.radii[k] - mp.radii[k - 1];
            double mx_ = 0.5 * (px[0] + px[1]);
            double my_ = 0.5 * (py[0] + py[1]);
            double mz_ = 0.5 * (pz[0] + pz[1]);
            double r_, sth_, cth_, phi_;
            cart_to_sph(mx_, my_, mz_, r_, sth_, cth_, phi_);
            half_dr *= math::sqrt(met.g_rr(r_, sth_, cth_));
            m_area = d * half_dr * 0.5;
          }

          h1inv_out[ei] = (m_area > 0) ? m_len / m_area : 0;
        });
      },
      this->get_ptrs(typename ExecPolicy::exec_tag{}), hodge1_inv);

  // hodge1_inv[vertical edge in slab k, sphere vertex s]: metric
  // radial length / metric dual polygon area.  The dual polygon has
  // one vertex per triangle incident at sphere vertex s (up to
  // max_vert_valence vertices); its vertices are the circumcenters of
  // those triangles in the slab.  Vertices are sorted by angle around
  // the edge midpoint, then the area is computed with the edge
  // midpoint as the fan center.
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
          int va = k * N_vert_s_local + s;
          int vb = (k + 1) * N_vert_s_local + s;

          double m_len =
              segment_length(met, mp.vert_x[va], mp.vert_y[va],
                             mp.vert_z[va], mp.vert_x[vb], mp.vert_y[vb],
                             mp.vert_z[vb]);

          int np = mp.vert_tri_count[s];

          // Gather circumcenter positions.
          double px[6], py[6], pz[6];
          for (int i = 0; i < np; i++) {
            int t = mp.vert_tris[s * max_valence_local + i];
            int pid = k * N_tri_local + t;
            px[i] = mp.cc_x[pid];
            py[i] = mp.cc_y[pid];
            pz[i] = mp.cc_z[pid];
          }

          // Edge midpoint (also the fan center for polygon_area_about).
          double emx = 0.5 * (mp.vert_x[va] + mp.vert_x[vb]);
          double emy = 0.5 * (mp.vert_y[va] + mp.vert_y[vb]);
          double emz = 0.5 * (mp.vert_z[va] + mp.vert_z[vb]);

          // Build an in-plane basis (u, v) perpendicular to the edge
          // axis, so we can sort circumcenters by angle.
          double nr = math::sqrt(emx * emx + emy * emy + emz * emz);
          double nnx = emx / nr, nny = emy / nr, nnz = emz / nr;
          double ux_, uy_, uz_;
          if (math::abs(nnx) < 0.9) {
            ux_ = 0;
            uy_ = -nnz;
            uz_ = nny;
          } else {
            ux_ = nnz;
            uy_ = 0;
            uz_ = -nnx;
          }
          double unorm =
              math::sqrt(ux_ * ux_ + uy_ * uy_ + uz_ * uz_);
          ux_ /= unorm;
          uy_ /= unorm;
          uz_ /= unorm;
          double vvx = nny * uz_ - nnz * uy_;
          double vvy = nnz * ux_ - nnx * uz_;
          double vvz = nnx * uy_ - nny * ux_;

          double angles[6];
          for (int i = 0; i < np; i++) {
            double dx = px[i] - emx, dy = py[i] - emy, dz = pz[i] - emz;
            angles[i] = math::atan2(dx * vvx + dy * vvy + dz * vvz,
                                    dx * ux_ + dy * uy_ + dz * uz_);
          }

          // Selection sort by angle (np ≤ 6).
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

          double spx[6], spy[6], spz[6];
          for (int i = 0; i < np; i++) {
            spx[i] = px[order[i]];
            spy[i] = py[order[i]];
            spz[i] = pz[order[i]];
          }

          double m_area =
              polygon_area_about_fixed(met, emx, emy, emz, spx, spy, spz, np);
          h1inv_out[ei] = (m_area > 0) ? m_len / m_area : 0;
        });
      },
      this->get_ptrs(typename ExecPolicy::exec_tag{}), hodge1_inv);

  ExecPolicy::sync();

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  if (mem == MemType::host_device) {
    copy_metric_to_host();
  }
#endif

  Logger::print_info("prismatic_mesh_metric::compute_metric done");
}

}  // namespace Aperture
