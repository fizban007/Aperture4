#include "systems/prismatic/prismatic_mesh_metric.h"
#include "utils/gauss_quadrature.h"
#include "utils/logger.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace Aperture {

namespace {

// Convert Cartesian point to spherical (r, sinθ, cosθ, φ)
void cart_to_sph(double x, double y, double z,
                 double& r, double& sth, double& cth, double& phi) {
  r = std::sqrt(x * x + y * y + z * z);
  cth = (r > 0) ? z / r : 1.0;
  sth = std::sqrt(std::max(0.0, 1.0 - cth * cth));
  phi = std::atan2(y, x);
}

// Metric inner product γ_{ij} V^i W^j at a Cartesian point P,
// where V and W are Cartesian vectors.  Uses the Jacobian at P to
// project Cartesian components onto the spherical coordinate basis,
// then contracts with the spherical metric components.
//
// For flat space (γ = diag(1, r², r²sin²θ)) this reduces exactly to
// V · W because (r̂, θ̂, φ̂) form an orthonormal basis.
double metric_inner_product(const spherical_metric_t& met,
                            double x, double y, double z,
                            double vx, double vy, double vz,
                            double wx, double wy, double wz) {
  double r, sth, cth, phi;
  cart_to_sph(x, y, z, r, sth, cth, phi);
  if (r < 1e-30) return vx * wx + vy * wy + vz * wz;

  double cph = std::cos(phi), sph = std::sin(phi);

  // Project Cartesian vectors onto (r̂, θ̂, φ̂).
  //   dr       = V · r̂,           with r̂  = (sθcφ, sθsφ, cθ)
  //   r dθ     = V · θ̂,           with θ̂  = (cθcφ, cθsφ, -sθ)
  //   r sθ dφ  = V · φ̂,           with φ̂  = (-sφ, cφ, 0)
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

// Metric length of a straight Cartesian line segment from P0 to P1.
// Used for vertical edges, dual edges, and polygon sub-triangle edges.
double segment_length(const spherical_metric_t& met,
                      double x0, double y0, double z0,
                      double x1, double y1, double z1) {
  double vx = x1 - x0, vy = y1 - y0, vz = z1 - z0;
  return gauss_quad([&](double t) {
    double x = x0 + t * vx, y = y0 + t * vy, z = z0 + t * vz;
    double ds2 = metric_inner_product(met, x, y, z, vx, vy, vz, vx, vy, vz);
    return std::sqrt(std::max(0.0, ds2));
  }, 0.0, 1.0);
}

// Metric length of a great-circle arc on a sphere of radius r between
// Cartesian points P0 and P1 (which must have |P0| = |P1| = r).
// Used for primal horizontal edges on shells.
double arc_length(const spherical_metric_t& met,
                  double x0, double y0, double z0,
                  double x1, double y1, double z1) {
  double r = std::sqrt(x0 * x0 + y0 * y0 + z0 * z0);
  double dot = (x0 * x1 + y0 * y1 + z0 * z1) / (r * r);
  dot = std::max(-1.0, std::min(1.0, dot));
  double Omega = std::acos(dot);
  if (Omega < 1e-12) return 0.0;

  // Unit vectors along the arc
  double sO = std::sin(Omega);
  // Parameterize: x(t) = r * (cos(tΩ) ŝ_a + sin(tΩ) ê_t) for t ∈ [0, 1]
  // where ê_t = (ŝ_b - cos(Ω) ŝ_a) / sin(Ω) is the unit tangent at P0.
  double sax = x0 / r, say = y0 / r, saz = z0 / r;
  double sbx = x1 / r, sby = y1 / r, sbz = z1 / r;
  double etx = (sbx - dot * sax) / sO;
  double ety = (sby - dot * say) / sO;
  double etz = (sbz - dot * saz) / sO;

  return gauss_quad([&](double t) {
    double s = t * Omega;
    double cs = std::cos(s), ss = std::sin(s);
    double x = r * (cs * sax + ss * etx);
    double y = r * (cs * say + ss * ety);
    double z = r * (cs * saz + ss * etz);
    // dx/dt = r Ω (-sin(s) ŝ_a + cos(s) ê_t)
    double tx = r * Omega * (-ss * sax + cs * etx);
    double ty = r * Omega * (-ss * say + cs * ety);
    double tz = r * Omega * (-ss * saz + cs * etz);
    double ds2 = metric_inner_product(met, x, y, z, tx, ty, tz, tx, ty, tz);
    return std::sqrt(std::max(0.0, ds2));
  }, 0.0, 1.0);
}

// Metric area of a flat Cartesian triangle with vertices V0, V1, V2.
// Uses nested Gauss quadrature on the barycentric parameterization
//   x(u, v) = V0 + u e1 + v e2,   u ∈ [0, 1], v ∈ [0, 1-u]
// where e1 = V1 - V0, e2 = V2 - V0.
double triangle_area(const spherical_metric_t& met,
                     double x0, double y0, double z0,
                     double x1, double y1, double z1,
                     double x2, double y2, double z2) {
  double e1x = x1 - x0, e1y = y1 - y0, e1z = z1 - z0;
  double e2x = x2 - x0, e2y = y2 - y0, e2z = z2 - z0;

  return gauss_quad([&](double u) {
    return gauss_quad([&](double v) {
      double x = x0 + u * e1x + v * e2x;
      double y = y0 + u * e1y + v * e2y;
      double z = z0 + u * e1z + v * e2z;
      double h11 = metric_inner_product(met, x, y, z,
                                        e1x, e1y, e1z, e1x, e1y, e1z);
      double h12 = metric_inner_product(met, x, y, z,
                                        e1x, e1y, e1z, e2x, e2y, e2z);
      double h22 = metric_inner_product(met, x, y, z,
                                        e2x, e2y, e2z, e2x, e2y, e2z);
      double det = h11 * h22 - h12 * h12;
      return std::sqrt(std::max(0.0, det));
    }, 0.0, 1.0 - u);
  }, 0.0, 1.0);
}

// Metric area of a rectangular face spanning (arc_angle u ∈ [0,Ω],
// radial v ∈ [r0, r1]).  The face is on a constant-angle surface:
//   x(u, v) = v · (cos u · ŝ_a + sin u · ê_t)
// This exactly reproduces the base-class formula  dr · 0.5(r0+r1) · Ω
// in the flat limit.
double rect_face_area(const spherical_metric_t& met,
                      double x0, double y0, double z0,  // (r0, a)
                      double x1, double y1, double z1,  // (r0, b)
                      double x2, double y2, double z2,  // (r1, b)
                      double x3, double y3, double z3)  // (r1, a)
{
  (void)x2; (void)y2; (void)z2;  // v2 is implied by bilinear consistency
  double r0 = std::sqrt(x0 * x0 + y0 * y0 + z0 * z0);
  double r1 = std::sqrt(x3 * x3 + y3 * y3 + z3 * z3);

  double sax = x0 / r0, say = y0 / r0, saz = z0 / r0;
  double sbx = x1 / r0, sby = y1 / r0, sbz = z1 / r0;
  double dot = sax * sbx + say * sby + saz * sbz;
  dot = std::max(-1.0, std::min(1.0, dot));
  double Omega = std::acos(dot);
  if (Omega < 1e-12 || r1 - r0 < 1e-30) return 0.0;

  double sO = std::sin(Omega);
  double etx = (sbx - dot * sax) / sO;
  double ety = (sby - dot * say) / sO;
  double etz = (sbz - dot * saz) / sO;

  return gauss_quad([&](double u) {  // u ∈ [0, Ω]
    double cs = std::cos(u), ss = std::sin(u);
    double ux = cs * sax + ss * etx;
    double uy = cs * say + ss * ety;
    double uz = cs * saz + ss * etz;
    // Tangent along angle: ∂x/∂u = v · (-sin u · ŝ_a + cos u · ê_t)
    double t_ang_x_norm = -ss * sax + cs * etx;
    double t_ang_y_norm = -ss * say + cs * ety;
    double t_ang_z_norm = -ss * saz + cs * etz;

    return gauss_quad([&](double v) {  // v ∈ [r0, r1]
      double x = v * ux, y = v * uy, z = v * uz;
      // ∂x/∂u (with dimensions): v * tangent_direction
      double du_x = v * t_ang_x_norm;
      double du_y = v * t_ang_y_norm;
      double du_z = v * t_ang_z_norm;
      // ∂x/∂v: unit radial direction at angle u
      double dv_x = ux, dv_y = uy, dv_z = uz;

      double h11 = metric_inner_product(met, x, y, z,
                                        du_x, du_y, du_z, du_x, du_y, du_z);
      double h12 = metric_inner_product(met, x, y, z,
                                        du_x, du_y, du_z, dv_x, dv_y, dv_z);
      double h22 = metric_inner_product(met, x, y, z,
                                        dv_x, dv_y, dv_z, dv_x, dv_y, dv_z);
      double det = h11 * h22 - h12 * h12;
      return std::sqrt(std::max(0.0, det));
    }, r0, r1);
  }, 0.0, Omega);
}

// Metric area of a polygon in Cartesian 3D, given ordered vertices and
// an explicit fan center (typically the dual edge's representative point:
// the polygon vertex average for horizontal edges, the primal edge midpoint
// for vertical edges, matching the base class convention).  Each sub-
// triangle is integrated with triangle_area.
double polygon_area_about(const spherical_metric_t& met,
                          double cx, double cy, double cz,
                          const std::vector<double>& px,
                          const std::vector<double>& py,
                          const std::vector<double>& pz) {
  int np = px.size();
  if (np < 3) return 0.0;

  double area = 0.0;
  for (int i = 0; i < np; i++) {
    int j = (i + 1) % np;
    area += triangle_area(met,
                          cx, cy, cz,
                          px[i], py[i], pz[i],
                          px[j], py[j], pz[j]);
  }
  return area;
}

// Convenience: fan from the polygon vertex average.
double polygon_area(const spherical_metric_t& met,
                    const std::vector<double>& px,
                    const std::vector<double>& py,
                    const std::vector<double>& pz) {
  int np = px.size();
  if (np < 3) return 0.0;

  double mx = 0, my = 0, mz = 0;
  for (int i = 0; i < np; i++) { mx += px[i]; my += py[i]; mz += pz[i]; }
  mx /= np; my /= np; mz /= np;
  return polygon_area_about(met, mx, my, mz, px, py, pz);
}

}  // anonymous namespace

// =========================================================================
// Recompute circumcenters from persisted sphere mesh data.
// Same algorithm as prismatic_mesh::compute_geometric_dual() step 1.
// =========================================================================
void prismatic_mesh_metric::compute_circumcenters(
    std::vector<double>& ccx, std::vector<double>& ccy,
    std::vector<double>& ccz) const {
  int N_prisms = m_N_tri * m_N_r;
  ccx.resize(N_prisms);
  ccy.resize(N_prisms);
  ccz.resize(N_prisms);

  const int* tv = tri_verts.host_ptr();
  const Scalar* svx = sphere_vx.host_ptr();
  const Scalar* svy = sphere_vy.host_ptr();
  const Scalar* svz = sphere_vz.host_ptr();

  for (int k = 0; k < m_N_r; k++) {
    double r_mid = 0.5 * (radii[k] + radii[k + 1]);
    for (int t = 0; t < m_N_tri; t++) {
      int pid = k * m_N_tri + t;
      int a = tv[t * 3 + 0];
      int b = tv[t * 3 + 1];
      int c = tv[t * 3 + 2];

      // Spherical circumcenter: cross product of chord edges
      double e1x = svx[b] - svx[a];
      double e1y = svy[b] - svy[a];
      double e1z = svz[b] - svz[a];
      double e2x = svx[c] - svx[a];
      double e2y = svy[c] - svy[a];
      double e2z = svz[c] - svz[a];
      double nx = e1y * e2z - e1z * e2y;
      double ny = e1z * e2x - e1x * e2z;
      double nz = e1x * e2y - e1y * e2x;
      double nlen = std::sqrt(nx * nx + ny * ny + nz * nz);

      double ux = 0, uy = 0, uz = 0;
      if (nlen > 0) {
        ux = nx / nlen; uy = ny / nlen; uz = nz / nlen;
        // Orient outward
        double mx = (svx[a] + svx[b] + svx[c]) / 3.0;
        double my = (svy[a] + svy[b] + svy[c]) / 3.0;
        double mz = (svz[a] + svz[b] + svz[c]) / 3.0;
        if (ux * mx + uy * my + uz * mz < 0) {
          ux = -ux; uy = -uy; uz = -uz;
        }
      }
      ccx[pid] = r_mid * ux;
      ccy[pid] = r_mid * uy;
      ccz[pid] = r_mid * uz;
    }
  }
}

// =========================================================================
// Build edge-to-triangle adjacency from sphere data.
// =========================================================================
void prismatic_mesh_metric::build_edge_tris(
    std::vector<std::array<int, 2>>& et) const {
  et.resize(m_N_edge_s);
  for (int e = 0; e < m_N_edge_s; e++) { et[e] = {-1, -1}; }
  const int* te = tri_edges_s.host_ptr();
  for (int t = 0; t < m_N_tri; t++) {
    for (int j = 0; j < 3; j++) {
      int e = te[t * 3 + j];
      if (et[e][0] == -1) et[e][0] = t;
      else et[e][1] = t;
    }
  }
}

// =========================================================================
// Compute metric-weighted Hodge stars.
// Overwrites hodge1_inv and hodge2 with metric-weighted values.
// =========================================================================
void prismatic_mesh_metric::compute_hodge_metric(
    const spherical_metric_t& met,
    const std::vector<double>& cx,
    const std::vector<double>& cy,
    const std::vector<double>& cz,
    const std::vector<std::array<int, 2>>& edge_tris) {

  // --- hodge2: metric dual_edge_length / metric face_area ---
  hodge2.resize(m_N_faces);

  // Triangular faces
  for (int k = 0; k <= m_N_r; k++) {
    for (int t = 0; t < m_N_tri; t++) {
      int fi = tri_face_idx(k, t);
      int a = tri_verts[t * 3 + 0];
      int b = tri_verts[t * 3 + 1];
      int c = tri_verts[t * 3 + 2];
      int va = vert_idx(k, a), vb = vert_idx(k, b), vc = vert_idx(k, c);

      double m_area = triangle_area(met,
          vert_x[va], vert_y[va], vert_z[va],
          vert_x[vb], vert_y[vb], vert_z[vb],
          vert_x[vc], vert_y[vc], vert_z[vc]);

      double dist;
      if (k == 0) {
        int pid_above = 0 * m_N_tri + t;
        double fx = (vert_x[va] + vert_x[vb] + vert_x[vc]) / 3.0;
        double fy = (vert_y[va] + vert_y[vb] + vert_y[vc]) / 3.0;
        double fz = (vert_z[va] + vert_z[vb] + vert_z[vc]) / 3.0;
        dist = segment_length(met, fx, fy, fz,
                              cx[pid_above], cy[pid_above], cz[pid_above]) * 2.0;
      } else if (k == m_N_r) {
        int pid_below = (m_N_r - 1) * m_N_tri + t;
        double fx = (vert_x[va] + vert_x[vb] + vert_x[vc]) / 3.0;
        double fy = (vert_y[va] + vert_y[vb] + vert_y[vc]) / 3.0;
        double fz = (vert_z[va] + vert_z[vb] + vert_z[vc]) / 3.0;
        dist = segment_length(met, fx, fy, fz,
                              cx[pid_below], cy[pid_below], cz[pid_below]) * 2.0;
      } else {
        int pid_below = (k - 1) * m_N_tri + t;
        int pid_above = k * m_N_tri + t;
        dist = segment_length(met,
                              cx[pid_below], cy[pid_below], cz[pid_below],
                              cx[pid_above], cy[pid_above], cz[pid_above]);
      }
      hodge2[fi] = (m_area > 0) ? dist / m_area : 0;
    }
  }

  // Rectangular faces
  for (int k = 0; k < m_N_r; k++) {
    for (int e = 0; e < m_N_edge_s; e++) {
      int fi = rect_face_idx(k, e);
      int t0 = edge_tris[e][0], t1 = edge_tris[e][1];
      if (t0 == -1 || t1 == -1) { hodge2[fi] = 0; continue; }

      int pid0 = k * m_N_tri + t0;
      int pid1 = k * m_N_tri + t1;
      double dist = segment_length(met,
                                   cx[pid0], cy[pid0], cz[pid0],
                                   cx[pid1], cy[pid1], cz[pid1]);

      int local = k * m_N_edge_s + e;
      int v0 = rect_face_v0[local], v1 = rect_face_v1[local];
      int v2 = rect_face_v2[local], v3 = rect_face_v3[local];
      double m_area = rect_face_area(met,
          vert_x[v0], vert_y[v0], vert_z[v0],
          vert_x[v1], vert_y[v1], vert_z[v1],
          vert_x[v2], vert_y[v2], vert_z[v2],
          vert_x[v3], vert_y[v3], vert_z[v3]);

      hodge2[fi] = (m_area > 0) ? dist / m_area : 0;
    }
  }

  // --- hodge1_inv: metric edge_length / metric dual_face_area ---
  hodge1_inv.resize(m_N_edges);

  // Horizontal edges
  for (int k = 0; k <= m_N_r; k++) {
    for (int e = 0; e < m_N_edge_s; e++) {
      int ei = h_edge_idx(k, e);
      int v0 = edge_v0[ei], v1 = edge_v1[ei];

      // Horizontal edges on a shell: integrate along the great-circle arc
      double m_len = arc_length(met,
          vert_x[v0], vert_y[v0], vert_z[v0],
          vert_x[v1], vert_y[v1], vert_z[v1]);

      int t0 = edge_tris[e][0], t1 = edge_tris[e][1];

      // Collect circumcenters forming the dual face polygon
      std::vector<double> px, py, pz;
      if (k > 0 && t0 >= 0) {
        int pid = (k - 1) * m_N_tri + t0;
        px.push_back(cx[pid]); py.push_back(cy[pid]); pz.push_back(cz[pid]);
      }
      if (k > 0 && t1 >= 0) {
        int pid = (k - 1) * m_N_tri + t1;
        px.push_back(cx[pid]); py.push_back(cy[pid]); pz.push_back(cz[pid]);
      }
      if (k < m_N_r && t1 >= 0) {
        int pid = k * m_N_tri + t1;
        px.push_back(cx[pid]); py.push_back(cy[pid]); pz.push_back(cz[pid]);
      }
      if (k < m_N_r && t0 >= 0) {
        int pid = k * m_N_tri + t0;
        px.push_back(cx[pid]); py.push_back(cy[pid]); pz.push_back(cz[pid]);
      }

      double m_area = polygon_area(met, px, py, pz);

      if (px.size() == 2 && m_area < 1e-30) {
        double d = segment_length(met, px[0], py[0], pz[0],
                                  px[1], py[1], pz[1]);
        double half_dr = (k == 0) ? radii[1] - radii[0] : radii[k] - radii[k - 1];
        // Scale half_dr by sqrt(g_rr) at midpoint
        double mx = 0.5 * (px[0] + px[1]);
        double my = 0.5 * (py[0] + py[1]);
        double mz = 0.5 * (pz[0] + pz[1]);
        double r, sth, cth, phi;
        cart_to_sph(mx, my, mz, r, sth, cth, phi);
        half_dr *= std::sqrt(met.g_rr(r, sth, cth));
        m_area = d * half_dr * 0.5;
      }

      hodge1_inv[ei] = (m_area > 0) ? m_len / m_area : 0;
    }
  }

  // Vertical edges
  // Build vertex-to-triangle adjacency
  std::vector<std::vector<int>> vert_tris(m_N_vert_s);
  for (int t = 0; t < m_N_tri; t++) {
    for (int j = 0; j < 3; j++) {
      vert_tris[tri_verts[t * 3 + j]].push_back(t);
    }
  }

  for (int k = 0; k < m_N_r; k++) {
    for (int s = 0; s < m_N_vert_s; s++) {
      int ei = v_edge_idx(k, s);
      int va = vert_idx(k, s), vb = vert_idx(k + 1, s);

      // Vertical edges: integrate along the straight radial line
      double m_len = segment_length(met,
          vert_x[va], vert_y[va], vert_z[va],
          vert_x[vb], vert_y[vb], vert_z[vb]);

      auto& tris = vert_tris[s];
      int np = tris.size();

      std::vector<double> px(np), py(np), pz(np);
      for (int i = 0; i < np; i++) {
        int pid = k * m_N_tri + tris[i];
        px[i] = cx[pid]; py[i] = cy[pid]; pz[i] = cz[pid];
      }

      // Sort by angle around the vertical edge
      double emx = 0.5 * (vert_x[va] + vert_x[vb]);
      double emy = 0.5 * (vert_y[va] + vert_y[vb]);
      double emz = 0.5 * (vert_z[va] + vert_z[vb]);

      double nr = std::sqrt(emx * emx + emy * emy + emz * emz);
      double nnx = emx / nr, nny = emy / nr, nnz = emz / nr;
      double ux, uy, uz;
      if (std::abs(nnx) < 0.9) {
        ux = 0; uy = -nnz; uz = nny;
      } else {
        ux = nnz; uy = 0; uz = -nnx;
      }
      double unorm = std::sqrt(ux * ux + uy * uy + uz * uz);
      ux /= unorm; uy /= unorm; uz /= unorm;
      double vx = nny * uz - nnz * uy;
      double vy = nnz * ux - nnx * uz;
      double vz = nnx * uy - nny * ux;

      std::vector<double> angles(np);
      for (int i = 0; i < np; i++) {
        double dx = px[i] - emx, dy = py[i] - emy, dz = pz[i] - emz;
        angles[i] = std::atan2(dx * vx + dy * vy + dz * vz,
                               dx * ux + dy * uy + dz * uz);
      }
      std::vector<int> order(np);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&](int a, int b) { return angles[a] < angles[b]; });

      // Reorder for polygon_area_about
      std::vector<double> spx(np), spy(np), spz(np);
      for (int i = 0; i < np; i++) {
        spx[i] = px[order[i]];
        spy[i] = py[order[i]];
        spz[i] = pz[order[i]];
      }

      // Use the edge midpoint as the fan center, matching the base class
      // convention.  For vertical edges this is on the edge axis.
      double m_area = polygon_area_about(met, emx, emy, emz, spx, spy, spz);
      hodge1_inv[ei] = (m_area > 0) ? m_len / m_area : 0;
    }
  }

  Logger::print_info(
      "Metric Hodge computed: hodge1_inv [{:.4f}, {:.4f}], "
      "hodge2 [{:.6f}, {:.6f}]",
      *std::min_element(&hodge1_inv[0], &hodge1_inv[m_N_edges - 1]),
      *std::max_element(&hodge1_inv[0], &hodge1_inv[m_N_edges - 1]),
      *std::min_element(&hodge2[0], &hodge2[m_N_faces - 1]),
      *std::max_element(&hodge2[0], &hodge2[m_N_faces - 1]));
}

// =========================================================================
// Main entry point: compute metric-weighted Hodge stars + per-element data.
// =========================================================================
void prismatic_mesh_metric::compute_metric(const spherical_metric_t& met) {
  // --- Allocate per-element arrays ---
  edge_r_coord.resize(m_N_edges);
  edge_sth.resize(m_N_edges);
  edge_cth.resize(m_N_edges);
  edge_alpha.resize(m_N_edges);
  edge_sq_gamma_beta_r.resize(m_N_edges);
  edge_sqrt_gamma.resize(m_N_edges);

  face_r_coord.resize(m_N_faces);
  face_sth.resize(m_N_faces);
  face_cth.resize(m_N_faces);
  face_alpha.resize(m_N_faces);
  face_sq_gamma_beta_r.resize(m_N_faces);
  face_sqrt_gamma.resize(m_N_faces);

  // --- Per-edge: evaluate metric at edge midpoints ---
  for (int e = 0; e < m_N_edges; e++) {
    int v0 = edge_v0[e], v1 = edge_v1[e];
    double mx = 0.5 * (vert_x[v0] + vert_x[v1]);
    double my = 0.5 * (vert_y[v0] + vert_y[v1]);
    double mz = 0.5 * (vert_z[v0] + vert_z[v1]);
    double r, sth, cth, phi;
    cart_to_sph(mx, my, mz, r, sth, cth, phi);

    edge_r_coord[e] = r;
    edge_sth[e] = sth;
    edge_cth[e] = cth;
    edge_alpha[e] = met.alpha(r, sth, cth);
    edge_sq_gamma_beta_r[e] = met.sq_gamma_beta_r(r, sth, cth);
    edge_sqrt_gamma[e] = met.sqrt_gamma(r, sth, cth);
  }

  // --- Per-face: evaluate metric at face centroids ---
  int n_tri_faces = m_N_tri * (m_N_r + 1);
  for (int fi = 0; fi < n_tri_faces; fi++) {
    int va = tri_face_v0[fi], vb = tri_face_v1[fi], vc = tri_face_v2[fi];
    double cx = (vert_x[va] + vert_x[vb] + vert_x[vc]) / 3.0;
    double cy = (vert_y[va] + vert_y[vb] + vert_y[vc]) / 3.0;
    double cz = (vert_z[va] + vert_z[vb] + vert_z[vc]) / 3.0;
    double r, sth, cth, phi;
    cart_to_sph(cx, cy, cz, r, sth, cth, phi);

    face_r_coord[fi] = r;
    face_sth[fi] = sth;
    face_cth[fi] = cth;
    face_alpha[fi] = met.alpha(r, sth, cth);
    face_sq_gamma_beta_r[fi] = met.sq_gamma_beta_r(r, sth, cth);
    face_sqrt_gamma[fi] = met.sqrt_gamma(r, sth, cth);
  }

  int n_rect_faces = m_N_edge_s * m_N_r;
  for (int ri = 0; ri < n_rect_faces; ri++) {
    int fi = n_tri_faces + ri;
    int va = rect_face_v0[ri], vb = rect_face_v1[ri];
    int vc = rect_face_v2[ri], vd = rect_face_v3[ri];
    double cx = 0.25 * (vert_x[va] + vert_x[vb] + vert_x[vc] + vert_x[vd]);
    double cy = 0.25 * (vert_y[va] + vert_y[vb] + vert_y[vc] + vert_y[vd]);
    double cz = 0.25 * (vert_z[va] + vert_z[vb] + vert_z[vc] + vert_z[vd]);
    double r, sth, cth, phi;
    cart_to_sph(cx, cy, cz, r, sth, cth, phi);

    face_r_coord[fi] = r;
    face_sth[fi] = sth;
    face_cth[fi] = cth;
    face_alpha[fi] = met.alpha(r, sth, cth);
    face_sq_gamma_beta_r[fi] = met.sq_gamma_beta_r(r, sth, cth);
    face_sqrt_gamma[fi] = met.sqrt_gamma(r, sth, cth);
  }

  // --- Recompute Hodge stars with the metric ---
  std::vector<double> ccx, ccy, ccz;
  compute_circumcenters(ccx, ccy, ccz);

  std::vector<std::array<int, 2>> et;
  build_edge_tris(et);

  compute_hodge_metric(met, ccx, ccy, ccz, et);
}

namespace {

template <typename Accessor>
void fill_metric_ptrs(prismatic_mesh_metric_ptrs& p,
                      const prismatic_mesh_metric& m, Accessor acc) {
  p.edge_r_coord = acc(m.edge_r_coord);
  p.edge_sth = acc(m.edge_sth);
  p.edge_cth = acc(m.edge_cth);
  p.edge_alpha = acc(m.edge_alpha);
  p.edge_sq_gamma_beta_r = acc(m.edge_sq_gamma_beta_r);
  p.edge_sqrt_gamma = acc(m.edge_sqrt_gamma);
  p.face_r_coord = acc(m.face_r_coord);
  p.face_sth = acc(m.face_sth);
  p.face_cth = acc(m.face_cth);
  p.face_alpha = acc(m.face_alpha);
  p.face_sq_gamma_beta_r = acc(m.face_sq_gamma_beta_r);
  p.face_sqrt_gamma = acc(m.face_sqrt_gamma);
  p.N_h_edges = (m.m_N_r + 1) * m.m_N_edge_s;
  p.N_tri_faces = (m.m_N_r + 1) * m.m_N_tri;
}

}  // anonymous namespace

prismatic_mesh_metric_ptrs prismatic_mesh_metric::host_ptrs_metric() const {
  prismatic_mesh_metric_ptrs p{};
  static_cast<prismatic_mesh_ptrs&>(p) = prismatic_mesh::host_ptrs();
  auto acc = [](const auto& buf) { return buf.host_ptr(); };
  fill_metric_ptrs(p, *this, acc);
  return p;
}

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
prismatic_mesh_metric_ptrs prismatic_mesh_metric::dev_ptrs_metric() const {
  prismatic_mesh_metric_ptrs p{};
  static_cast<prismatic_mesh_ptrs&>(p) = prismatic_mesh::dev_ptrs();
  auto acc = [](const auto& buf) { return buf.dev_ptr(); };
  fill_metric_ptrs(p, *this, acc);
  return p;
}

void prismatic_mesh_metric::copy_to_device() {
  prismatic_mesh::copy_to_device();

  auto copy = [](auto& buf) { buf.copy_to_device(); };
  copy(edge_r_coord); copy(edge_sth); copy(edge_cth);
  copy(edge_alpha); copy(edge_sq_gamma_beta_r); copy(edge_sqrt_gamma);
  copy(face_r_coord); copy(face_sth); copy(face_cth);
  copy(face_alpha); copy(face_sq_gamma_beta_r); copy(face_sqrt_gamma);
}
#endif

}  // namespace Aperture
