#include "systems/prismatic/prismatic_mesh_metric.h"
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

// Metric distance between two Cartesian points (midpoint rule).
//
// Uses the Jacobian at the midpoint to convert the Cartesian displacement
// to spherical coordinate displacement, then contracts with γ_{ij}.
// For flat space this gives EXACTLY the Euclidean distance because
// (r̂, θ̂, φ̂) form an orthonormal basis:
//   dr² + r²dθ² + r²sin²θ dφ² = dx² + dy² + dz²
double metric_dist(const spherical_metric_t& met,
                   double x0, double y0, double z0,
                   double x1, double y1, double z1) {
  // Cartesian displacement
  double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;

  // Midpoint in spherical
  double mx = 0.5 * (x0 + x1), my = 0.5 * (y0 + y1), mz = 0.5 * (z0 + z1);
  double r, sth, cth, phi;
  cart_to_sph(mx, my, mz, r, sth, cth, phi);
  if (r < 1e-30) return std::sqrt(dx*dx + dy*dy + dz*dz);

  double cph = std::cos(phi), sph = std::sin(phi);

  // Jacobian: project Cartesian displacement onto spherical basis
  //   dr   = ΔX · r̂   where r̂ = (sθcφ, sθsφ, cθ)
  //   r dθ = ΔX · θ̂   where θ̂ = (cθcφ, cθsφ, -sθ)
  //   r sθ dφ = ΔX · φ̂ where φ̂ = (-sφ, cφ, 0)
  double dr  = sth * cph * dx + sth * sph * dy + cth * dz;
  double dth = (cth * cph * dx + cth * sph * dy - sth * dz) / r;
  double dph = (sth > 1e-12) ? (-sph * dx + cph * dy) / (r * sth) : 0.0;

  double g11 = met.g_rr(r, sth, cth);
  double g22 = met.g_thth(r, sth, cth);
  double g33 = met.g_phph(r, sth, cth);
  double g13 = met.g_rph(r, sth, cth);

  double ds2 = g11 * dr * dr + g22 * dth * dth + g33 * dph * dph +
               2.0 * g13 * dr * dph;
  return std::sqrt(std::max(0.0, ds2));
}

// Metric area of a polygon given by ordered vertices in Cartesian 3D.
// Uses fan triangulation from centroid, measuring each sub-triangle
// area with the metric at the sub-triangle centroid.
double metric_polygon_area(const spherical_metric_t& met,
                           const std::vector<double>& px,
                           const std::vector<double>& py,
                           const std::vector<double>& pz) {
  int np = px.size();
  if (np < 3) return 0.0;

  // Polygon centroid
  double mx = 0, my = 0, mz = 0;
  for (int i = 0; i < np; i++) { mx += px[i]; my += py[i]; mz += pz[i]; }
  mx /= np; my /= np; mz /= np;

  double area = 0.0;
  for (int i = 0; i < np; i++) {
    int j = (i + 1) % np;

    // Sub-triangle centroid
    double cx = (mx + px[i] + px[j]) / 3.0;
    double cy = (my + py[i] + py[j]) / 3.0;
    double cz = (mz + pz[i] + pz[j]) / 3.0;
    double r, sth, cth, phi;
    cart_to_sph(cx, cy, cz, r, sth, cth, phi);

    // Flat area via cross product
    double ax = px[i] - mx, ay = py[i] - my, az = pz[i] - mz;
    double bx = px[j] - mx, by = py[j] - my, bz = pz[j] - mz;
    double nx = ay * bz - az * by;
    double ny = az * bx - ax * bz;
    double nz = ax * by - ay * bx;
    double flat_area = 0.5 * std::sqrt(nx * nx + ny * ny + nz * nz);

    // Scale by metric/flat determinant ratio:
    // √γ_metric / √γ_flat.  Both contain a factor of sinθ which
    // cancels, so compute the ratio as
    //   (√γ_metric / sinθ) / (√γ_flat / sinθ)  = √γ_tilde_metric / r²
    // to avoid division by zero at the poles.
    double sg_metric_tilde = (sth > 1e-15)
        ? met.sqrt_gamma(r, sth, cth) / sth
        : met.sqrt_gamma(r, 1e-15, cth) / 1e-15;
    double sg_flat_tilde = r * r;  // r² sinθ / sinθ = r²
    double scale = (sg_flat_tilde > 1e-30)
        ? sg_metric_tilde / sg_flat_tilde : 1.0;

    area += flat_area * scale;
  }
  return area;
}

// Metric area of a triangular face on a shell at radius r.
// Vertices given in Cartesian.
double metric_tri_area(const spherical_metric_t& met,
                       double x0, double y0, double z0,
                       double x1, double y1, double z1,
                       double x2, double y2, double z2) {
  // Centroid
  double cx = (x0 + x1 + x2) / 3.0;
  double cy = (y0 + y1 + y2) / 3.0;
  double cz = (z0 + z1 + z2) / 3.0;
  double r, sth, cth, phi;
  cart_to_sph(cx, cy, cz, r, sth, cth, phi);

  // Flat area from cross product
  double ax = x1 - x0, ay = y1 - y0, az = z1 - z0;
  double bx = x2 - x0, by = y2 - y0, bz = z2 - z0;
  double nx = ay * bz - az * by;
  double ny = az * bx - ax * bz;
  double nz = ax * by - ay * bx;
  double flat_area = 0.5 * std::sqrt(nx * nx + ny * ny + nz * nz);

  // Scale by √γ_metric / √γ_flat, canceling the common sinθ factor
  // to stay well-defined at the poles.
  double sg_met_tilde = (sth > 1e-15)
      ? met.sqrt_gamma(r, sth, cth) / sth
      : met.sqrt_gamma(r, 1e-15, cth) / 1e-15;
  double sg_flat_tilde = r * r;
  double scale = (sg_flat_tilde > 1e-30)
      ? sg_met_tilde / sg_flat_tilde : 1.0;
  return flat_area * scale;
}

// Metric area of a rectangular face spanning (r0→r1, angular edge a→b).
// Vertices: v0 = (r0, angle_a), v1 = (r0, angle_b),
//           v2 = (r1, angle_b), v3 = (r1, angle_a)
// Uses 2-point Gauss quadrature in the radial direction to handle the
// r-dependent area element accurately even for thick radial layers.
double metric_rect_area(const spherical_metric_t& met,
                        double x0, double y0, double z0,
                        double x1, double y1, double z1,
                        double x2, double y2, double z2,
                        double x3, double y3, double z3) {
  // Angular edge at the bottom shell: v0 → v1
  // Radial edge at one side: v0 → v3
  // We integrate over the face using 2 Gauss points in the radial direction.
  // At each radial sample, the angular width is the metric distance along
  // the interpolated edge.
  static constexpr double gp = 0.2113248654;  // (1 - 1/sqrt(3))/2
  double w[2] = {0.5, 0.5};
  double t[2] = {gp, 1.0 - gp};

  double area = 0.0;
  for (int i = 0; i < 2; i++) {
    // Interpolate bottom edge (v0→v1) and top edge (v3→v2) at t[i]
    double ax = (1-t[i])*x0 + t[i]*x3;  // left side at height t
    double ay = (1-t[i])*y0 + t[i]*y3;
    double az = (1-t[i])*z0 + t[i]*z3;
    double bx = (1-t[i])*x1 + t[i]*x2;  // right side at height t
    double by = (1-t[i])*y1 + t[i]*y2;
    double bz = (1-t[i])*z1 + t[i]*z2;

    // Angular width at this radial position
    double angular_len = metric_dist(met, ax, ay, az, bx, by, bz);

    // Radial element: metric distance from bottom to top along this side
    // Use the left side (v0→v3) as representative for the radial length.
    // (Could average left and right, but they're very close for resolved meshes.)
    double radial_len = metric_dist(met, x0, y0, z0, x3, y3, z3);

    area += w[i] * angular_len * radial_len;
  }
  return area;
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

      double m_area = metric_tri_area(met,
          vert_x[va], vert_y[va], vert_z[va],
          vert_x[vb], vert_y[vb], vert_z[vb],
          vert_x[vc], vert_y[vc], vert_z[vc]);

      double dist;
      if (k == 0) {
        int pid_above = 0 * m_N_tri + t;
        double fx = (vert_x[va] + vert_x[vb] + vert_x[vc]) / 3.0;
        double fy = (vert_y[va] + vert_y[vb] + vert_y[vc]) / 3.0;
        double fz = (vert_z[va] + vert_z[vb] + vert_z[vc]) / 3.0;
        dist = metric_dist(met, fx, fy, fz,
                           cx[pid_above], cy[pid_above], cz[pid_above]) * 2.0;
      } else if (k == m_N_r) {
        int pid_below = (m_N_r - 1) * m_N_tri + t;
        double fx = (vert_x[va] + vert_x[vb] + vert_x[vc]) / 3.0;
        double fy = (vert_y[va] + vert_y[vb] + vert_y[vc]) / 3.0;
        double fz = (vert_z[va] + vert_z[vb] + vert_z[vc]) / 3.0;
        dist = metric_dist(met, fx, fy, fz,
                           cx[pid_below], cy[pid_below], cz[pid_below]) * 2.0;
      } else {
        int pid_below = (k - 1) * m_N_tri + t;
        int pid_above = k * m_N_tri + t;
        dist = metric_dist(met,
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
      double dist = metric_dist(met,
                                cx[pid0], cy[pid0], cz[pid0],
                                cx[pid1], cy[pid1], cz[pid1]);

      int a_s = tri_verts[t0 * 3 + 0];  // just need sphere edge endpoints
      // Get rect face vertices for area
      int local = k * m_N_edge_s + e;
      int v0 = rect_face_v0[local], v1 = rect_face_v1[local];
      int v2 = rect_face_v2[local], v3 = rect_face_v3[local];
      double m_area = metric_rect_area(met,
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

      double m_len = metric_dist(met,
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

      double m_area = metric_polygon_area(met, px, py, pz);

      if (px.size() == 2 && m_area < 1e-30) {
        double d = metric_dist(met, px[0], py[0], pz[0], px[1], py[1], pz[1]);
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

      double m_len = metric_dist(met,
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

      // Reorder for metric_polygon_area
      std::vector<double> spx(np), spy(np), spz(np);
      for (int i = 0; i < np; i++) {
        spx[i] = px[order[i]];
        spy[i] = py[order[i]];
        spz[i] = pz[order[i]];
      }

      double m_area = metric_polygon_area(met, spx, spy, spz);
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

  face_r_coord.resize(m_N_faces);
  face_sth.resize(m_N_faces);
  face_cth.resize(m_N_faces);
  face_alpha.resize(m_N_faces);
  face_sq_gamma_beta_r.resize(m_N_faces);

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
  p.face_r_coord = acc(m.face_r_coord);
  p.face_sth = acc(m.face_sth);
  p.face_cth = acc(m.face_cth);
  p.face_alpha = acc(m.face_alpha);
  p.face_sq_gamma_beta_r = acc(m.face_sq_gamma_beta_r);
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
  copy(edge_alpha); copy(edge_sq_gamma_beta_r);
  copy(face_r_coord); copy(face_sth); copy(face_cth);
  copy(face_alpha); copy(face_sq_gamma_beta_r);
}
#endif

}  // namespace Aperture
