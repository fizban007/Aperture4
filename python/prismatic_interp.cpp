// Standalone pybind11 module for Whitney-form field interpolation on
// the prismatic mesh.  No Aperture library dependency — uses only the
// header-only mesh_ptrs and deposit/interpolation code.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace py = pybind11;

// =========================================================================
// Minimal type definitions (avoid pulling in the framework)
// =========================================================================
using Scalar = float;

// Pull in the header-only mesh pointer struct and interpolation routines.
// These have no library dependencies — just HD_INLINE / HOST_DEVICE functions.
// On CPU builds (no __CUDACC__ / __HIPCC__), HD_INLINE = inline,
// HOST_DEVICE = empty.
#ifndef __CUDACC__
#ifndef __HIPCC__
#define HOST_DEVICE
#define HD_INLINE inline
#endif
#endif

// =========================================================================
// Inline the mesh pointer struct (mirrors prismatic_mesh_ptrs.h)
// =========================================================================
struct mesh_data {
  int N_r, N_tri, N_vert_s, N_edge_s, N_verts, N_edges, N_faces;

  const Scalar* radii;
  const int* d1_row_ptr;
  const int* d1_col_idx;
  const Scalar* d1_val;
  const int* d1t_row_ptr;
  const int* d1t_col_idx;
  const Scalar* d1t_val;
  const Scalar* hodge1_inv;
  const Scalar* hodge2;
  const int* edge_boundary;
  const int* face_boundary;
  const int* edge_radial_layer;
  const int* face_radial_layer;
  // Note: the full mesh stores (r, θ, φ) per vertex, but this Python module
  // only needs the unit-sphere vertex directions (sphere_v{x,y,z}) and the
  // radii[] array to reconstruct Cartesian positions.  No per-vertex
  // Cartesian or spherical arrays are referenced here.
  const int* edge_v0;
  const int* edge_v1;
  const int* tri_face_v0;
  const int* tri_face_v1;
  const int* tri_face_v2;
  const int* rect_face_v0;
  const int* rect_face_v1;
  const int* rect_face_v2;
  const int* rect_face_v3;
  const Scalar* sphere_vx;
  const Scalar* sphere_vy;
  const Scalar* sphere_vz;
  const int* tri_verts;
  const int* tri_edges_s;
  const int* tri_edge_signs;
  const int* tri_neighbor;

  // Indexing
  int h_edge_idx(int k, int e) const { return k * N_edge_s + e; }
  int v_edge_idx(int k, int s) const {
    return (N_r + 1) * N_edge_s + k * N_vert_s + s;
  }
  int tri_face_idx(int k, int t) const { return k * N_tri + t; }
  int rect_face_idx(int k, int e) const {
    return (N_r + 1) * N_tri + k * N_edge_s + e;
  }

  void prism_edge_indices(int t, int k, int edges[9]) const {
    edges[0] = h_edge_idx(k, tri_edges_s[t*3+0]);
    edges[1] = h_edge_idx(k, tri_edges_s[t*3+1]);
    edges[2] = h_edge_idx(k, tri_edges_s[t*3+2]);
    edges[3] = h_edge_idx(k+1, tri_edges_s[t*3+0]);
    edges[4] = h_edge_idx(k+1, tri_edges_s[t*3+1]);
    edges[5] = h_edge_idx(k+1, tri_edges_s[t*3+2]);
    edges[6] = v_edge_idx(k, tri_verts[t*3+0]);
    edges[7] = v_edge_idx(k, tri_verts[t*3+1]);
    edges[8] = v_edge_idx(k, tri_verts[t*3+2]);
  }

  void compute_barycentric(int t, Scalar sx, Scalar sy, Scalar sz,
                           Scalar& l1, Scalar& l2, Scalar& l3) const {
    int v0 = tri_verts[t*3], v1 = tri_verts[t*3+1], v2 = tri_verts[t*3+2];
    Scalar e1x = sphere_vx[v1]-sphere_vx[v0], e1y = sphere_vy[v1]-sphere_vy[v0], e1z = sphere_vz[v1]-sphere_vz[v0];
    Scalar e2x = sphere_vx[v2]-sphere_vx[v0], e2y = sphere_vy[v2]-sphere_vy[v0], e2z = sphere_vz[v2]-sphere_vz[v0];
    Scalar nx = e1y*e2z - e1z*e2y, ny = e1z*e2x - e1x*e2z, nz = e1x*e2y - e1y*e2x;
    Scalar nn = nx*nx + ny*ny + nz*nz;
    Scalar d1x = sphere_vx[v1]-sx, d1y = sphere_vy[v1]-sy, d1z = sphere_vz[v1]-sz;
    Scalar d2x = sphere_vx[v2]-sx, d2y = sphere_vy[v2]-sy, d2z = sphere_vz[v2]-sz;
    l1 = ((d1y*d2z-d1z*d2y)*nx + (d1z*d2x-d1x*d2z)*ny + (d1x*d2y-d1y*d2x)*nz) / nn;
    Scalar d0x = sphere_vx[v0]-sx, d0y = sphere_vy[v0]-sy, d0z = sphere_vz[v0]-sz;
    l2 = ((d2y*d0z-d2z*d0y)*nx + (d2z*d0x-d2x*d0z)*ny + (d2x*d0y-d2y*d0x)*nz) / nn;
    l3 = Scalar(1) - l1 - l2;
  }

  int find_radial_layer(Scalar r) const {
    if (r < radii[0] || r > radii[N_r]) return -1;
    int lo = 0, hi = N_r - 1;
    while (lo < hi) { int mid = (lo+hi)/2; if (r < radii[mid+1]) hi = mid; else lo = mid+1; }
    return lo;
  }

  Scalar compute_zeta(int k, Scalar r) const {
    return (r - radii[k]) / (radii[k+1] - radii[k]);
  }

  int find_triangle(Scalar sx, Scalar sy, Scalar sz, int hint = 0) const {
    int t = (hint >= 0 && hint < N_tri) ? hint : 0;
    const int opp[3] = {1, 2, 0};
    for (int iter = 0; iter < N_tri; iter++) {
      Scalar l1, l2, l3;
      compute_barycentric(t, sx, sy, sz, l1, l2, l3);
      if (l1 >= Scalar(-1e-10) && l2 >= Scalar(-1e-10) && l3 >= Scalar(-1e-10))
        return t;
      Scalar lam[3] = {l1, l2, l3};
      int mi = 0;
      if (lam[1] < lam[mi]) mi = 1;
      if (lam[2] < lam[mi]) mi = 2;
      int next = tri_neighbor[t*3 + opp[mi]];
      if (next < 0) return t;
      t = next;
    }
    return t;
  }
};

// =========================================================================
// Whitney-form field interpolation (mirrors prismatic_deposit.h)
// =========================================================================
static void interp_fields(const mesh_data& mp, int tri, int layer,
                           const Scalar l[3], Scalar zeta,
                           const Scalar* E_e, const Scalar* B_f,
                           Scalar& Ex, Scalar& Ey, Scalar& Ez,
                           Scalar& Bx, Scalar& By, Scalar& Bz) {
  int edges[9];
  mp.prism_edge_indices(tri, layer, edges);
  int sv[3] = {mp.tri_verts[tri*3], mp.tri_verts[tri*3+1], mp.tri_verts[tri*3+2]};

  Scalar r_mid = Scalar(0.5) * (mp.radii[layer] + mp.radii[layer+1]);
  Scalar px[3], py[3], pz[3];
  for (int i = 0; i < 3; i++) {
    px[i] = r_mid * mp.sphere_vx[sv[i]];
    py[i] = r_mid * mp.sphere_vy[sv[i]];
    pz[i] = r_mid * mp.sphere_vz[sv[i]];
  }

  Scalar e1x=px[1]-px[0], e1y=py[1]-py[0], e1z=pz[1]-pz[0];
  Scalar e2x=px[2]-px[0], e2y=py[2]-py[0], e2z=pz[2]-pz[0];
  Scalar nx=e1y*e2z-e1z*e2y, ny=e1z*e2x-e1x*e2z, nz=e1x*e2y-e1y*e2x;
  Scalar nn = nx*nx+ny*ny+nz*nz;

  Scalar gl[3][3];
  for (int i = 0; i < 3; i++) {
    int j=(i+1)%3, k=(i+2)%3;
    Scalar dx=px[k]-px[j], dy=py[k]-py[j], dz=pz[k]-pz[j];
    gl[i][0]=(ny*dz-nz*dy)/nn; gl[i][1]=(nz*dx-nx*dz)/nn; gl[i][2]=(nx*dy-ny*dx)/nn;
  }

  Scalar rh_x=0, rh_y=0, rh_z=0;
  for (int i=0;i<3;i++) { rh_x+=l[i]*mp.sphere_vx[sv[i]]; rh_y+=l[i]*mp.sphere_vy[sv[i]]; rh_z+=l[i]*mp.sphere_vz[sv[i]]; }
  Scalar rn = std::sqrt(rh_x*rh_x+rh_y*rh_y+rh_z*rh_z);
  if (rn>0) { rh_x/=rn; rh_y/=rn; rh_z/=rn; }
  Scalar dr = mp.radii[layer+1]-mp.radii[layer];
  Scalar dz_x=rh_x/dr, dz_y=rh_y/dr, dz_z=rh_z/dr;

  Scalar phi[2] = {Scalar(1)-zeta, zeta};
  const int cf[3]={0,1,2}, ct[3]={1,2,0};

  Ex=Ey=Ez=0;
  for (int j=0;j<3;j++) {
    int fi=cf[j], ti=ct[j], sign=mp.tri_edge_signs[tri*3+j];
    Scalar wx=l[fi]*gl[ti][0]-l[ti]*gl[fi][0];
    Scalar wy=l[fi]*gl[ti][1]-l[ti]*gl[fi][1];
    Scalar wz=l[fi]*gl[ti][2]-l[ti]*gl[fi][2];
    for (int k=0;k<2;k++) { Scalar c=Scalar(sign)*E_e[edges[j+k*3]]*phi[k]; Ex+=c*wx; Ey+=c*wy; Ez+=c*wz; }
  }
  for (int i=0;i<3;i++) { Scalar c=E_e[edges[6+i]]*l[i]; Ex+=c*dz_x; Ey+=c*dz_y; Ez+=c*dz_z; }

  Bx=By=Bz=0;
  Scalar dl12_x=gl[0][1]*gl[1][2]-gl[0][2]*gl[1][1];
  Scalar dl12_y=gl[0][2]*gl[1][0]-gl[0][0]*gl[1][2];
  Scalar dl12_z=gl[0][0]*gl[1][1]-gl[0][1]*gl[1][0];
  for (int k=0;k<2;k++) { Scalar c=Scalar(2)*B_f[mp.tri_face_idx(layer+k,tri)]*phi[k]; Bx+=c*dl12_x; By+=c*dl12_y; Bz+=c*dl12_z; }
  for (int j=0;j<3;j++) {
    int fi=cf[j], ti=ct[j], sign=mp.tri_edge_signs[tri*3+j];
    int se=mp.tri_edges_s[tri*3+j];
    Scalar wx=l[fi]*gl[ti][0]-l[ti]*gl[fi][0];
    Scalar wy=l[fi]*gl[ti][1]-l[ti]*gl[fi][1];
    Scalar wz=l[fi]*gl[ti][2]-l[ti]*gl[fi][2];
    Scalar bx=wy*dz_z-wz*dz_y, by=wz*dz_x-wx*dz_z, bz=wx*dz_y-wy*dz_x;
    Scalar c=Scalar(sign)*B_f[mp.rect_face_idx(layer,se)];
    Bx+=c*bx; By+=c*by; Bz+=c*bz;
  }
}

// =========================================================================
// Double-precision sph-shell gathers (fast path for sph_from_dump.py).
//
// Mirrors prismatic_recovery.locate (central-projection barycentric via
// the SAME precomputed per-triangle inverse matrices, passed in from
// Python) and sph_from_dump.gather_at_shell (explicit layer/zeta per
// shell — part of the output definition).  All math in float64 so the
// results match the pure-Python path to storage precision.
// =========================================================================
namespace shellgather {

using D = double;

struct MeshD {
  int N_r, N_tri, N_vert_s, N_edge_s;
  const D* radii;           // (N_r+1)
  const D* sv;              // (N_vert_s, 3) unit sphere vertices
  const int* tri_verts;     // (N_tri, 3)
  const int* tri_edges_s;   // (N_tri, 3)
  const int* tri_edge_signs;// (N_tri, 3)
  const int* tri_neighbor;  // (N_tri, 3)
  const D* tri_vinv;        // (N_tri, 3, 3) = inv(V^T) per triangle

  int h_edge_idx(int k, int e) const { return k * N_edge_s + e; }
  int v_edge_idx(int k, int s) const {
    return (N_r + 1) * N_edge_s + k * N_vert_s + s;
  }
  int tri_face_idx(int k, int t) const { return k * N_tri + t; }
  int rect_face_idx(int k, int e) const {
    return (N_r + 1) * N_tri + k * N_edge_s + e;
  }
};

inline void bary_central(const MeshD& m, int t, const D p[3], D lam[3]) {
  const D* Vi = m.tri_vinv + 9 * t;
  D raw[3];
  for (int a = 0; a < 3; a++)
    raw[a] = Vi[3 * a] * p[0] + Vi[3 * a + 1] * p[1] + Vi[3 * a + 2] * p[2];
  D s = std::abs(raw[0] + raw[1] + raw[2]);
  lam[0] = raw[0] / s; lam[1] = raw[1] / s; lam[2] = raw[2] / s;
}

// Walk + clipped/renormalized lam; matches prismatic_recovery.locate.
inline int locate_one(const MeshD& m, const D p[3], int hint, D lam[3]) {
  static const int opp[3] = {1, 2, 0};
  int t = (hint >= 0 && hint < m.N_tri) ? hint : 0;
  bool found = false;
  for (int it = 0; it < m.N_tri; it++) {
    bary_central(m, t, p, lam);
    if (lam[0] >= -1e-12 && lam[1] >= -1e-12 && lam[2] >= -1e-12) {
      found = true;
      break;
    }
    int mi = 0;
    if (lam[1] < lam[mi]) mi = 1;
    if (lam[2] < lam[mi]) mi = 2;
    int nxt = m.tri_neighbor[3 * t + opp[mi]];
    if (nxt < 0) break;
    t = nxt;
  }
  if (!found) {  // global fallback, as in the Python locate
    D best = -1e300;
    int bt = 0;
    D bl[3];
    for (int tt = 0; tt < m.N_tri; tt++) {
      bary_central(m, tt, p, bl);
      D mn = std::min(bl[0], std::min(bl[1], bl[2]));
      if (mn > best) { best = mn; bt = tt; }
    }
    t = bt;
    bary_central(m, t, p, lam);
  }
  for (int a = 0; a < 3; a++) lam[a] = lam[a] < 0 ? 0 : lam[a];
  D s = lam[0] + lam[1] + lam[2];
  for (int a = 0; a < 3; a++) lam[a] /= s;
  return t;
}

// Whitney E and B at one angular location with explicit (layer k, zeta);
// line-by-line port of sph_from_dump.gather_at_shell.
inline void gather_one(const MeshD& m, int t, const D l[3], int k, D zeta,
                       const D* E_e, const D* B_f, D E[3], D B[3]) {
  const int sv3[3] = {m.tri_verts[3 * t], m.tri_verts[3 * t + 1],
                      m.tri_verts[3 * t + 2]};
  const D r_mid = D(0.5) * (m.radii[k] + m.radii[k + 1]);
  D p[3][3];
  for (int i = 0; i < 3; i++)
    for (int c = 0; c < 3; c++) p[i][c] = r_mid * m.sv[3 * sv3[i] + c];

  const D e1[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]};
  const D e2[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]};
  const D n[3] = {e1[1] * e2[2] - e1[2] * e2[1],
                  e1[2] * e2[0] - e1[0] * e2[2],
                  e1[0] * e2[1] - e1[1] * e2[0]};
  const D nn = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];

  D gl[3][3];
  for (int a = 0; a < 3; a++) {
    const int b = (a + 1) % 3, c = (a + 2) % 3;
    const D d[3] = {p[c][0] - p[b][0], p[c][1] - p[b][1], p[c][2] - p[b][2]};
    gl[a][0] = (n[1] * d[2] - n[2] * d[1]) / nn;
    gl[a][1] = (n[2] * d[0] - n[0] * d[2]) / nn;
    gl[a][2] = (n[0] * d[1] - n[1] * d[0]) / nn;
  }

  D rh[3] = {0, 0, 0};
  for (int a = 0; a < 3; a++)
    for (int c = 0; c < 3; c++) rh[c] += l[a] * m.sv[3 * sv3[a] + c];
  const D rn = std::sqrt(rh[0] * rh[0] + rh[1] * rh[1] + rh[2] * rh[2]);
  const D dr = m.radii[k + 1] - m.radii[k];
  D dz[3];
  for (int c = 0; c < 3; c++) dz[c] = rh[c] / (rn * dr);

  const D phi[2] = {D(1) - zeta, zeta};
  E[0] = E[1] = E[2] = 0;
  B[0] = B[1] = B[2] = 0;

  for (int j = 0; j < 3; j++) {
    const int fi = j, ti = (j + 1) % 3;
    const int sign = m.tri_edge_signs[3 * t + j];
    const int se = m.tri_edges_s[3 * t + j];
    D w[3];
    for (int c = 0; c < 3; c++) w[c] = l[fi] * gl[ti][c] - l[ti] * gl[fi][c];
    for (int kk = 0; kk < 2; kk++) {
      const D cc = D(sign) * E_e[m.h_edge_idx(k + kk, se)] * phi[kk];
      for (int c = 0; c < 3; c++) E[c] += cc * w[c];
    }
    const D cb = D(sign) * B_f[m.rect_face_idx(k, se)];
    B[0] += cb * (w[1] * dz[2] - w[2] * dz[1]);
    B[1] += cb * (w[2] * dz[0] - w[0] * dz[2]);
    B[2] += cb * (w[0] * dz[1] - w[1] * dz[0]);
  }
  for (int a = 0; a < 3; a++) {
    const D cc = E_e[m.v_edge_idx(k, sv3[a])] * l[a];
    for (int c = 0; c < 3; c++) E[c] += cc * dz[c];
  }
  const D dl12[3] = {gl[0][1] * gl[1][2] - gl[0][2] * gl[1][1],
                     gl[0][2] * gl[1][0] - gl[0][0] * gl[1][2],
                     gl[0][0] * gl[1][1] - gl[0][1] * gl[1][0]};
  for (int kk = 0; kk < 2; kk++) {
    const D cc = D(2) * B_f[m.tri_face_idx(k + kk, t)] * phi[kk];
    for (int c = 0; c < 3; c++) B[c] += cc * dl12[c];
  }
}

inline MeshD unpack_mesh(int N_r, int N_tri, int N_vert_s, int N_edge_s,
                         const py::array_t<D>& radii,
                         const py::array_t<D>& sphere_v,
                         const py::array_t<int>& tri_verts,
                         const py::array_t<int>& tri_edges_s,
                         const py::array_t<int>& tri_edge_signs,
                         const py::array_t<int>& tri_neighbor,
                         const py::array_t<D>& tri_vinv) {
  MeshD m;
  m.N_r = N_r; m.N_tri = N_tri;
  m.N_vert_s = N_vert_s; m.N_edge_s = N_edge_s;
  m.radii = radii.data();
  m.sv = sphere_v.data();
  m.tri_verts = tri_verts.data();
  m.tri_edges_s = tri_edges_s.data();
  m.tri_edge_signs = tri_edge_signs.data();
  m.tri_neighbor = tri_neighbor.data();
  m.tri_vinv = tri_vinv.data();
  return m;
}

}  // namespace shellgather

// =========================================================================
// Python module
// =========================================================================
PYBIND11_MODULE(prismatic_interp, m) {
  m.doc() = "Whitney-form field interpolation on the prismatic mesh";

  m.def("locate_points",
    [](int N_r, int N_tri, int N_vert_s, int N_edge_s,
       py::array_t<double> radii, py::array_t<double> sphere_v,
       py::array_t<int> tri_verts, py::array_t<int> tri_edges_s,
       py::array_t<int> tri_edge_signs, py::array_t<int> tri_neighbor,
       py::array_t<double> tri_vinv, py::array_t<double> s_hat) {
      auto mp = shellgather::unpack_mesh(
          N_r, N_tri, N_vert_s, N_edge_s, radii, sphere_v, tri_verts,
          tri_edges_s, tri_edge_signs, tri_neighbor, tri_vinv);
      auto pts = s_hat.unchecked<2>();
      const int N = pts.shape(0);
      py::array_t<int> tri_out(N);
      py::array_t<double> lam_out({N, 3});
      auto tri = tri_out.mutable_unchecked<1>();
      auto lam = lam_out.mutable_unchecked<2>();
      {
        py::gil_scoped_release rel;
        int hint = 0;
        for (int i = 0; i < N; i++) {
          const double p[3] = {pts(i, 0), pts(i, 1), pts(i, 2)};
          double l[3];
          hint = shellgather::locate_one(mp, p, hint, l);
          tri(i) = hint;
          lam(i, 0) = l[0]; lam(i, 1) = l[1]; lam(i, 2) = l[2];
        }
      }
      return py::make_tuple(tri_out, lam_out);
    },
    "Locate unit directions on the sphere: (tri, lam) with the "
    "central-projection barycentric convention of prismatic_recovery."
    "locate.  tri_vinv = inv(V^T) per triangle, as cached by the Python "
    "Mesh (_tri_vinv).");

  m.def("gather_shells",
    [](int N_r, int N_tri, int N_vert_s, int N_edge_s,
       py::array_t<double> radii, py::array_t<double> sphere_v,
       py::array_t<int> tri_verts, py::array_t<int> tri_edges_s,
       py::array_t<int> tri_edge_signs, py::array_t<int> tri_neighbor,
       py::array_t<double> tri_vinv,
       py::array_t<int> tri_pts, py::array_t<double> lam_pts,
       py::array_t<int> layers, py::array_t<double> zetas,
       py::array_t<double> E_e, py::array_t<double> B_f) {
      auto mp = shellgather::unpack_mesh(
          N_r, N_tri, N_vert_s, N_edge_s, radii, sphere_v, tri_verts,
          tri_edges_s, tri_edge_signs, tri_neighbor, tri_vinv);
      auto tri = tri_pts.unchecked<1>();
      auto lam = lam_pts.unchecked<2>();
      auto lay = layers.unchecked<1>();
      auto zet = zetas.unchecked<1>();
      const int N = tri.shape(0);
      const int NS = lay.shape(0);
      py::array_t<double> E_out({NS, N, 3}), B_out({NS, N, 3});
      auto Eo = E_out.mutable_unchecked<3>();
      auto Bo = B_out.mutable_unchecked<3>();
      const double* ee = E_e.data();
      const double* bf = B_f.data();
      {
        py::gil_scoped_release rel;
        for (int s = 0; s < NS; s++) {
          const int k = lay(s);
          const double z = zet(s);
          for (int i = 0; i < N; i++) {
            const double l[3] = {lam(i, 0), lam(i, 1), lam(i, 2)};
            double E[3], B[3];
            shellgather::gather_one(mp, tri(i), l, k, z, ee, bf, E, B);
            for (int c = 0; c < 3; c++) {
              Eo(s, i, c) = E[c];
              Bo(s, i, c) = B[c];
            }
          }
        }
      }
      return py::make_tuple(E_out, B_out);
    },
    "Whitney E and B at fixed angular points (tri, lam) for a list of "
    "shells with explicit (layer, zeta) — the sph_from_dump.gather_at_"
    "shell convention.  Returns (NS, N, 3) float64 arrays.");

  m.def("interpolate_slice",
    [](int N_r, int N_tri, int N_vert_s, int N_edge_s,
       int N_verts, int N_edges, int N_faces,
       py::array_t<Scalar> radii_arr,
       py::array_t<Scalar> sphere_vx_arr, py::array_t<Scalar> sphere_vy_arr,
       py::array_t<Scalar> sphere_vz_arr,
       py::array_t<int> tri_verts_arr, py::array_t<int> tri_edges_s_arr,
       py::array_t<int> tri_edge_signs_arr, py::array_t<int> tri_neighbor_arr,
       py::array_t<Scalar> x_arr, py::array_t<Scalar> z_arr,
       py::array_t<Scalar> E_e_arr, py::array_t<Scalar> B_f_arr) {

      mesh_data mp;
      mp.N_r = N_r; mp.N_tri = N_tri; mp.N_vert_s = N_vert_s;
      mp.N_edge_s = N_edge_s; mp.N_verts = N_verts;
      mp.N_edges = N_edges; mp.N_faces = N_faces;
      mp.radii = radii_arr.data();
      mp.sphere_vx = sphere_vx_arr.data();
      mp.sphere_vy = sphere_vy_arr.data();
      mp.sphere_vz = sphere_vz_arr.data();
      mp.tri_verts = tri_verts_arr.data();
      mp.tri_edges_s = tri_edges_s_arr.data();
      mp.tri_edge_signs = tri_edge_signs_arr.data();
      mp.tri_neighbor = tri_neighbor_arr.data();

      auto x = x_arr.unchecked<1>();
      auto z = z_arr.unchecked<1>();
      int N = x.shape(0);

      py::array_t<Scalar> Bx_out(N), By_out(N), Bz_out(N);
      py::array_t<Scalar> Ex_out(N), Ey_out(N), Ez_out(N);
      auto Bx=Bx_out.mutable_unchecked<1>(), By=By_out.mutable_unchecked<1>(), Bz=Bz_out.mutable_unchecked<1>();
      auto Ex=Ex_out.mutable_unchecked<1>(), Ey=Ey_out.mutable_unchecked<1>(), Ez=Ez_out.mutable_unchecked<1>();

      int tri_hint = 0;
      for (int i = 0; i < N; i++) {
        Scalar xi = x(i), zi = z(i);
        Scalar r = std::sqrt(xi*xi + zi*zi);
        int layer = mp.find_radial_layer(r);
        if (layer < 0) { Bx(i)=By(i)=Bz(i)=Ex(i)=Ey(i)=Ez(i)=0; continue; }
        Scalar ri = Scalar(1)/r;
        int tri = mp.find_triangle(xi*ri, 0, zi*ri, tri_hint);
        tri_hint = tri;
        Scalar l1,l2,l3;
        mp.compute_barycentric(tri, xi*ri, 0, zi*ri, l1, l2, l3);
        Scalar zeta = mp.compute_zeta(layer, r);
        Scalar l[3] = {l1, l2, l3};
        Scalar iEx,iEy,iEz,iBx,iBy,iBz;
        interp_fields(mp, tri, layer, l, zeta, E_e_arr.data(), B_f_arr.data(),
                      iEx, iEy, iEz, iBx, iBy, iBz);
        Bx(i)=iBx; By(i)=iBy; Bz(i)=iBz;
        Ex(i)=iEx; Ey(i)=iEy; Ez(i)=iEz;
      }
      return py::make_tuple(Bx_out, By_out, Bz_out, Ex_out, Ey_out, Ez_out);
    },
    "Interpolate E, B on (x,z) points in the y=0 plane using Whitney forms.\n"
    "Pass mesh arrays from mesh.h5 and field arrays from a snapshot.");
}
