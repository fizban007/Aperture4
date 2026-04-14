#include "systems/prismatic/prismatic_mesh_gr_ks.h"
#include "systems/physics/metric_ks_cartesian.hpp"

namespace Aperture {

void prismatic_mesh_gr_ks::compute_metric(Scalar a) {
  m_a = a;

  // --- Allocate per-edge arrays ---
  edge_f.resize(m_N_edges);
  edge_lx.resize(m_N_edges);
  edge_ly.resize(m_N_edges);
  edge_lz.resize(m_N_edges);
  edge_alpha.resize(m_N_edges);
  edge_sgb_x.resize(m_N_edges);
  edge_sgb_y.resize(m_N_edges);
  edge_sgb_z.resize(m_N_edges);
  edge_r.resize(m_N_edges);

  // --- Allocate per-face arrays ---
  face_f.resize(m_N_faces);
  face_lx.resize(m_N_faces);
  face_ly.resize(m_N_faces);
  face_lz.resize(m_N_faces);
  face_alpha.resize(m_N_faces);
  face_sgb_x.resize(m_N_faces);
  face_sgb_y.resize(m_N_faces);
  face_sgb_z.resize(m_N_faces);
  face_r.resize(m_N_faces);

  // --- Compute per-edge metric at edge midpoints ---
  for (int e = 0; e < m_N_edges; e++) {
    int v0 = edge_v0[e];
    int v1 = edge_v1[e];
    Scalar mx = Scalar(0.5) * (vert_x[v0] + vert_x[v1]);
    Scalar my = Scalar(0.5) * (vert_y[v0] + vert_y[v1]);
    Scalar mz = Scalar(0.5) * (vert_z[v0] + vert_z[v1]);

    Scalar r, fv, alp, sg;
    vec_t<Scalar, 3> l, bu, sgb;
    Metric_KS_Cart::compute_all(mx, my, mz, a, r, l, fv, alp, bu, sgb, sg);

    edge_f[e] = fv;
    edge_lx[e] = l[0];
    edge_ly[e] = l[1];
    edge_lz[e] = l[2];
    edge_alpha[e] = alp;
    edge_sgb_x[e] = sgb[0];
    edge_sgb_y[e] = sgb[1];
    edge_sgb_z[e] = sgb[2];
    edge_r[e] = r;
  }

  // --- Compute per-face metric at face centroids ---

  // Triangular faces: centroid = average of 3 vertices
  int n_tri_faces = m_N_tri * (m_N_r + 1);
  for (int fi = 0; fi < n_tri_faces; fi++) {
    int va = tri_face_v0[fi];
    int vb = tri_face_v1[fi];
    int vc = tri_face_v2[fi];
    Scalar cx = (vert_x[va] + vert_x[vb] + vert_x[vc]) / Scalar(3.0);
    Scalar cy = (vert_y[va] + vert_y[vb] + vert_y[vc]) / Scalar(3.0);
    Scalar cz = (vert_z[va] + vert_z[vb] + vert_z[vc]) / Scalar(3.0);

    Scalar r, fv, alp, sg;
    vec_t<Scalar, 3> l, bu, sgb;
    Metric_KS_Cart::compute_all(cx, cy, cz, a, r, l, fv, alp, bu, sgb, sg);

    face_f[fi] = fv;
    face_lx[fi] = l[0];
    face_ly[fi] = l[1];
    face_lz[fi] = l[2];
    face_alpha[fi] = alp;
    face_sgb_x[fi] = sgb[0];
    face_sgb_y[fi] = sgb[1];
    face_sgb_z[fi] = sgb[2];
    face_r[fi] = r;
  }

  // Rectangular faces: centroid = average of 4 vertices
  int n_rect_faces = m_N_edge_s * m_N_r;
  for (int ri = 0; ri < n_rect_faces; ri++) {
    int fi = n_tri_faces + ri;  // global face index
    int va = rect_face_v0[ri];
    int vb = rect_face_v1[ri];
    int vc = rect_face_v2[ri];
    int vd = rect_face_v3[ri];
    Scalar cx = Scalar(0.25) * (vert_x[va] + vert_x[vb] + vert_x[vc] + vert_x[vd]);
    Scalar cy = Scalar(0.25) * (vert_y[va] + vert_y[vb] + vert_y[vc] + vert_y[vd]);
    Scalar cz = Scalar(0.25) * (vert_z[va] + vert_z[vb] + vert_z[vc] + vert_z[vd]);

    Scalar r, fv, alp, sg;
    vec_t<Scalar, 3> l, bu, sgb;
    Metric_KS_Cart::compute_all(cx, cy, cz, a, r, l, fv, alp, bu, sgb, sg);

    face_f[fi] = fv;
    face_lx[fi] = l[0];
    face_ly[fi] = l[1];
    face_lz[fi] = l[2];
    face_alpha[fi] = alp;
    face_sgb_x[fi] = sgb[0];
    face_sgb_y[fi] = sgb[1];
    face_sgb_z[fi] = sgb[2];
    face_r[fi] = r;
  }
}

// Helper: fill the GR metric pointers from a buffer accessor (host or dev)
template <typename Accessor>
static void fill_gr_ptrs(prismatic_mesh_gr_ks_ptrs& p,
                         const prismatic_mesh_gr_ks& m, Accessor acc) {
  p.edge_f     = acc(m.edge_f);
  p.edge_lx    = acc(m.edge_lx);
  p.edge_ly    = acc(m.edge_ly);
  p.edge_lz    = acc(m.edge_lz);
  p.edge_alpha = acc(m.edge_alpha);
  p.edge_sgb_x = acc(m.edge_sgb_x);
  p.edge_sgb_y = acc(m.edge_sgb_y);
  p.edge_sgb_z = acc(m.edge_sgb_z);
  p.edge_r     = acc(m.edge_r);

  p.face_f     = acc(m.face_f);
  p.face_lx    = acc(m.face_lx);
  p.face_ly    = acc(m.face_ly);
  p.face_lz    = acc(m.face_lz);
  p.face_alpha = acc(m.face_alpha);
  p.face_sgb_x = acc(m.face_sgb_x);
  p.face_sgb_y = acc(m.face_sgb_y);
  p.face_sgb_z = acc(m.face_sgb_z);
  p.face_r     = acc(m.face_r);
}

prismatic_mesh_gr_ks_ptrs prismatic_mesh_gr_ks::host_ptrs_gr() const {
  prismatic_mesh_gr_ks_ptrs p{};
  // Fill base class fields
  static_cast<prismatic_mesh_ptrs&>(p) = prismatic_mesh::host_ptrs();
  // Fill GR metric fields
  auto host_acc = [](const auto& buf) { return buf.host_ptr(); };
  fill_gr_ptrs(p, *this, host_acc);
  return p;
}

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
prismatic_mesh_gr_ks_ptrs prismatic_mesh_gr_ks::dev_ptrs_gr() const {
  prismatic_mesh_gr_ks_ptrs p{};
  static_cast<prismatic_mesh_ptrs&>(p) = prismatic_mesh::dev_ptrs();
  auto dev_acc = [](const auto& buf) { return buf.dev_ptr(); };
  fill_gr_ptrs(p, *this, dev_acc);
  return p;
}

void prismatic_mesh_gr_ks::copy_to_device() {
  prismatic_mesh::copy_to_device();

  auto copy = [](auto& buf) { buf.copy_to_device(); };
  copy(edge_f); copy(edge_lx); copy(edge_ly); copy(edge_lz);
  copy(edge_alpha);
  copy(edge_sgb_x); copy(edge_sgb_y); copy(edge_sgb_z);
  copy(edge_r);

  copy(face_f); copy(face_lx); copy(face_ly); copy(face_lz);
  copy(face_alpha);
  copy(face_sgb_x); copy(face_sgb_y); copy(face_sgb_z);
  copy(face_r);
}
#endif

}  // namespace Aperture
