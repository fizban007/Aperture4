#include "systems/prismatic/prismatic_mesh_metric_local.h"
#include "systems/prismatic/prismatic_mesh_metric.h"

namespace Aperture {

namespace {

template <typename T>
void copy_with_offset(const T* base, const distributed_cochain_layout& layout,
                       std::vector<T>& dst) {
  const int n = layout.local_size();
  dst.resize(n);
  for (int l = 0; l < n; ++l) {
    dst[l] = base[layout.to_global(l)];
  }
}

}  // namespace

prismatic_mesh_metric_local prismatic_mesh_metric_local::build(
    const prismatic_mesh_metric& m, const prismatic_mesh_partition& mp) {
  prismatic_mesh_metric_local out;
  out.m_partition = &mp;

  const int N_tri_faces = (m.m_N_r + 1) * m.m_N_tri;
  const int N_h_edges   = (m.m_N_r + 1) * m.m_N_edge_s;

  auto const& L_tri  = mp.layout(cochain_type::tri_face);
  auto const& L_rect = mp.layout(cochain_type::rect_face);
  auto const& L_he   = mp.layout(cochain_type::h_edge);
  auto const& L_ve   = mp.layout(cochain_type::v_edge);

  // ---- Tri faces (offset 0 in face_*) ----
  copy_with_offset(m.face_r_coord.host_ptr(),         L_tri, out.tri_face_r_coord);
  copy_with_offset(m.face_sth.host_ptr(),             L_tri, out.tri_face_sth);
  copy_with_offset(m.face_cth.host_ptr(),             L_tri, out.tri_face_cth);
  copy_with_offset(m.face_alpha.host_ptr(),           L_tri, out.tri_face_alpha);
  copy_with_offset(m.face_sq_gamma_beta_r.host_ptr(), L_tri, out.tri_face_sq_gamma_beta_r);
  copy_with_offset(m.face_sqrt_gamma.host_ptr(),      L_tri, out.tri_face_sqrt_gamma);

  // ---- Rect faces (offset N_tri_faces) ----
  copy_with_offset(m.face_r_coord.host_ptr()         + N_tri_faces, L_rect, out.rect_face_r_coord);
  copy_with_offset(m.face_sth.host_ptr()             + N_tri_faces, L_rect, out.rect_face_sth);
  copy_with_offset(m.face_cth.host_ptr()             + N_tri_faces, L_rect, out.rect_face_cth);
  copy_with_offset(m.face_alpha.host_ptr()           + N_tri_faces, L_rect, out.rect_face_alpha);
  copy_with_offset(m.face_sq_gamma_beta_r.host_ptr() + N_tri_faces, L_rect, out.rect_face_sq_gamma_beta_r);
  copy_with_offset(m.face_sqrt_gamma.host_ptr()      + N_tri_faces, L_rect, out.rect_face_sqrt_gamma);

  // ---- H edges (offset 0 in edge_*) ----
  copy_with_offset(m.edge_r_coord.host_ptr(),         L_he, out.h_edge_r_coord);
  copy_with_offset(m.edge_sth.host_ptr(),             L_he, out.h_edge_sth);
  copy_with_offset(m.edge_cth.host_ptr(),             L_he, out.h_edge_cth);
  copy_with_offset(m.edge_alpha.host_ptr(),           L_he, out.h_edge_alpha);
  copy_with_offset(m.edge_sq_gamma_beta_r.host_ptr(), L_he, out.h_edge_sq_gamma_beta_r);
  copy_with_offset(m.edge_sqrt_gamma.host_ptr(),      L_he, out.h_edge_sqrt_gamma);

  // ---- V edges (offset N_h_edges) ----
  copy_with_offset(m.edge_r_coord.host_ptr()         + N_h_edges, L_ve, out.v_edge_r_coord);
  copy_with_offset(m.edge_sth.host_ptr()             + N_h_edges, L_ve, out.v_edge_sth);
  copy_with_offset(m.edge_cth.host_ptr()             + N_h_edges, L_ve, out.v_edge_cth);
  copy_with_offset(m.edge_alpha.host_ptr()           + N_h_edges, L_ve, out.v_edge_alpha);
  copy_with_offset(m.edge_sq_gamma_beta_r.host_ptr() + N_h_edges, L_ve, out.v_edge_sq_gamma_beta_r);
  copy_with_offset(m.edge_sqrt_gamma.host_ptr()      + N_h_edges, L_ve, out.v_edge_sqrt_gamma);

  return out;
}

}  // namespace Aperture
