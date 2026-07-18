#include "systems/prismatic/prismatic_mesh_local_ptrs.h"

namespace Aperture {

namespace {

// PTR(x) resolves member x from either host or device side via the
// accessor passed in; keeps the two assembly functions in lockstep.
template <typename GetS, typename GetI>
prismatic_mesh_local_ptrs assemble(const prismatic_mesh_local& ml,
                                   const prismatic_d1_local& d1,
                                   GetS s, GetI i) {
  prismatic_mesh_local_ptrs p;
  auto const& mp = ml.partition();
  auto const& L_tri = mp.layout(cochain_type::tri_face);
  auto const& L_rect = mp.layout(cochain_type::rect_face);
  auto const& L_he = mp.layout(cochain_type::h_edge);
  auto const& L_ve = mp.layout(cochain_type::v_edge);
  auto const& L_v = mp.layout(cochain_type::vertex);
  p.n_owned_tri = L_tri.owned_size();
  p.n_local_tri = L_tri.local_size();
  p.n_owned_rect = L_rect.owned_size();
  p.n_local_rect = L_rect.local_size();
  p.n_owned_he = L_he.owned_size();
  p.n_local_he = L_he.local_size();
  p.n_owned_ve = L_ve.owned_size();
  p.n_local_ve = L_ve.local_size();
  p.n_owned_v = L_v.owned_size();
  p.n_local_v = L_v.local_size();

  p.tri_face_area = s(ml.tri_face_area);
  p.tri_face_hodge2 = s(ml.tri_face_hodge2);
  p.tri_face_boundary = i(ml.tri_face_boundary);
  p.tri_face_radial_layer = i(ml.tri_face_radial_layer);
  p.rect_face_area = s(ml.rect_face_area);
  p.rect_face_hodge2 = s(ml.rect_face_hodge2);
  p.rect_face_boundary = i(ml.rect_face_boundary);
  p.rect_face_radial_layer = i(ml.rect_face_radial_layer);

  p.h_edge_length = s(ml.h_edge_length);
  p.h_edge_hodge1_inv = s(ml.h_edge_hodge1_inv);
  p.h_edge_v0 = i(ml.h_edge_v0);
  p.h_edge_v1 = i(ml.h_edge_v1);
  p.h_edge_boundary = i(ml.h_edge_boundary);
  p.h_edge_radial_layer = i(ml.h_edge_radial_layer);
  p.v_edge_length = s(ml.v_edge_length);
  p.v_edge_hodge1_inv = s(ml.v_edge_hodge1_inv);
  p.v_edge_v0 = i(ml.v_edge_v0);
  p.v_edge_v1 = i(ml.v_edge_v1);
  p.v_edge_boundary = i(ml.v_edge_boundary);
  p.v_edge_radial_layer = i(ml.v_edge_radial_layer);

  p.vert_r = s(ml.vert_r);
  p.vert_theta = s(ml.vert_theta);
  p.vert_phi = s(ml.vert_phi);

  p.d1_tri_h_row = i(d1.d1_tri_h.row_ptr);
  p.d1_tri_h_col = i(d1.d1_tri_h.col_idx);
  p.d1_tri_h_val = s(d1.d1_tri_h.val);
  p.d1_rect_h_row = i(d1.d1_rect_h.row_ptr);
  p.d1_rect_h_col = i(d1.d1_rect_h.col_idx);
  p.d1_rect_h_val = s(d1.d1_rect_h.val);
  p.d1_rect_v_row = i(d1.d1_rect_v.row_ptr);
  p.d1_rect_v_col = i(d1.d1_rect_v.col_idx);
  p.d1_rect_v_val = s(d1.d1_rect_v.val);
  p.d1t_h_tri_row = i(d1.d1t_h_tri.row_ptr);
  p.d1t_h_tri_col = i(d1.d1t_h_tri.col_idx);
  p.d1t_h_tri_val = s(d1.d1t_h_tri.val);
  p.d1t_h_rect_row = i(d1.d1t_h_rect.row_ptr);
  p.d1t_h_rect_col = i(d1.d1t_h_rect.col_idx);
  p.d1t_h_rect_val = s(d1.d1t_h_rect.val);
  p.d1t_v_rect_row = i(d1.d1t_v_rect.row_ptr);
  p.d1t_v_rect_col = i(d1.d1t_v_rect.col_idx);
  p.d1t_v_rect_val = s(d1.d1t_v_rect.val);
  return p;
}

}  // namespace

prismatic_mesh_local_ptrs make_local_ptrs_host(
    const prismatic_mesh_local& ml, const prismatic_d1_local& d1) {
  return assemble(ml, d1,
                  [](const buffer<Scalar>& b) { return b.host_ptr(); },
                  [](const buffer<int>& b) { return b.host_ptr(); });
}

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
prismatic_mesh_local_ptrs make_local_ptrs_dev(
    const prismatic_mesh_local& ml, const prismatic_d1_local& d1) {
  return assemble(ml, d1,
                  [](const buffer<Scalar>& b) { return b.dev_ptr(); },
                  [](const buffer<int>& b) { return b.dev_ptr(); });
}
#endif

}  // namespace Aperture
