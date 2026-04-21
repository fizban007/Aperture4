#include "systems/prismatic/prismatic_mesh_local.h"
#include "systems/prismatic/prismatic_cochain_layout.h"
#include "systems/prismatic/prismatic_mesh.h"

namespace Aperture {

namespace {

// Copy a global buffer's local portion into `dst`, following `layout`.
// The base pointer is offset into the mesh's flat storage so that
// `layout.to_global(l)` addresses the right cell type (h or v edge,
// tri or rect face).
template <typename T>
void copy_with_offset(const T* global_base, const distributed_cochain_layout& layout,
                       std::vector<T>& dst) {
  const int n = layout.local_size();
  dst.resize(n);
  for (int l = 0; l < n; ++l) {
    dst[l] = global_base[layout.to_global(l)];
  }
}

}  // namespace

prismatic_mesh_local prismatic_mesh_local::build(
    const prismatic_mesh& mesh, const prismatic_mesh_partition& mp) {
  prismatic_mesh_local out;
  out.m_partition = &mp;

  // Base pointers into the mesh's flat storage.
  //   Face buffers:  tri faces at [0, N_tri_faces),
  //                  rect faces at [N_tri_faces, N_faces).
  //   Edge buffers:  h edges   at [0, N_h_edges),
  //                  v edges   at [N_h_edges, N_edges).
  //   Vertex buffers are already a single block of size N_verts.
  const int N_tri_faces = (mesh.m_N_r + 1) * mesh.m_N_tri;
  const int N_h_edges   = (mesh.m_N_r + 1) * mesh.m_N_edge_s;

  auto const& L_tri  = mp.layout(cochain_type::tri_face);
  auto const& L_rect = mp.layout(cochain_type::rect_face);
  auto const& L_he   = mp.layout(cochain_type::h_edge);
  auto const& L_ve   = mp.layout(cochain_type::v_edge);
  auto const& L_vert = mp.layout(cochain_type::vertex);

  // ---- Tri-face buffers (base offset = 0) ----
  copy_with_offset(mesh.face_area.host_ptr(),         L_tri, out.tri_face_area);
  copy_with_offset(mesh.hodge2.host_ptr(),            L_tri, out.tri_face_hodge2);
  copy_with_offset(mesh.face_boundary.host_ptr(),     L_tri, out.tri_face_boundary);
  copy_with_offset(mesh.face_radial_layer.host_ptr(), L_tri, out.tri_face_radial_layer);

  // ---- Rect-face buffers (base offset = N_tri_faces) ----
  copy_with_offset(mesh.face_area.host_ptr()         + N_tri_faces, L_rect, out.rect_face_area);
  copy_with_offset(mesh.hodge2.host_ptr()            + N_tri_faces, L_rect, out.rect_face_hodge2);
  copy_with_offset(mesh.face_boundary.host_ptr()     + N_tri_faces, L_rect, out.rect_face_boundary);
  copy_with_offset(mesh.face_radial_layer.host_ptr() + N_tri_faces, L_rect, out.rect_face_radial_layer);

  // ---- H-edge buffers (base offset = 0) ----
  copy_with_offset(mesh.edge_length.host_ptr(),       L_he, out.h_edge_length);
  copy_with_offset(mesh.hodge1_inv.host_ptr(),        L_he, out.h_edge_hodge1_inv);
  copy_with_offset(mesh.edge_v0.host_ptr(),           L_he, out.h_edge_v0);
  copy_with_offset(mesh.edge_v1.host_ptr(),           L_he, out.h_edge_v1);
  copy_with_offset(mesh.edge_boundary.host_ptr(),     L_he, out.h_edge_boundary);
  copy_with_offset(mesh.edge_radial_layer.host_ptr(), L_he, out.h_edge_radial_layer);

  // ---- V-edge buffers (base offset = N_h_edges) ----
  copy_with_offset(mesh.edge_length.host_ptr()       + N_h_edges, L_ve, out.v_edge_length);
  copy_with_offset(mesh.hodge1_inv.host_ptr()        + N_h_edges, L_ve, out.v_edge_hodge1_inv);
  copy_with_offset(mesh.edge_v0.host_ptr()           + N_h_edges, L_ve, out.v_edge_v0);
  copy_with_offset(mesh.edge_v1.host_ptr()           + N_h_edges, L_ve, out.v_edge_v1);
  copy_with_offset(mesh.edge_boundary.host_ptr()     + N_h_edges, L_ve, out.v_edge_boundary);
  copy_with_offset(mesh.edge_radial_layer.host_ptr() + N_h_edges, L_ve, out.v_edge_radial_layer);

  // ---- Vertex buffers (base offset = 0) ----
  copy_with_offset(mesh.vert_r.host_ptr(),     L_vert, out.vert_r);
  copy_with_offset(mesh.vert_theta.host_ptr(), L_vert, out.vert_theta);
  copy_with_offset(mesh.vert_phi.host_ptr(),   L_vert, out.vert_phi);

  return out;
}

}  // namespace Aperture
