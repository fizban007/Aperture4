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
                       buffer<T>& dst, MemType mem_type) {
  const int n = layout.local_size();
  dst.set_memtype(mem_type);
  dst.resize(n);
  for (int l = 0; l < n; ++l) {
    dst[l] = global_base[layout.to_global(l)];
  }
}

}  // namespace

prismatic_mesh_local prismatic_mesh_local::build(
    const prismatic_mesh& mesh, const prismatic_mesh_partition& mp,
    MemType mem_type) {
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
  copy_with_offset(mesh.face_area.host_ptr(),         L_tri, out.tri_face_area, mem_type);
  copy_with_offset(mesh.hodge2.host_ptr(),            L_tri, out.tri_face_hodge2, mem_type);
  copy_with_offset(mesh.face_boundary.host_ptr(),     L_tri, out.tri_face_boundary, mem_type);
  copy_with_offset(mesh.face_radial_layer.host_ptr(), L_tri, out.tri_face_radial_layer, mem_type);

  // ---- Rect-face buffers (base offset = N_tri_faces) ----
  copy_with_offset(mesh.face_area.host_ptr()         + N_tri_faces, L_rect, out.rect_face_area, mem_type);
  copy_with_offset(mesh.hodge2.host_ptr()            + N_tri_faces, L_rect, out.rect_face_hodge2, mem_type);
  copy_with_offset(mesh.face_boundary.host_ptr()     + N_tri_faces, L_rect, out.rect_face_boundary, mem_type);
  copy_with_offset(mesh.face_radial_layer.host_ptr() + N_tri_faces, L_rect, out.rect_face_radial_layer, mem_type);

  // ---- H-edge buffers (base offset = 0) ----
  copy_with_offset(mesh.edge_length.host_ptr(),       L_he, out.h_edge_length, mem_type);
  copy_with_offset(mesh.hodge1_inv.host_ptr(),        L_he, out.h_edge_hodge1_inv, mem_type);
  copy_with_offset(mesh.edge_v0.host_ptr(),           L_he, out.h_edge_v0, mem_type);
  copy_with_offset(mesh.edge_v1.host_ptr(),           L_he, out.h_edge_v1, mem_type);
  copy_with_offset(mesh.edge_boundary.host_ptr(),     L_he, out.h_edge_boundary, mem_type);
  copy_with_offset(mesh.edge_radial_layer.host_ptr(), L_he, out.h_edge_radial_layer, mem_type);

  // ---- V-edge buffers (base offset = N_h_edges) ----
  copy_with_offset(mesh.edge_length.host_ptr()       + N_h_edges, L_ve, out.v_edge_length, mem_type);
  copy_with_offset(mesh.hodge1_inv.host_ptr()        + N_h_edges, L_ve, out.v_edge_hodge1_inv, mem_type);
  copy_with_offset(mesh.edge_v0.host_ptr()           + N_h_edges, L_ve, out.v_edge_v0, mem_type);
  copy_with_offset(mesh.edge_v1.host_ptr()           + N_h_edges, L_ve, out.v_edge_v1, mem_type);
  copy_with_offset(mesh.edge_boundary.host_ptr()     + N_h_edges, L_ve, out.v_edge_boundary, mem_type);
  copy_with_offset(mesh.edge_radial_layer.host_ptr() + N_h_edges, L_ve, out.v_edge_radial_layer, mem_type);

  // ---- Vertex buffers (base offset = 0) ----
  copy_with_offset(mesh.vert_r.host_ptr(),     L_vert, out.vert_r, mem_type);
  copy_with_offset(mesh.vert_theta.host_ptr(), L_vert, out.vert_theta, mem_type);
  copy_with_offset(mesh.vert_phi.host_ptr(),   L_vert, out.vert_phi, mem_type);

  // ---- Local -> global maps ----
  auto fill_l2g = [mem_type](const distributed_cochain_layout& layout,
                             buffer<int>& dst) {
    const int n = layout.local_size();
    dst.set_memtype(mem_type);
    dst.resize(n);
    for (int l = 0; l < n; ++l) dst[l] = layout.to_global(l);
  };
  fill_l2g(L_tri,  out.tri_face_l2g);
  fill_l2g(L_rect, out.rect_face_l2g);
  fill_l2g(L_he,   out.h_edge_l2g);
  fill_l2g(L_ve,   out.v_edge_l2g);

  return out;
}

void prismatic_mesh_local::copy_to_device() {
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  auto copy = [](auto& b) {
    if (b.mem_type() != MemType::host_only) b.copy_to_device();
  };
  copy(tri_face_area); copy(tri_face_hodge2);
  copy(tri_face_boundary); copy(tri_face_radial_layer);
  copy(rect_face_area); copy(rect_face_hodge2);
  copy(rect_face_boundary); copy(rect_face_radial_layer);
  copy(h_edge_length); copy(h_edge_hodge1_inv);
  copy(h_edge_v0); copy(h_edge_v1);
  copy(h_edge_boundary); copy(h_edge_radial_layer);
  copy(v_edge_length); copy(v_edge_hodge1_inv);
  copy(v_edge_v0); copy(v_edge_v1);
  copy(v_edge_boundary); copy(v_edge_radial_layer);
  copy(vert_r); copy(vert_theta); copy(vert_phi);
  copy(tri_face_l2g); copy(rect_face_l2g);
  copy(h_edge_l2g); copy(v_edge_l2g);
#endif
}

}  // namespace Aperture
