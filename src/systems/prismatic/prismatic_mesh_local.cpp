#include "systems/prismatic/prismatic_mesh_local.h"
#include "systems/prismatic/prismatic_cochain_layout.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_geom.h"

namespace Aperture {

namespace {

// Fill a local buffer by evaluating `f(global_sub_index)` for every
// local slot of `layout` (Phase 7D: geometry is COMPUTED from the
// sphere stage — bit-identical to the retired global arrays, pinned by
// test_prismatic_mesh_geom — so the mesh needs no 3D allocations).
template <typename T, typename F>
void fill_local(const distributed_cochain_layout& layout, buffer<T>& dst,
                MemType mem_type, F&& f) {
  const int n = layout.local_size();
  dst.set_memtype(mem_type);
  dst.resize(n);
  for (int l = 0; l < n; ++l) {
    dst[l] = f(layout.to_global(l));
  }
}

}  // namespace

prismatic_mesh_local prismatic_mesh_local::build(
    const prismatic_mesh& mesh, const prismatic_mesh_partition& mp,
    MemType mem_type) {
  namespace pg = prismatic_geom;
  prismatic_mesh_local out;
  out.m_partition = &mp;

  const int NT = mesh.m_N_tri;
  const int NE = mesh.m_N_edge_s;
  const int NV = mesh.m_N_vert_s;

  auto const& L_tri  = mp.layout(cochain_type::tri_face);
  auto const& L_rect = mp.layout(cochain_type::rect_face);
  auto const& L_he   = mp.layout(cochain_type::h_edge);
  auto const& L_ve   = mp.layout(cochain_type::v_edge);
  auto const& L_vert = mp.layout(cochain_type::vertex);

  // ---- Tri-face buffers ----
  fill_local(L_tri, out.tri_face_area, mem_type, [&](gidx_t g) {
    return pg::tri_area(mesh, int(g / NT), int(g % NT));
  });
  fill_local(L_tri, out.tri_face_hodge2, mem_type, [&](gidx_t g) {
    return pg::hodge2_tri(mesh, int(g / NT), int(g % NT));
  });
  fill_local(L_tri, out.tri_face_boundary, mem_type, [&](gidx_t g) {
    return pg::tri_face_boundary(mesh, int(g / NT));
  });
  fill_local(L_tri, out.tri_face_radial_layer, mem_type,
             [&](gidx_t g) { return int(g / NT); });

  // ---- Rect-face buffers ----
  fill_local(L_rect, out.rect_face_area, mem_type, [&](gidx_t g) {
    return pg::rect_area(mesh, int(g / NE), int(g % NE));
  });
  fill_local(L_rect, out.rect_face_hodge2, mem_type, [&](gidx_t g) {
    return pg::hodge2_rect(mesh, int(g / NE), int(g % NE));
  });
  fill_local(L_rect, out.rect_face_boundary, mem_type, [&](gidx_t g) {
    return pg::rect_face_boundary(mesh, int(g / NE));
  });
  fill_local(L_rect, out.rect_face_radial_layer, mem_type,
             [&](gidx_t g) { return int(g / NE); });

  // ---- H-edge buffers ----
  fill_local(L_he, out.h_edge_length, mem_type, [&](gidx_t g) {
    return pg::h_edge_length(mesh, int(g / NE), int(g % NE));
  });
  fill_local(L_he, out.h_edge_hodge1_inv, mem_type, [&](gidx_t g) {
    return pg::hodge1_inv_h(mesh, int(g / NE), int(g % NE));
  });
  fill_local(L_he, out.h_edge_v0, mem_type, [&](gidx_t g) {
    return pg::h_edge_gv0(mesh, int(g / NE), int(g % NE));
  });
  fill_local(L_he, out.h_edge_v1, mem_type, [&](gidx_t g) {
    return pg::h_edge_gv1(mesh, int(g / NE), int(g % NE));
  });
  fill_local(L_he, out.h_edge_boundary, mem_type, [&](gidx_t g) {
    return pg::h_edge_boundary(mesh, int(g / NE));
  });
  fill_local(L_he, out.h_edge_radial_layer, mem_type,
             [&](gidx_t g) { return int(g / NE); });

  // ---- V-edge buffers ----
  fill_local(L_ve, out.v_edge_length, mem_type, [&](gidx_t g) {
    return pg::v_edge_length(mesh, int(g / NV));
  });
  fill_local(L_ve, out.v_edge_hodge1_inv, mem_type, [&](gidx_t g) {
    return pg::hodge1_inv_v(mesh, int(g / NV), int(g % NV));
  });
  fill_local(L_ve, out.v_edge_v0, mem_type, [&](gidx_t g) {
    return pg::v_edge_gv0(mesh, int(g / NV), int(g % NV));
  });
  fill_local(L_ve, out.v_edge_v1, mem_type, [&](gidx_t g) {
    return pg::v_edge_gv1(mesh, int(g / NV), int(g % NV));
  });
  fill_local(L_ve, out.v_edge_boundary, mem_type, [&](gidx_t g) {
    return pg::v_edge_boundary(mesh, int(g / NV));
  });
  fill_local(L_ve, out.v_edge_radial_layer, mem_type,
             [&](gidx_t g) { return int(g / NV); });

  // ---- Vertex buffers ----
  fill_local(L_vert, out.vert_r, mem_type,
             [&](gidx_t g) { return pg::vert_r(mesh, int(g / NV)); });
  fill_local(L_vert, out.vert_theta, mem_type,
             [&](gidx_t g) { return pg::vert_theta(mesh, int(g % NV)); });
  fill_local(L_vert, out.vert_phi, mem_type,
             [&](gidx_t g) { return pg::vert_phi(mesh, int(g % NV)); });

  // ---- Local -> global maps ----
  auto fill_l2g = [mem_type](const distributed_cochain_layout& layout,
                             buffer<gidx_t>& dst) {
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
