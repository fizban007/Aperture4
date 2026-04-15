#include "systems/prismatic/prismatic_mesh_metric_impl.hpp"

namespace Aperture {

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
  p.cc_x = acc(m.cc_x);
  p.cc_y = acc(m.cc_y);
  p.cc_z = acc(m.cc_z);
  p.edge_tris = acc(m.edge_tris);
  p.vert_tri_count = acc(m.vert_tri_count);
  p.vert_tris = acc(m.vert_tris);
  p.N_h_edges = (m.m_N_r + 1) * m.m_N_edge_s;
  p.N_tri_faces = (m.m_N_r + 1) * m.m_N_tri;
}

}  // namespace

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

  auto copy = [](auto& buf) {
    if (buf.mem_type() == MemType::host_device) buf.copy_to_device();
  };
  copy(edge_r_coord); copy(edge_sth); copy(edge_cth);
  copy(edge_alpha); copy(edge_sq_gamma_beta_r); copy(edge_sqrt_gamma);
  copy(face_r_coord); copy(face_sth); copy(face_cth);
  copy(face_alpha); copy(face_sq_gamma_beta_r); copy(face_sqrt_gamma);
  copy(cc_x); copy(cc_y); copy(cc_z);
  copy(edge_tris); copy(vert_tri_count); copy(vert_tris);
}

void prismatic_mesh_metric::copy_metric_to_host() {
  auto copy = [](auto& buf) {
    if (buf.mem_type() == MemType::host_device) buf.copy_to_host();
  };
  copy(hodge1_inv); copy(hodge2);
  copy(edge_r_coord); copy(edge_sth); copy(edge_cth);
  copy(edge_alpha); copy(edge_sq_gamma_beta_r); copy(edge_sqrt_gamma);
  copy(face_r_coord); copy(face_sth); copy(face_cth);
  copy(face_alpha); copy(face_sq_gamma_beta_r); copy(face_sqrt_gamma);
  copy(cc_x); copy(cc_y); copy(cc_z);
}
#endif

// Explicit instantiations of the templated compute_metric for the two
// concrete metric types.  When GPU is enabled the instantiation is
// done in prismatic_mesh_metric.hip.cpp instead (the kernel launches
// inside compute_metric only compile under the CUDA/HIP compiler).
#if !(defined(CUDA_ENABLED) || defined(HIP_ENABLED))
template void prismatic_mesh_metric::compute_metric<flat_spherical_metric>(
    const flat_spherical_metric&);
template void prismatic_mesh_metric::compute_metric<ks_spherical_metric>(
    const ks_spherical_metric&);
#endif

}  // namespace Aperture
