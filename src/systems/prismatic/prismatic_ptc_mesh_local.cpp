#include "systems/prismatic/prismatic_ptc_mesh_local.h"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_cochain_layout.h"
#include "systems/prismatic/prismatic_mesh_geom.h"
#include "systems/prismatic/prismatic_vertex_recovery.h"
#include "utils/logger.h"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace Aperture {

namespace {

// Fill a buffer<int>/buffer<Scalar> from a std::vector.
template <typename T>
void fill_buffer(buffer<T>& b, const std::vector<T>& v, MemType mem) {
  b.set_memtype(mem);
  b.resize(std::max<size_t>(v.size(), 1));
  for (size_t i = 0; i < v.size(); ++i) b[i] = v[i];
}

}  // namespace

void prismatic_ptc_mesh_local::build(const prismatic_mesh& mesh,
                                     const prismatic_mesh_partition& mp,
                                     const prismatic_vertex_recovery* recovery,
                                     MemType mem) {
  m_mesh = &mesh;
  const prismatic_partition& part = mp.partition();
  const icosphere_topology& topo = mp.topology();

  const int NT = topo.N_tri();
  const int NV = topo.N_vert_s();
  const int NE = topo.N_edge_s();
  m_n_tri_global = NT;
  m_N_r_global = part.N_r_global;

  // -----------------------------------------------------------------------
  // Local tri set: owned (ascending global), then the 1-ring (tris
  // sharing >= 1 sphere-vertex with an owned tri), ascending global.
  // -----------------------------------------------------------------------
  std::vector<char> tri_owned(NT, 0), tri_halo(NT, 0);
  for (int t = 0; t < NT; ++t) tri_owned[t] = part.owns_sub_tri(t) ? 1 : 0;
  for (int v = 0; v < NV; ++v) {
    const int* fan = topo.vertex_tris(v);
    const int n = topo.vertex_tri_count(v);
    bool touches_owned = false;
    for (int j = 0; j < n; ++j) {
      if (tri_owned[fan[j]]) {
        touches_owned = true;
        break;
      }
    }
    if (touches_owned) {
      for (int j = 0; j < n; ++j) tri_halo[fan[j]] = 1;
    }
  }

  m_tri_l2g_host.clear();
  for (int t = 0; t < NT; ++t) {
    if (tri_owned[t]) m_tri_l2g_host.push_back(t);
  }
  m_n_tri_own = int(m_tri_l2g_host.size());
  for (int t = 0; t < NT; ++t) {
    if (tri_halo[t] && !tri_owned[t]) m_tri_l2g_host.push_back(t);
  }
  m_n_tri_local = int(m_tri_l2g_host.size());
  m_tri_g2l.assign(NT, -1);
  for (int l = 0; l < m_n_tri_local; ++l) m_tri_g2l[m_tri_l2g_host[l]] = l;

  // -----------------------------------------------------------------------
  // Local sphere vertices / edges: elements of local tris, owned block
  // first, each block ascending global.
  // -----------------------------------------------------------------------
  std::vector<char> vert_used(NV, 0), edge_used(NE, 0);
  const int* g_tri_verts = mesh.tri_verts.host_ptr();
  const int* g_tri_edges = mesh.tri_edges_s.host_ptr();
  for (int l = 0; l < m_n_tri_local; ++l) {
    const int t = m_tri_l2g_host[l];
    for (int j = 0; j < 3; ++j) {
      vert_used[g_tri_verts[t * 3 + j]] = 1;
      edge_used[g_tri_edges[t * 3 + j]] = 1;
    }
  }
  auto build_l2g = [&](int n_glob, const std::vector<char>& used,
                       auto&& owns, std::vector<int>& l2g, int& n_own) {
    l2g.clear();
    for (int g = 0; g < n_glob; ++g) {
      if (used[g] && owns(g)) l2g.push_back(g);
    }
    n_own = int(l2g.size());
    for (int g = 0; g < n_glob; ++g) {
      if (used[g] && !owns(g)) l2g.push_back(g);
    }
  };
  build_l2g(NV, vert_used,
            [&](int v) { return part.owns_sphere_vertex(v); }, m_vert_l2g,
            m_n_vert_s_own);
  build_l2g(NE, edge_used,
            [&](int e) { return part.owns_sphere_edge(e); }, m_edge_l2g,
            m_n_edge_s_own);
  m_n_vert_s_local = int(m_vert_l2g.size());
  m_n_edge_s_local = int(m_edge_l2g.size());

  std::vector<int> vert_g2l(NV, -1), edge_g2l(NE, -1);
  for (int l = 0; l < m_n_vert_s_local; ++l) vert_g2l[m_vert_l2g[l]] = l;
  for (int l = 0; l < m_n_edge_s_local; ++l) edge_g2l[m_edge_l2g[l]] = l;

  // Owned sphere elements always land in the local set (an owned
  // vertex/edge has an owned incident tri).
  for (int v = 0; v < NV; ++v) {
    if (part.owns_sphere_vertex(v) && vert_g2l[v] < 0) {
      throw std::runtime_error(
          "ptc_mesh_local: owned sphere vertex missing from local set");
    }
  }

  // -----------------------------------------------------------------------
  // Radial extent: owned slabs padded by one ghost prism per interior
  // side.  k0 = global index of local layer 0.
  // -----------------------------------------------------------------------
  const int N_r_g = part.N_r_global;
  const bool has_lower = part.radial_rank > 0;
  const bool has_upper = part.radial_rank < part.n_radial_ranks - 1;
  const int kl = part.shell_k_lo;
  const int owned_top_slab = std::min(part.shell_k_hi, N_r_g) - 1;
  m_k0 = kl - (has_lower ? 1 : 0);
  const int top_slab = owned_top_slab + (has_upper ? 1 : 0);
  m_n_layers = top_slab - m_k0 + 1;
  m_lay_own_lo = kl - m_k0;
  m_lay_own_hi = owned_top_slab + 1 - m_k0;
  m_shell_own_lo = kl - m_k0;
  m_shell_own_hi = std::min(part.shell_k_hi, N_r_g + 1) - m_k0;

  // Particle LOCAL cells are uint32 (lay * N_tri_local + tri, with
  // empty_cell = uint32 max as the sentinel).  The GLOBAL wire is
  // 64-bit, but a rank whose LOCAL cell space reaches the sentinel
  // would corrupt silently — refuse loudly (means: decompose more).
  if (size_t(m_n_tri_local) * size_t(m_n_layers) >= size_t(empty_cell)) {
    Logger::print_err(
        "prismatic_ptc_mesh_local: local cell space {} x {} exceeds the "
        "uint32 particle-cell encoding; increase A x K",
        m_n_tri_local, m_n_layers);
    std::abort();
  }

  // -----------------------------------------------------------------------
  // Remapped sphere tables.
  // -----------------------------------------------------------------------
  const int* g_signs = mesh.tri_edge_signs.host_ptr();
  const int* g_neigh = mesh.tri_neighbor.host_ptr();
  std::vector<int> l_tri_verts(m_n_tri_local * 3);
  std::vector<int> l_tri_edges(m_n_tri_local * 3);
  std::vector<int> l_signs(m_n_tri_local * 3);
  std::vector<int> l_neigh(m_n_tri_local * 3);
  for (int l = 0; l < m_n_tri_local; ++l) {
    const int t = m_tri_l2g_host[l];
    for (int j = 0; j < 3; ++j) {
      l_tri_verts[l * 3 + j] = vert_g2l[g_tri_verts[t * 3 + j]];
      l_tri_edges[l * 3 + j] = edge_g2l[g_tri_edges[t * 3 + j]];
      l_signs[l * 3 + j] = g_signs[t * 3 + j];
      const int ng = g_neigh[t * 3 + j];
      l_neigh[l * 3 + j] = ng >= 0 ? m_tri_g2l[ng] : -1;
    }
  }
  fill_buffer(tri_verts, l_tri_verts, mem);
  fill_buffer(tri_edges_s, l_tri_edges, mem);
  fill_buffer(tri_edge_signs, l_signs, mem);
  fill_buffer(tri_neighbor, l_neigh, mem);

  auto remap_scalar = [&](const buffer<Scalar>& g, buffer<Scalar>& out,
                          const std::vector<int>& l2g) {
    std::vector<Scalar> v(l2g.size());
    const Scalar* gp = g.host_ptr();
    for (size_t i = 0; i < l2g.size(); ++i) v[i] = gp[l2g[i]];
    fill_buffer(out, v, mem);
  };
  remap_scalar(mesh.sphere_vx, sphere_vx, m_vert_l2g);
  remap_scalar(mesh.sphere_vy, sphere_vy, m_vert_l2g);
  remap_scalar(mesh.sphere_vz, sphere_vz, m_vert_l2g);
  remap_scalar(mesh.sphere_theta, sphere_theta, m_vert_l2g);
  remap_scalar(mesh.sphere_phi, sphere_phi, m_vert_l2g);

  // -----------------------------------------------------------------------
  // Tensor -> layout maps.  Total on pic-depth layouts (asserted).
  // -----------------------------------------------------------------------
  const auto& L_h = mp.layout(cochain_type::h_edge);
  const auto& L_v = mp.layout(cochain_type::v_edge);
  const auto& L_tri = mp.layout(cochain_type::tri_face);
  const auto& L_rect = mp.layout(cochain_type::rect_face);
  const auto& L_vert = mp.layout(cochain_type::vertex);
  m_e_split = L_h.local_size();
  m_b_split = L_tri.local_size();
  m_n_verts_layout = L_vert.local_size();
  m_n_edges_layout = L_h.local_size() + L_v.local_size();
  m_n_faces_layout = L_tri.local_size() + L_rect.local_size();

  auto build_map = [&](const distributed_cochain_layout& L, int n_levels,
                       int width_glob, const std::vector<int>& s_l2g,
                       int n_s_loc, buffer<int>& out, const char* what) {
    std::vector<int> m(size_t(n_levels) * n_s_loc);
    for (int k = 0; k < n_levels; ++k) {
      const int kg = m_k0 + k;
      for (int s = 0; s < n_s_loc; ++s) {
        const gidx_t g = gidx_t(kg) * width_glob + s_l2g[s];
        const int loc = L.to_local(g);
        if (loc < 0) {
          throw std::runtime_error(
              std::string("ptc_mesh_local: tensor slot missing from the ") +
              what +
              " layout — the mesh_partition must be built at pic depth");
        }
        m[size_t(k) * n_s_loc + s] = loc;
      }
    }
    fill_buffer(out, m, mem);
  };
  build_map(L_h, m_n_layers + 1, NE, m_edge_l2g, m_n_edge_s_local, map_h,
            "h_edge");
  build_map(L_v, m_n_layers, NV, m_vert_l2g, m_n_vert_s_local, map_v,
            "v_edge");
  build_map(L_tri, m_n_layers + 1, NT, m_tri_l2g_host, m_n_tri_local,
            map_tri, "tri_face");
  build_map(L_rect, m_n_layers, NE, m_edge_l2g, m_n_edge_s_local, map_rect,
            "rect_face");
  build_map(L_vert, m_n_layers + 1, NV, m_vert_l2g, m_n_vert_s_local,
            map_vert, "vertex");

  // -----------------------------------------------------------------------
  // Layout-indexed geometry: dual volumes (vertex layout) and hodge1_inv
  // (combined [h|v] edge layout).
  // -----------------------------------------------------------------------
  // Phase 7D: computed from the sphere stage (bit-identical to the
  // retired global arrays; see prismatic_mesh_geom.h).
  {
    std::vector<Scalar> v(m_n_verts_layout);
    for (int l = 0; l < m_n_verts_layout; ++l) {
      const gidx_t g = L_vert.to_global(l);
      v[l] = prismatic_geom::vert_dual_vol(mesh, int(g / NV), int(g % NV));
    }
    fill_buffer(vert_dual_vol, v, mem);
  }
  {
    std::vector<Scalar> v(m_n_edges_layout);
    for (int l = 0; l < m_e_split; ++l) {
      const gidx_t g = L_h.to_global(l);
      v[l] = prismatic_geom::hodge1_inv_h(mesh, int(g / NE), int(g % NE));
    }
    for (int l = 0; l < L_v.local_size(); ++l) {
      const gidx_t g = L_v.to_global(l);
      v[m_e_split + l] = prismatic_geom::hodge1_inv_v(mesh, int(g / NV), int(g % NV));
    }
    fill_buffer(hodge1_inv, v, mem);
  }

  // -----------------------------------------------------------------------
  // Recovery tables remapped to local sphere ids.  Fan entries off the
  // halo become −1; they can only appear on non-owned vertex rows,
  // whose Bv arrives via the halo exchange (asserted for owned rows).
  // -----------------------------------------------------------------------
  m_has_recovery = recovery != nullptr;
  if (m_has_recovery) {
    const int* g_val = recovery->valence.host_ptr();
    const int* g_tf = recovery->tri_fan.host_ptr();
    const int* g_ef = recovery->edge_fan.host_ptr();
    std::vector<int> l_val(m_n_vert_s_local);
    std::vector<int> l_tf(m_n_vert_s_local * 6, -1);
    std::vector<int> l_ef(m_n_vert_s_local * 6, -1);
    for (int l = 0; l < m_n_vert_s_local; ++l) {
      const int s = m_vert_l2g[l];
      l_val[l] = g_val[s];
      for (int j = 0; j < g_val[s]; ++j) {
        const int tf = m_tri_g2l[g_tf[s * 6 + j]];
        const int ef = edge_g2l[g_ef[s * 6 + j]];
        l_tf[l * 6 + j] = tf;
        l_ef[l * 6 + j] = ef;
        if (l < m_n_vert_s_own && (tf < 0 || ef < 0)) {
          throw std::runtime_error(
              "ptc_mesh_local: owned vertex has a recovery fan member "
              "outside the local halo");
        }
      }
    }
    fill_buffer(rec_valence, l_val, mem);
    fill_buffer(rec_tri_fan, l_tf, mem);
    fill_buffer(rec_edge_fan, l_ef, mem);

    auto remap_rows = [&](const buffer<Scalar>& g, buffer<Scalar>& out,
                          int row_len) {
      std::vector<Scalar> v(size_t(m_n_vert_s_local) * row_len);
      const Scalar* gp = g.host_ptr();
      for (int l = 0; l < m_n_vert_s_local; ++l) {
        const int s = m_vert_l2g[l];
        std::copy(gp + size_t(s) * row_len, gp + size_t(s + 1) * row_len,
                  v.begin() + size_t(l) * row_len);
      }
      fill_buffer(out, v, mem);
    };
    remap_rows(recovery->w_int, rec_w_int, 3 * 30);
    remap_rows(recovery->w_inner, rec_w_inner, 3 * 24);
    remap_rows(recovery->w_outer, rec_w_outer, 3 * 24);
    m_rec_r_ref = mesh.radii[1];
  }

  // -----------------------------------------------------------------------
  // Migration tables: global tri + canonical angular rank per local tri.
  // -----------------------------------------------------------------------
  {
    std::vector<int> rank_of(m_n_tri_local);
    for (int l = 0; l < m_n_tri_local; ++l) {
      const int g = m_tri_l2g_host[l];
      rank_of[l] = part.angular_rank_of_path_unit(
          part.path_of_unit(part.unit_of_tri(g)));
    }
    fill_buffer(tri_l2g, m_tri_l2g_host, mem);
    fill_buffer(tri_ang_rank, rank_of, mem);
  }
  m_A = part.n_angular_ranks;
  const int K = part.n_radial_ranks;
  m_slab_base = N_r_g / K;
  m_slab_rem = N_r_g - m_slab_base * K;
}

void prismatic_ptc_mesh_local::copy_to_device() {
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  for (auto* b : {&sphere_vx, &sphere_vy, &sphere_vz, &sphere_theta,
                  &sphere_phi, &vert_dual_vol, &hodge1_inv, &rec_w_int,
                  &rec_w_inner, &rec_w_outer}) {
    if (b->size() > 0) b->copy_to_device();
  }
  for (auto* b : {&tri_verts, &tri_edges_s, &tri_edge_signs, &tri_neighbor,
                  &map_h, &map_v, &map_tri, &map_rect, &map_vert,
                  &rec_valence, &rec_tri_fan, &rec_edge_fan, &tri_l2g,
                  &tri_ang_rank}) {
    if (b->size() > 0) b->copy_to_device();
  }
#endif
}

namespace {

template <typename T>
const T* pick_ptr(const buffer<T>& b, bool dev) {
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  if (dev) return b.dev_ptr();
#endif
  (void)dev;
  return b.host_ptr();
}

}  // namespace

prismatic_ptc_mesh_ptrs prismatic_ptc_mesh_local::host_ptrs() const {
  prismatic_ptc_mesh_ptrs p{};
  p.N_r = m_n_layers;
  p.N_tri = m_n_tri_local;
  p.N_vert_s = m_n_vert_s_local;
  p.N_edge_s = m_n_edge_s_local;
  p.N_verts = m_n_verts_layout;
  p.N_edges = m_n_edges_layout;
  p.N_faces = m_n_faces_layout;
  p.n_tri_own = m_n_tri_own;
  p.n_vert_s_own = m_n_vert_s_own;
  p.n_edge_s_own = m_n_edge_s_own;
  p.k0 = m_k0;
  p.lay_own_lo = m_lay_own_lo;
  p.lay_own_hi = m_lay_own_hi;
  p.shell_own_lo = m_shell_own_lo;
  p.shell_own_hi = m_shell_own_hi;
  p.N_r_global = m_N_r_global;
  p.radii = m_mesh->radii.host_ptr() + m_k0;
  p.sphere_vx = sphere_vx.host_ptr();
  p.sphere_vy = sphere_vy.host_ptr();
  p.sphere_vz = sphere_vz.host_ptr();
  p.sphere_theta = sphere_theta.host_ptr();
  p.sphere_phi = sphere_phi.host_ptr();
  p.tri_verts = tri_verts.host_ptr();
  p.tri_edges_s = tri_edges_s.host_ptr();
  p.tri_edge_signs = tri_edge_signs.host_ptr();
  p.tri_neighbor = tri_neighbor.host_ptr();
  p.map_h = map_h.host_ptr();
  p.map_v = map_v.host_ptr();
  p.map_tri = map_tri.host_ptr();
  p.map_rect = map_rect.host_ptr();
  p.map_vert = map_vert.host_ptr();
  p.e_split = m_e_split;
  p.b_split = m_b_split;
  p.vert_dual_vol = vert_dual_vol.host_ptr();
  p.hodge1_inv = hodge1_inv.host_ptr();
  if (m_has_recovery) {
    p.rec_valence = rec_valence.host_ptr();
    p.rec_tri_fan = rec_tri_fan.host_ptr();
    p.rec_edge_fan = rec_edge_fan.host_ptr();
    p.rec_w_int = rec_w_int.host_ptr();
    p.rec_w_inner = rec_w_inner.host_ptr();
    p.rec_w_outer = rec_w_outer.host_ptr();
  }
  p.rec_r_ref = m_rec_r_ref;
  p.tri_l2g = tri_l2g.host_ptr();
  p.tri_ang_rank = tri_ang_rank.host_ptr();
  p.N_tri_global = m_n_tri_global;
  p.mig_A = m_A;
  p.mig_slab_base = m_slab_base;
  p.mig_slab_rem = m_slab_rem;
  return p;
}

prismatic_ptc_mesh_ptrs prismatic_ptc_mesh_local::dev_ptrs() const {
  prismatic_ptc_mesh_ptrs p = host_ptrs();
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  p.radii = m_mesh->radii.dev_ptr() + m_k0;
  p.sphere_vx = sphere_vx.dev_ptr();
  p.sphere_vy = sphere_vy.dev_ptr();
  p.sphere_vz = sphere_vz.dev_ptr();
  p.sphere_theta = sphere_theta.dev_ptr();
  p.sphere_phi = sphere_phi.dev_ptr();
  p.tri_verts = tri_verts.dev_ptr();
  p.tri_edges_s = tri_edges_s.dev_ptr();
  p.tri_edge_signs = tri_edge_signs.dev_ptr();
  p.tri_neighbor = tri_neighbor.dev_ptr();
  p.map_h = map_h.dev_ptr();
  p.map_v = map_v.dev_ptr();
  p.map_tri = map_tri.dev_ptr();
  p.map_rect = map_rect.dev_ptr();
  p.map_vert = map_vert.dev_ptr();
  p.vert_dual_vol = vert_dual_vol.dev_ptr();
  p.hodge1_inv = hodge1_inv.dev_ptr();
  if (m_has_recovery) {
    p.rec_valence = rec_valence.dev_ptr();
    p.rec_tri_fan = rec_tri_fan.dev_ptr();
    p.rec_edge_fan = rec_edge_fan.dev_ptr();
    p.rec_w_int = rec_w_int.dev_ptr();
    p.rec_w_inner = rec_w_inner.dev_ptr();
    p.rec_w_outer = rec_w_outer.dev_ptr();
  }
  p.tri_l2g = tri_l2g.dev_ptr();
  p.tri_ang_rank = tri_ang_rank.dev_ptr();
#endif
  return p;
}

}  // namespace Aperture
