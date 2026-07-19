#include "systems/prismatic/prismatic_d1_local.h"
#include <vector>
#include "systems/prismatic/prismatic_mesh.h"
#include <cassert>

namespace Aperture {

prismatic_d1_local prismatic_d1_local::build(
    const prismatic_mesh& mesh, const prismatic_mesh_partition& mp,
    MemType mem_type) {
  prismatic_d1_local out;

  struct csr_stage {
    std::vector<int> row_ptr, col_idx;
    std::vector<Scalar> val;
  };
  csr_stage st_d1_tri_h, st_d1_rect_h, st_d1_rect_v, st_d1t_h_tri, st_d1t_h_rect, st_d1t_v_rect;
  out.m_partition = &mp;

  // Phase 7D: rows are SYNTHESIZED from the persisted sphere tables —
  // the incidence structure is analytic in (shell pattern × sphere
  // topology), so no global CSR is needed.  Entry ORDER per row matches
  // the historical build_incidence / transpose_d1 conventions exactly
  // (tri rows: tri_edges order; rect rows: bottom h, right v, top h,
  // left v; d1t rows: ascending global face index).
  const int NT = mesh.m_N_tri;
  const int NE = mesh.m_N_edge_s;
  const int NV = mesh.m_N_vert_s;
  const int N_r = mesh.m_N_r;

  auto const& L_tri  = mp.layout(cochain_type::tri_face);
  auto const& L_rect = mp.layout(cochain_type::rect_face);
  auto const& L_he   = mp.layout(cochain_type::h_edge);
  auto const& L_ve   = mp.layout(cochain_type::v_edge);

  const int* tri_edges_s = mesh.tri_edges_s.host_ptr();
  const int* tri_signs = mesh.tri_edge_signs.host_ptr();
  const int* edge_a = mesh.sphere_edge_v0.host_ptr();
  const int* edge_b = mesh.sphere_edge_v1.host_ptr();

  // Per-vertex incident sphere-edge lists (ascending e — the d1t
  // transpose order for v-edge rows) with the vertical-edge sign:
  // +1 when the vertex is the edge's b endpoint (right vertical),
  // −1 when it is the a endpoint (left vertical).
  std::vector<std::vector<int>> vert_edges(NV);
  std::vector<std::vector<Scalar>> vert_edge_sign(NV);
  for (int e = 0; e < NE; ++e) {
    vert_edges[edge_a[e]].push_back(e);
    vert_edge_sign[edge_a[e]].push_back(Scalar(-1));
    vert_edges[edge_b[e]].push_back(e);
    vert_edge_sign[edge_b[e]].push_back(Scalar(+1));
  }

  // ===== d1: tri faces → h edges =====
  st_d1_tri_h.row_ptr.assign(L_tri.owned_size() + 1, 0);
  for (int l = 0; l < L_tri.owned_size(); ++l) {
    const gidx_t g_face = L_tri.to_global(l);  // [0, N_tri_faces)
    const int k = int(g_face / NT), t = int(g_face % NT);
    for (int j = 0; j < 3; ++j) {
      const int local_he = L_he.to_local(gidx_t(k) * NE + tri_edges_s[t * 3 + j]);
      assert(local_he >= 0 && "h_edge boundary of owned tri must be in halo");
      st_d1_tri_h.col_idx.push_back(local_he);
      st_d1_tri_h.val.push_back(Scalar(tri_signs[t * 3 + j]));
    }
    st_d1_tri_h.row_ptr[l + 1] = int(st_d1_tri_h.col_idx.size());
  }

  // ===== d1: rect faces → h and v edges =====
  // Circuit (build_incidence): bottom h(k,e) +1, right v(k,b) +1,
  // top h(k+1,e) −1, left v(k,a) −1.
  st_d1_rect_h.row_ptr.assign(L_rect.owned_size() + 1, 0);
  st_d1_rect_v.row_ptr.assign(L_rect.owned_size() + 1, 0);
  for (int l = 0; l < L_rect.owned_size(); ++l) {
    const gidx_t g_rect = L_rect.to_global(l);  // [0, N_rect_faces)
    const int k = int(g_rect / NE), e = int(g_rect % NE);
    const int a = edge_a[e], b = edge_b[e];

    const int he_bot = L_he.to_local(gidx_t(k) * NE + e);
    const int he_top = L_he.to_local(gidx_t(k + 1) * NE + e);
    const int ve_r = L_ve.to_local(gidx_t(k) * NV + b);
    const int ve_l = L_ve.to_local(gidx_t(k) * NV + a);
    assert(he_bot >= 0 && he_top >= 0 &&
           "h_edge boundary of owned rect must be in halo");
    assert(ve_r >= 0 && ve_l >= 0 &&
           "v_edge boundary of owned rect must be in halo");
    st_d1_rect_h.col_idx.push_back(he_bot);
    st_d1_rect_h.val.push_back(Scalar(+1));
    st_d1_rect_v.col_idx.push_back(ve_r);
    st_d1_rect_v.val.push_back(Scalar(+1));
    st_d1_rect_h.col_idx.push_back(he_top);
    st_d1_rect_h.val.push_back(Scalar(-1));
    st_d1_rect_v.col_idx.push_back(ve_l);
    st_d1_rect_v.val.push_back(Scalar(-1));

    st_d1_rect_h.row_ptr[l + 1] = int(st_d1_rect_h.col_idx.size());
    st_d1_rect_v.row_ptr[l + 1] = int(st_d1_rect_v.col_idx.size());
  }

  // ===== d1^T: h edges → tri and rect faces =====
  // Transpose order = ascending global face index: the two shell-k tri
  // faces containing e (ascending t), then rect (k−1, e) [top edge of
  // the layer below, −1], then rect (k, e) [bottom edge, +1].
  st_d1t_h_tri.row_ptr.assign(L_he.owned_size() + 1, 0);
  st_d1t_h_rect.row_ptr.assign(L_he.owned_size() + 1, 0);
  for (int l = 0; l < L_he.owned_size(); ++l) {
    const gidx_t g_he = L_he.to_global(l);  // [0, N_h_edges)
    const int k = int(g_he / NE), e = int(g_he % NE);
    const int t0 = mesh.sph_edge_tri0[e], t1 = mesh.sph_edge_tri1[e];
    for (int t : {t0 < t1 ? t0 : t1, t0 < t1 ? t1 : t0}) {
      if (t < 0) continue;
      // Sign = orientation of e within tri t.
      Scalar sgn = 0;
      for (int j = 0; j < 3; ++j) {
        if (tri_edges_s[t * 3 + j] == e) sgn = Scalar(tri_signs[t * 3 + j]);
      }
      const int local_tri = L_tri.to_local(gidx_t(k) * NT + t);
      assert(local_tri >= 0 &&
             "tri face adjacent to owned h_edge must be in halo");
      st_d1t_h_tri.col_idx.push_back(local_tri);
      st_d1t_h_tri.val.push_back(sgn);
    }
    if (k > 0) {
      const int local_rect = L_rect.to_local(gidx_t(k - 1) * NE + e);
      assert(local_rect >= 0 &&
             "rect face adjacent to owned h_edge must be in halo");
      st_d1t_h_rect.col_idx.push_back(local_rect);
      st_d1t_h_rect.val.push_back(Scalar(-1));
    }
    if (k < N_r) {
      const int local_rect = L_rect.to_local(gidx_t(k) * NE + e);
      assert(local_rect >= 0 &&
             "rect face adjacent to owned h_edge must be in halo");
      st_d1t_h_rect.col_idx.push_back(local_rect);
      st_d1t_h_rect.val.push_back(Scalar(+1));
    }
    st_d1t_h_tri.row_ptr[l + 1]  = int(st_d1t_h_tri.col_idx.size());
    st_d1t_h_rect.row_ptr[l + 1] = int(st_d1t_h_rect.col_idx.size());
  }

  // ===== d1^T: v edges → rect faces =====
  // Transpose order: rect faces (k, e) for the fan edges e of vertex s,
  // ascending e; sign +1 when s is the edge's b endpoint (right
  // vertical), −1 for the a endpoint.
  st_d1t_v_rect.row_ptr.assign(L_ve.owned_size() + 1, 0);
  for (int l = 0; l < L_ve.owned_size(); ++l) {
    const gidx_t g_ve = L_ve.to_global(l);  // [0, N_v_edges)
    const int k = int(g_ve / NV), sv = int(g_ve % NV);
    const auto& fan = vert_edges[sv];
    const auto& sgn = vert_edge_sign[sv];
    for (size_t j = 0; j < fan.size(); ++j) {
      const int local_rect = L_rect.to_local(gidx_t(k) * NE + fan[j]);
      assert(local_rect >= 0 &&
             "rect face adjacent to owned v_edge must be in halo");
      st_d1t_v_rect.col_idx.push_back(local_rect);
      st_d1t_v_rect.val.push_back(sgn[j]);
    }
    st_d1t_v_rect.row_ptr[l + 1] = int(st_d1t_v_rect.col_idx.size());
  }


  // Move the staged CSR blocks into the buffer-backed storage.
  auto commit = [mem_type](csr_stage& st, auto& blk) {
    blk.row_ptr.set_memtype(mem_type);
    blk.col_idx.set_memtype(mem_type);
    blk.val.set_memtype(mem_type);
    blk.row_ptr.resize(st.row_ptr.size());
    for (size_t i = 0; i < st.row_ptr.size(); i++) blk.row_ptr[i] = st.row_ptr[i];
    blk.col_idx.resize(st.col_idx.size());
    for (size_t i = 0; i < st.col_idx.size(); i++) blk.col_idx[i] = st.col_idx[i];
    blk.val.resize(st.val.size());
    for (size_t i = 0; i < st.val.size(); i++) blk.val[i] = st.val[i];
  };
  commit(st_d1_tri_h, out.d1_tri_h);
  commit(st_d1_rect_h, out.d1_rect_h);
  commit(st_d1_rect_v, out.d1_rect_v);
  commit(st_d1t_h_tri, out.d1t_h_tri);
  commit(st_d1t_h_rect, out.d1t_h_rect);
  commit(st_d1t_v_rect, out.d1t_v_rect);

  return out;
}

}  // namespace Aperture
