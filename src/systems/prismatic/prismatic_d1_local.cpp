#include "systems/prismatic/prismatic_d1_local.h"
#include <vector>
#include "systems/prismatic/prismatic_mesh.h"
#include <cassert>

namespace Aperture {

prismatic_d1_local prismatic_d1_local::build(
    const prismatic_mesh& mesh, const prismatic_mesh_partition& mp) {
  prismatic_d1_local out;

  struct csr_stage {
    std::vector<int> row_ptr, col_idx;
    std::vector<Scalar> val;
  };
  csr_stage st_d1_tri_h, st_d1_rect_h, st_d1_rect_v, st_d1t_h_tri, st_d1t_h_rect, st_d1t_v_rect;
  out.m_partition = &mp;

  // Global flat-index offsets in the mesh's combined CSR.
  const int N_tri_faces = (mesh.m_N_r + 1) * mesh.m_N_tri;
  const int N_h_edges   = (mesh.m_N_r + 1) * mesh.m_N_edge_s;

  auto const& L_tri  = mp.layout(cochain_type::tri_face);
  auto const& L_rect = mp.layout(cochain_type::rect_face);
  auto const& L_he   = mp.layout(cochain_type::h_edge);
  auto const& L_ve   = mp.layout(cochain_type::v_edge);

  const int* d1_row_ptr  = mesh.d1_row_ptr.host_ptr();
  const int* d1_col_idx  = mesh.d1_col_idx.host_ptr();
  const Scalar* d1_val   = mesh.d1_val.host_ptr();
  const int* d1t_row_ptr = mesh.d1t_row_ptr.host_ptr();
  const int* d1t_col_idx = mesh.d1t_col_idx.host_ptr();
  const Scalar* d1t_val  = mesh.d1t_val.host_ptr();

  // ===== d1: tri faces → h edges =====
  st_d1_tri_h.row_ptr.assign(L_tri.owned_size() + 1, 0);
  for (int l = 0; l < L_tri.owned_size(); ++l) {
    const int g_face = L_tri.to_global(l);  // [0, N_tri_faces)
    const int row_start = d1_row_ptr[g_face];
    const int row_end   = d1_row_ptr[g_face + 1];
    for (int j = row_start; j < row_end; ++j) {
      const int g_edge = d1_col_idx[j];
      // Tri faces have only h-edge boundaries.
      assert(g_edge < N_h_edges);
      const int local_he = L_he.to_local(g_edge);
      assert(local_he >= 0 && "h_edge boundary of owned tri must be in halo");
      st_d1_tri_h.col_idx.push_back(local_he);
      st_d1_tri_h.val.push_back(d1_val[j]);
    }
    st_d1_tri_h.row_ptr[l + 1] = int(st_d1_tri_h.col_idx.size());
  }

  // ===== d1: rect faces → h and v edges =====
  st_d1_rect_h.row_ptr.assign(L_rect.owned_size() + 1, 0);
  st_d1_rect_v.row_ptr.assign(L_rect.owned_size() + 1, 0);
  for (int l = 0; l < L_rect.owned_size(); ++l) {
    const int g_rect = L_rect.to_global(l);            // [0, N_rect_faces)
    const int g_face = N_tri_faces + g_rect;           // combined-index face
    const int row_start = d1_row_ptr[g_face];
    const int row_end   = d1_row_ptr[g_face + 1];
    for (int j = row_start; j < row_end; ++j) {
      const int g_edge = d1_col_idx[j];
      if (g_edge < N_h_edges) {
        const int local_he = L_he.to_local(g_edge);
        assert(local_he >= 0 && "h_edge boundary of owned rect must be in halo");
        st_d1_rect_h.col_idx.push_back(local_he);
        st_d1_rect_h.val.push_back(d1_val[j]);
      } else {
        const int local_ve = L_ve.to_local(g_edge - N_h_edges);
        assert(local_ve >= 0 && "v_edge boundary of owned rect must be in halo");
        st_d1_rect_v.col_idx.push_back(local_ve);
        st_d1_rect_v.val.push_back(d1_val[j]);
      }
    }
    st_d1_rect_h.row_ptr[l + 1] = int(st_d1_rect_h.col_idx.size());
    st_d1_rect_v.row_ptr[l + 1] = int(st_d1_rect_v.col_idx.size());
  }

  // ===== d1^T: h edges → tri and rect faces =====
  st_d1t_h_tri.row_ptr.assign(L_he.owned_size() + 1, 0);
  st_d1t_h_rect.row_ptr.assign(L_he.owned_size() + 1, 0);
  for (int l = 0; l < L_he.owned_size(); ++l) {
    const int g_he   = L_he.to_global(l);  // [0, N_h_edges)
    const int g_edge = g_he;                // h_edges are at offset 0 in d1^T
    const int row_start = d1t_row_ptr[g_edge];
    const int row_end   = d1t_row_ptr[g_edge + 1];
    for (int j = row_start; j < row_end; ++j) {
      const int g_face = d1t_col_idx[j];
      if (g_face < N_tri_faces) {
        const int local_tri = L_tri.to_local(g_face);
        assert(local_tri >= 0 &&
               "tri face adjacent to owned h_edge must be in halo");
        st_d1t_h_tri.col_idx.push_back(local_tri);
        st_d1t_h_tri.val.push_back(d1t_val[j]);
      } else {
        const int local_rect = L_rect.to_local(g_face - N_tri_faces);
        assert(local_rect >= 0 &&
               "rect face adjacent to owned h_edge must be in halo");
        st_d1t_h_rect.col_idx.push_back(local_rect);
        st_d1t_h_rect.val.push_back(d1t_val[j]);
      }
    }
    st_d1t_h_tri.row_ptr[l + 1]  = int(st_d1t_h_tri.col_idx.size());
    st_d1t_h_rect.row_ptr[l + 1] = int(st_d1t_h_rect.col_idx.size());
  }

  // ===== d1^T: v edges → rect faces =====
  st_d1t_v_rect.row_ptr.assign(L_ve.owned_size() + 1, 0);
  for (int l = 0; l < L_ve.owned_size(); ++l) {
    const int g_ve   = L_ve.to_global(l);     // [0, N_v_edges)
    const int g_edge = N_h_edges + g_ve;       // v_edges in d1^T's combined space
    const int row_start = d1t_row_ptr[g_edge];
    const int row_end   = d1t_row_ptr[g_edge + 1];
    for (int j = row_start; j < row_end; ++j) {
      const int g_face = d1t_col_idx[j];
      // v_edges only border rect faces.
      assert(g_face >= N_tri_faces);
      const int local_rect = L_rect.to_local(g_face - N_tri_faces);
      assert(local_rect >= 0 &&
             "rect face adjacent to owned v_edge must be in halo");
      st_d1t_v_rect.col_idx.push_back(local_rect);
      st_d1t_v_rect.val.push_back(d1t_val[j]);
    }
    st_d1t_v_rect.row_ptr[l + 1] = int(st_d1t_v_rect.col_idx.size());
  }


  // Move the staged CSR blocks into the buffer-backed storage.
  auto commit = [](csr_stage& st, auto& blk) {
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
