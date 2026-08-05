// Dump the discrete frame-drag operators for offline spectral analysis:
// d1 (face x edge CSR), W (edge x face CSR, the frame-drag EMF stencil),
// hodge diagonals, boundary flags.  Single rank, host policy.
#define private public
#include "systems/prismatic/dec_solver_dist.h"
#undef private
#include "systems/prismatic/dec_solver_geometry.hpp"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_partition.h"
#include <cstdio>
#include <cstdlib>

using namespace Aperture;
using policy = prismatic_exec_policy_host;
using core_t = dec_solver_dist<policy>;

static void dump_int(FILE* f, const char* name, const int* a, long n) {
  std::fprintf(f, "%s %ld\n", name, n);
  for (long i = 0; i < n; i++) std::fprintf(f, "%d\n", a[i]);
}
static void dump_sc(FILE* f, const char* name, const Scalar* a, long n) {
  std::fprintf(f, "%s %ld\n", name, n);
  for (long i = 0; i < n; i++) std::fprintf(f, "%.9g\n", double(a[i]));
}

int main(int argc, char** argv) {
  const int L = (argc > 1) ? atoi(argv[1]) : 3;
  const int N_r = (argc > 2) ? atoi(argv[2]) : 24;
  const double r_max = (argc > 3) ? atof(argv[3]) : 3.0;
  const char* out = (argc > 4) ? argv[4] : "fd_ops.txt";
  const Scalar w0 = 0.05;

  prismatic_mesh mesh;
  mesh.build(L, N_r, 1.0, Scalar(r_max));
  auto topo = icosphere_topology::build_from_mesh(mesh);
  auto part = prismatic_partition::single_rank(L, N_r);
  part.set_topology(&topo);
  auto mp_b = prismatic_mesh_partition::build(part, topo);
  core_t core;
  core.build(mesh, mp_b);
  core.build_frame_drag(w0, Scalar(1.0), 3);

  auto lp = core.get_lp(exec_tags::host{});
  FILE* f = std::fopen(out, "w");
  std::fprintf(f, "meta %d %d %g %d %d %d %d %d %d\n", L, N_r, r_max,
               lp.n_owned_he, lp.n_owned_ve, lp.n_owned_tri, lp.n_owned_rect,
               core.e_split(), core.b_split());
  // d1: face -> edge circulations (tri faces: h edges; rect: h + v)
  dump_int(f, "d1_tri_h_row", lp.d1_tri_h_row, lp.n_owned_tri + 1);
  dump_int(f, "d1_tri_h_col", lp.d1_tri_h_col,
           lp.d1_tri_h_row[lp.n_owned_tri]);
  dump_sc(f, "d1_tri_h_val", lp.d1_tri_h_val,
          lp.d1_tri_h_row[lp.n_owned_tri]);
  dump_int(f, "d1_rect_h_row", lp.d1_rect_h_row, lp.n_owned_rect + 1);
  dump_int(f, "d1_rect_h_col", lp.d1_rect_h_col,
           lp.d1_rect_h_row[lp.n_owned_rect]);
  dump_sc(f, "d1_rect_h_val", lp.d1_rect_h_val,
          lp.d1_rect_h_row[lp.n_owned_rect]);
  dump_int(f, "d1_rect_v_row", lp.d1_rect_v_row, lp.n_owned_rect + 1);
  dump_int(f, "d1_rect_v_col", lp.d1_rect_v_col,
           lp.d1_rect_v_row[lp.n_owned_rect]);
  dump_sc(f, "d1_rect_v_val", lp.d1_rect_v_val,
          lp.d1_rect_v_row[lp.n_owned_rect]);
  // W: edge -> sum over adjacent faces (d1t sparsity + fd values)
  dump_int(f, "d1t_h_tri_row", lp.d1t_h_tri_row, lp.n_owned_he + 1);
  dump_int(f, "d1t_h_tri_col", lp.d1t_h_tri_col,
           lp.d1t_h_tri_row[lp.n_owned_he]);
  dump_sc(f, "fd_h_tri_val", core.m_fd_h_tri_val.host_ptr(),
          lp.d1t_h_tri_row[lp.n_owned_he]);
  dump_int(f, "d1t_h_rect_row", lp.d1t_h_rect_row, lp.n_owned_he + 1);
  dump_int(f, "d1t_h_rect_col", lp.d1t_h_rect_col,
           lp.d1t_h_rect_row[lp.n_owned_he]);
  dump_sc(f, "fd_h_rect_val", core.m_fd_h_rect_val.host_ptr(),
          lp.d1t_h_rect_row[lp.n_owned_he]);
  dump_int(f, "d1t_v_rect_row", lp.d1t_v_rect_row, lp.n_owned_ve + 1);
  dump_int(f, "d1t_v_rect_col", lp.d1t_v_rect_col,
           lp.d1t_v_rect_row[lp.n_owned_ve]);
  dump_sc(f, "fd_v_rect_val", core.m_fd_v_rect_val.host_ptr(),
          lp.d1t_v_rect_row[lp.n_owned_ve]);
  // hodges + boundary flags
  dump_sc(f, "tri_face_hodge2", lp.tri_face_hodge2, lp.n_owned_tri);
  dump_sc(f, "rect_face_hodge2", lp.rect_face_hodge2, lp.n_owned_rect);
  dump_sc(f, "h_edge_hodge1_inv", lp.h_edge_hodge1_inv, lp.n_owned_he);
  dump_sc(f, "v_edge_hodge1_inv", lp.v_edge_hodge1_inv, lp.n_owned_ve);
  dump_int(f, "h_edge_boundary", lp.h_edge_boundary, lp.n_owned_he);
  dump_int(f, "v_edge_boundary", lp.v_edge_boundary, lp.n_owned_ve);
  dump_int(f, "tri_face_boundary", lp.tri_face_boundary, lp.n_owned_tri);
  dump_int(f, "rect_face_boundary", lp.rect_face_boundary, lp.n_owned_rect);
  std::fclose(f);
  std::printf("dumped %s\n", out);
  return 0;
}
