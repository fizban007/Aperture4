#include "catch2/catch_all.hpp"
#include "systems/physics/spherical_metric.hpp"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_mesh_metric.h"
#include "systems/prismatic/prismatic_mesh_metric_local.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_partition.h"
#include <memory>
#include <vector>

using namespace Aperture;

namespace {

std::unique_ptr<prismatic_mesh_metric> make_metric_mesh(int L, int N_r,
                                                         double a) {
  auto mesh = std::make_unique<prismatic_mesh_metric>();
  mesh->build(L, N_r, 1.0, 2.0);
  mesh->compute_metric(ks_spherical_metric{Scalar(a)});
  return mesh;
}

template <typename LocalVec, typename GlobalPtr>
void check_match(const LocalVec& local, GlobalPtr global, int offset,
                  const distributed_cochain_layout& layout) {
  REQUIRE(int(local.size()) == layout.local_size());
  for (int l = 0; l < layout.local_size(); ++l) {
    REQUIRE(local[l] == global[offset + layout.to_global(l)]);
  }
}

}  // namespace

// =========================================================================
// Single-rank: local metric arrays equal global metric mesh entry by entry.
// =========================================================================
TEST_CASE("metric_local single-rank: values match global metric mesh",
          "[prismatic][mesh_metric_local]") {
  const int L = 2;
  const int N_r = 4;
  auto met = make_metric_mesh(L, N_r, 0.9);
  auto topo = icosphere_topology::build_from_mesh(*met);
  auto part = prismatic_partition::single_rank(L, N_r);
  part.set_topology(&topo);
  auto mp = prismatic_mesh_partition::build(part, topo);
  auto local = prismatic_mesh_metric_local::build(*met, mp);

  const int N_tri_faces = (met->m_N_r + 1) * met->m_N_tri;
  const int N_h_edges   = (met->m_N_r + 1) * met->m_N_edge_s;

  // Tri face metric.
  check_match(local.tri_face_alpha, met->face_alpha.host_ptr(), 0,
              mp.layout(cochain_type::tri_face));
  check_match(local.tri_face_sqrt_gamma, met->face_sqrt_gamma.host_ptr(), 0,
              mp.layout(cochain_type::tri_face));
  check_match(local.tri_face_sq_gamma_beta_r,
              met->face_sq_gamma_beta_r.host_ptr(), 0,
              mp.layout(cochain_type::tri_face));
  check_match(local.tri_face_r_coord, met->face_r_coord.host_ptr(), 0,
              mp.layout(cochain_type::tri_face));

  // Rect face metric (offset by N_tri_faces).
  check_match(local.rect_face_alpha, met->face_alpha.host_ptr(), N_tri_faces,
              mp.layout(cochain_type::rect_face));
  check_match(local.rect_face_sq_gamma_beta_r,
              met->face_sq_gamma_beta_r.host_ptr(), N_tri_faces,
              mp.layout(cochain_type::rect_face));

  // H edge metric.
  check_match(local.h_edge_alpha, met->edge_alpha.host_ptr(), 0,
              mp.layout(cochain_type::h_edge));
  check_match(local.h_edge_sq_gamma_beta_r,
              met->edge_sq_gamma_beta_r.host_ptr(), 0,
              mp.layout(cochain_type::h_edge));
  check_match(local.h_edge_r_coord, met->edge_r_coord.host_ptr(), 0,
              mp.layout(cochain_type::h_edge));

  // V edge metric (offset by N_h_edges).
  check_match(local.v_edge_alpha, met->edge_alpha.host_ptr(), N_h_edges,
              mp.layout(cochain_type::v_edge));
  check_match(local.v_edge_sqrt_gamma, met->edge_sqrt_gamma.host_ptr(),
              N_h_edges, mp.layout(cochain_type::v_edge));
  check_match(local.v_edge_sq_gamma_beta_r,
              met->edge_sq_gamma_beta_r.host_ptr(), N_h_edges,
              mp.layout(cochain_type::v_edge));
}

// =========================================================================
// Combined partition: per-rank local metric is correct and strictly smaller
// than global.
// =========================================================================
TEST_CASE("metric_local combined partition: shrinks + values match",
          "[prismatic][mesh_metric_local]") {
  const int L = 2;
  const int N_r = 8;
  const int K = 4;
  auto met = make_metric_mesh(L, N_r, 0.5);
  auto topo = icosphere_topology::build_from_mesh(*met);

  const int N_tri_faces = (met->m_N_r + 1) * met->m_N_tri;
  const int N_h_edges   = (met->m_N_r + 1) * met->m_N_edge_s;

  for (auto rk : std::vector<std::pair<int, int>>{
           {0, 7}, {1, 13}, {2, 19}, {3, 0}}) {
    auto part = prismatic_partition::combined_ico_face(L, N_r, K, rk.first, rk.second);
    part.set_topology(&topo);
    auto mp = prismatic_mesh_partition::build(part, topo);
    auto local = prismatic_mesh_metric_local::build(*met, mp);

    REQUIRE(int(local.tri_face_alpha.size()) <
            (met->m_N_r + 1) * met->m_N_tri);
    REQUIRE(int(local.h_edge_alpha.size()) <
            (met->m_N_r + 1) * met->m_N_edge_s);

    check_match(local.tri_face_alpha, met->face_alpha.host_ptr(), 0,
                mp.layout(cochain_type::tri_face));
    check_match(local.rect_face_sqrt_gamma, met->face_sqrt_gamma.host_ptr(),
                N_tri_faces, mp.layout(cochain_type::rect_face));
    check_match(local.h_edge_alpha, met->edge_alpha.host_ptr(), 0,
                mp.layout(cochain_type::h_edge));
    check_match(local.v_edge_alpha, met->edge_alpha.host_ptr(), N_h_edges,
                mp.layout(cochain_type::v_edge));
  }
}
