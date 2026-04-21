#include "systems/prismatic/prismatic_mesh_partition.h"

namespace Aperture {

prismatic_mesh_partition prismatic_mesh_partition::build(
    const prismatic_partition& part, const icosphere_topology& topo) {
  prismatic_mesh_partition out;
  out.m_partition = part;
  out.m_topology = &topo;
  // Ensure the partition's topology pointer is set so ownership queries
  // can use the full-fidelity sphere-edge / sphere-vertex path rather
  // than the owns_all_angular() fallback.
  out.m_partition.set_topology(&topo);

  // Build plans (global-indexed) for each cochain type on both axes.
  auto fill_plans = [&](cochain_type t, plans_pair& pp) {
    pp.angular_global = build_angular_halo_plan(t, out.m_partition, topo);
    pp.radial_global  = build_radial_halo_plan(t, out.m_partition);
  };
  fill_plans(cochain_type::tri_face,  out.m_tri_face_plans);
  fill_plans(cochain_type::rect_face, out.m_rect_face_plans);
  fill_plans(cochain_type::h_edge,    out.m_h_edge_plans);
  fill_plans(cochain_type::v_edge,    out.m_v_edge_plans);
  fill_plans(cochain_type::vertex,    out.m_vertex_plans);

  // Build layouts, gathering ghosts from both angular and radial plans.
  auto build_layout = [&](cochain_type t, distributed_cochain_layout& lay,
                          const plans_pair& pp) {
    lay = distributed_cochain_layout::build(
        t, out.m_partition, &topo, {&pp.angular_global, &pp.radial_global});
  };
  build_layout(cochain_type::tri_face,  out.m_tri_face,  out.m_tri_face_plans);
  build_layout(cochain_type::rect_face, out.m_rect_face, out.m_rect_face_plans);
  build_layout(cochain_type::h_edge,    out.m_h_edge,    out.m_h_edge_plans);
  build_layout(cochain_type::v_edge,    out.m_v_edge,    out.m_v_edge_plans);
  build_layout(cochain_type::vertex,    out.m_vertex,    out.m_vertex_plans);

  // Translate plans to local indices using the just-built layouts.
  auto localize_plans = [&](cochain_type t,
                            const distributed_cochain_layout& lay,
                            plans_pair& pp) {
    (void)t;
    pp.angular_local = lay.localize(pp.angular_global);
    pp.radial_local  = lay.localize(pp.radial_global);
  };
  localize_plans(cochain_type::tri_face,  out.m_tri_face,  out.m_tri_face_plans);
  localize_plans(cochain_type::rect_face, out.m_rect_face, out.m_rect_face_plans);
  localize_plans(cochain_type::h_edge,    out.m_h_edge,    out.m_h_edge_plans);
  localize_plans(cochain_type::v_edge,    out.m_v_edge,    out.m_v_edge_plans);
  localize_plans(cochain_type::vertex,    out.m_vertex,    out.m_vertex_plans);

  return out;
}

const halo_plan& prismatic_mesh_partition::angular_plan_global(
    cochain_type t) const {
  return plans(t).angular_global;
}

const halo_plan& prismatic_mesh_partition::radial_plan_global(
    cochain_type t) const {
  return plans(t).radial_global;
}

const halo_plan& prismatic_mesh_partition::angular_plan_local(
    cochain_type t) const {
  return plans(t).angular_local;
}

const halo_plan& prismatic_mesh_partition::radial_plan_local(
    cochain_type t) const {
  return plans(t).radial_local;
}

}  // namespace Aperture
