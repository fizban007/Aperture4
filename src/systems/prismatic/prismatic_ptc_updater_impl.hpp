#pragma once

#include "systems/prismatic/prismatic_ptc_updater.h"
#include "systems/prismatic/prismatic_ptc_update_kernel.hpp"
#include "core/particles_functions.h"
#include "framework/environment.h"
#include "utils/logger.h"
#include <cmath>

namespace Aperture {

template <typename ExecPolicy>
prismatic_ptc_updater<ExecPolicy>::prismatic_ptc_updater(prismatic_mesh& mesh)
    : m_mesh(mesh) {}

template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::register_data_components() {
  auto mem = ExecPolicy::data_mem_type();
  m_E = sim_env().template register_data<prismatic_edge_field>("E", m_mesh, mem);
  m_B = sim_env().template register_data<prismatic_face_field>("B", m_mesh, mem);
  m_J = sim_env().template register_data<prismatic_edge_field>("J", m_mesh, mem);
  m_rho = sim_env().template register_data<prismatic_vertex_field>("rho", m_mesh, mem);

  int max_ptc = 100000;
  sim_env().params().get_value("max_ptc_num", max_ptc);
  m_ptc = sim_env().template register_data<prismatic_particle_data>(
      "particles", max_ptc, mem);
}

template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::init() {
  sim_env().params().get_value("q_e", m_charge_e);
  sim_env().params().get_value("m_e", m_mass_e);
  sim_env().params().get_value("sort_interval", m_sort_interval);
  sim_env().params().get_value("use_gca", m_use_gca);
  sim_env().params().get_value("include_curvature", m_include_curvature);
  sim_env().params().get_value("use_recovery_gather", m_use_recovery_gather);
  if (m_use_recovery_gather) {
    m_recovery.build(m_mesh);
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    m_recovery.copy_to_device();
#endif
  }
  Logger::print_info("Prismatic particle updater initialized: {} particles",
                     m_ptc->size());
}

template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::update(double dt, uint32_t step) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  int N_tri = mp.N_tri;
  size_t num = m_ptc->number();
  Scalar charge_e = m_charge_e;
  Scalar mass_e = m_mass_e;

  // Clear rho and J before deposit
  ExecPolicy::launch(
      [N_edges = mp.N_edges, N_verts = mp.N_verts]
      LAMBDA(auto J_e, auto rho) {
        ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
          J_e[e] = Scalar(0);
        });
        ExecPolicy::loop(0, N_verts, [&] LAMBDA(int v) {
          rho[v] = Scalar(0);
        });
      },
      m_J->data(), m_rho->data());

  // Refresh the recovery vertex field from the current B cochain (one
  // fitted B vector per mesh vertex; see prismatic_vertex_recovery.h).
  const Scalar* Bv_rec = nullptr;
  if (m_use_recovery_gather) {
    auto rp = m_recovery.get_ptrs(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [mp, rp] LAMBDA(auto B_f) {
          ExecPolicy::loop(0, rp.N_verts, [&] LAMBDA(int vi) {
            rp.compute_vertex_B(mp, B_f, vi);
          });
        },
        m_B->data());
    ExecPolicy::sync();
    Bv_rec = rp.Bv;
  }

  // Particle update loop
  bool use_gca = m_use_gca;
  bool include_curvature = m_include_curvature;
  ExecPolicy::launch(
      [num, N_tri, charge_e, mass_e, dt, mp, use_gca, include_curvature,
       Bv_rec]
      LAMBDA(auto ptc, auto E_e, auto B_f, auto J_e, auto rho) {
        ExecPolicy::loop(0, (int)num, [&] LAMBDA(int n) {
          if (ptc.cell[n] == empty_cell) return;
          int sp = get_ptc_type(ptc.flag[n]);
          Scalar q = (sp == (int)PtcType::positron) ? -charge_e : charge_e;
          update_single_particle(mp, N_tri, ptc, n, E_e, B_f, J_e, rho,
                                 q, mass_e, Scalar(dt),
                                 use_gca, include_curvature, Bv_rec);
        });
      },
      *m_ptc, m_E->data(), m_B->data(), m_J->data(), m_rho->data());

  ExecPolicy::sync();

  // Periodically sort particles by cell for GPU cache efficiency
  if (m_sort_interval > 0 && step % m_sort_interval == 0) {
    size_t max_cell = m_mesh.m_N_tri * m_mesh.m_N_r;
    ptc_sort_by_cell(typename ExecPolicy::exec_tag{}, *m_ptc, max_cell);
  }
}

template <typename ExecPolicy>
int prismatic_ptc_updater<ExecPolicy>::add_particle(
    Scalar x, Scalar y, Scalar z, Scalar px, Scalar py, Scalar pz,
    Scalar weight, uint32_t flag) {
  auto mp = m_mesh.host_ptrs();
  int tri_idx, layer_idx;
  Scalar l1, l2, zeta;
  if (!cartesian_to_local_impl(mp, x, y, z, tri_idx, layer_idx, l1, l2, zeta))
    return -1;

  size_t idx = m_ptc->number();
  if (idx >= m_ptc->size()) return -1;

  auto ptrs = m_ptc->get_host_ptrs();
  ptrs.x1[idx] = l1; ptrs.x2[idx] = l2; ptrs.x3[idx] = zeta;
  ptrs.p1[idx] = px; ptrs.p2[idx] = py; ptrs.p3[idx] = pz;
  ptrs.E[idx] = std::sqrt(Scalar(1) + px*px + py*py + pz*pz);
  ptrs.weight[idx] = weight;
  ptrs.cell[idx] = prism_cell_encode(tri_idx, layer_idx, m_mesh.m_N_tri);
  ptrs.flag[idx] = flag;
  ptrs.id[idx] = idx;
  m_ptc->set_num(idx + 1);
  return static_cast<int>(idx);
}

}  // namespace Aperture
