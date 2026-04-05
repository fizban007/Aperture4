#include "systems/prismatic/prismatic_ptc_updater.h"
#include "systems/prismatic/prismatic_ptc_update_kernel.hpp"
#include "framework/environment.h"
#include "utils/logger.h"
#include <cmath>

namespace Aperture {

// =========================================================================
// Constructor and data registration
// =========================================================================

prismatic_ptc_updater::prismatic_ptc_updater(prismatic_mesh& mesh)
    : m_mesh(mesh) {}

void prismatic_ptc_updater::register_data_components() {
  auto mem = MemType::host_only;  // CPU particles for now
  m_E = sim_env().register_data<prismatic_edge_field>("E", m_mesh, mem);
  m_B = sim_env().register_data<prismatic_face_field>("B", m_mesh, mem);
  m_J = sim_env().register_data<prismatic_edge_field>("J", m_mesh, mem);
  m_rho = sim_env().register_data<prismatic_vertex_field>("rho", m_mesh, mem);

  int max_ptc = 100000;
  sim_env().params().get_value("max_ptc_num", max_ptc);
  m_ptc = sim_env().register_data<prismatic_particle_data>(
      "particles", max_ptc, mem);
}

void prismatic_ptc_updater::init() {
  sim_env().params().get_value("q_e", m_charge_e);
  sim_env().params().get_value("m_e", m_mass_e);
  Logger::print_info("Prismatic particle updater initialized: {} particles",
                     m_ptc->size());
}

// =========================================================================
// Main update loop
// =========================================================================

void prismatic_ptc_updater::update(double dt, uint32_t step) {
  auto ptrs = m_ptc->get_host_ptrs();
  size_t num = m_ptc->number();
  auto mp = m_mesh.host_ptrs();

  // Clear rho and J before deposit
  m_rho->data().assign(exec_tags::host{}, 0, m_mesh.m_N_verts, Scalar(0));
  m_J->data().assign(exec_tags::host{}, 0, m_mesh.m_N_edges, Scalar(0));

  update_particles_loop(mp, m_mesh.m_N_r, ptrs, num,
                        m_E->host_ptr(), m_B->host_ptr(),
                        m_J->host_ptr(), m_rho->host_ptr(),
                        m_charge_e, m_mass_e, dt);
}

// =========================================================================
// Add particle
// =========================================================================

int prismatic_ptc_updater::add_particle(Scalar x, Scalar y, Scalar z,
                                        Scalar px, Scalar py, Scalar pz,
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
  ptrs.cell[idx] = prism_cell_encode(tri_idx, layer_idx, m_mesh.m_N_r);
  ptrs.flag[idx] = flag;
  ptrs.id[idx] = idx;
  m_ptc->set_num(idx + 1);
  return static_cast<int>(idx);
}

void prismatic_ptc_updater::remove_dead_particles() {
  auto ptrs = m_ptc->get_host_ptrs();
  size_t num = m_ptc->number();
  size_t write = 0;
  for (size_t read = 0; read < num; read++) {
    if (ptrs.cell[read] != empty_cell) {
      if (write != read) {
        single_prism_ptc_t p;
        assign_ptc(p, ptrs, read);
        assign_ptc(ptrs, write, p);
      }
      write++;
    }
  }
  m_ptc->set_num(write);
}

}  // namespace Aperture
