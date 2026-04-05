#include "systems/prismatic/prismatic_ptc_updater.h"
#include "systems/prismatic/prismatic_deposit.h"
#include "framework/environment.h"
#include "utils/logger.h"
#include "utils/util_functions.h"
#include <cmath>

namespace Aperture {

// =========================================================================
// HD_INLINE coordinate conversion helpers (using mesh_ptrs)
// =========================================================================

HD_INLINE void local_to_cartesian_impl(
    const prismatic_mesh_ptrs& mp, int tri_idx, int layer_idx,
    Scalar l1, Scalar l2, Scalar zeta,
    Scalar& x, Scalar& y, Scalar& z) {
  Scalar l3 = Scalar(1) - l1 - l2;
  int v0 = mp.tri_verts[tri_idx * 3 + 0];
  int v1 = mp.tri_verts[tri_idx * 3 + 1];
  int v2 = mp.tri_verts[tri_idx * 3 + 2];
  Scalar sx = l1*mp.sphere_vx[v0] + l2*mp.sphere_vx[v1] + l3*mp.sphere_vx[v2];
  Scalar sy = l1*mp.sphere_vy[v0] + l2*mp.sphere_vy[v1] + l3*mp.sphere_vy[v2];
  Scalar sz = l1*mp.sphere_vz[v0] + l2*mp.sphere_vz[v1] + l3*mp.sphere_vz[v2];
  Scalar s_inv = Scalar(1) / std::sqrt(sx*sx + sy*sy + sz*sz);
  sx *= s_inv; sy *= s_inv; sz *= s_inv;
  Scalar r = mp.radii[layer_idx] +
             zeta * (mp.radii[layer_idx + 1] - mp.radii[layer_idx]);
  x = r * sx; y = r * sy; z = r * sz;
}

// Returns false if point is outside the mesh.
HD_INLINE bool cartesian_to_local_impl(
    const prismatic_mesh_ptrs& mp,
    Scalar x, Scalar y, Scalar z,
    int& tri_idx, int& layer_idx,
    Scalar& l1, Scalar& l2, Scalar& zeta,
    int tri_hint = -1) {
  Scalar r = std::sqrt(x*x + y*y + z*z);
  layer_idx = mp.find_radial_layer(r);
  if (layer_idx < 0) return false;
  zeta = mp.compute_zeta(layer_idx, r);
  Scalar ri = Scalar(1) / r;
  Scalar sx = x*ri, sy = y*ri, sz = z*ri;
  tri_idx = mp.find_triangle(sx, sy, sz, tri_hint);
  Scalar l3;
  mp.compute_barycentric(tri_idx, sx, sy, sz, l1, l2, l3);
  return true;
}

// =========================================================================
// Constructor and init
// =========================================================================

prismatic_ptc_updater::prismatic_ptc_updater(prismatic_mesh& mesh,
                                             buffer<Scalar>& E_e,
                                             buffer<Scalar>& B_f,
                                             buffer<Scalar>& J_e)
    : m_mesh(mesh), m_E_e(E_e), m_B_f(B_f), m_J_e(J_e),
      m_rho(mesh.m_N_verts, MemType::host_only) {}

void prismatic_ptc_updater::init() {
  int max_ptc = 100000;
  sim_env().params().get_value("max_ptc_num", max_ptc);
  sim_env().params().get_value("q_e", m_charge_e);
  sim_env().params().get_value("m_e", m_mass_e);

  m_particles = prismatic_particles_t(max_ptc, MemType::host_only);
  m_particles.init();

  m_rho.assign(0, m_mesh.m_N_verts, Scalar(0));

  Logger::print_info("Prismatic particle updater initialized: max_ptc={}",
                     max_ptc);
}

// =========================================================================
// Main update loop
// =========================================================================

void prismatic_ptc_updater::update(double dt, uint32_t step) {
  auto ptrs = m_particles.get_host_ptrs();
  size_t num = m_particles.number();
  int N_r = m_mesh.m_N_r;
  auto mp = m_mesh.host_ptrs();

  const Scalar* E_e = m_E_e.host_ptr();
  const Scalar* B_f = m_B_f.host_ptr();
  Scalar* J_e = m_J_e.host_ptr();
  Scalar* rho = m_rho.host_ptr();

  // Clear rho and J before deposit
  m_rho.assign(0, m_mesh.m_N_verts, Scalar(0));
  m_J_e.assign(exec_tags::host{}, 0, m_mesh.m_N_edges, Scalar(0));


  for (size_t n = 0; n < num; n++) {
    if (ptrs.cell[n] == empty_cell) continue;

    int tri_idx, layer_idx;
    prism_cell_decode(ptrs.cell[n], N_r, tri_idx, layer_idx);
    Scalar l1 = ptrs.x1[n], l2 = ptrs.x2[n];
    Scalar l3 = Scalar(1) - l1 - l2;
    Scalar zeta = ptrs.x3[n];



    // 1. Interpolate E, B at particle position
    Scalar l[3] = {l1, l2, l3};
    Scalar Ex, Ey, Ez, Bx, By, Bz;
    interpolate_fields(mp, tri_idx, layer_idx, l, zeta,
                       E_e, B_f, Ex, Ey, Ez, Bx, By, Bz);

    // 2. Boris push
    int sp = get_ptc_type(ptrs.flag[n]);
    Scalar q = (sp == (int)PtcType::positron) ? -m_charge_e : m_charge_e;
    Scalar m = m_mass_e;
    Scalar qdt_2m = q * Scalar(dt) / (Scalar(2) * m);

    Scalar px = ptrs.p1[n] + qdt_2m * Ex;
    Scalar py = ptrs.p2[n] + qdt_2m * Ey;
    Scalar pz = ptrs.p3[n] + qdt_2m * Ez;

    Scalar gamma_mid = std::sqrt(Scalar(1) + px*px + py*py + pz*pz);
    Scalar tx = qdt_2m * Bx / gamma_mid;
    Scalar ty = qdt_2m * By / gamma_mid;
    Scalar tz = qdt_2m * Bz / gamma_mid;
    Scalar t2 = tx*tx + ty*ty + tz*tz;
    Scalar sx = Scalar(2)*tx / (Scalar(1)+t2);
    Scalar sy = Scalar(2)*ty / (Scalar(1)+t2);
    Scalar sz = Scalar(2)*tz / (Scalar(1)+t2);

    Scalar ppx = px + (py*tz - pz*ty);
    Scalar ppy = py + (pz*tx - px*tz);
    Scalar ppz = pz + (px*ty - py*tx);

    px += (ppy*sz - ppz*sy);
    py += (ppz*sx - ppx*sz);
    pz += (ppx*sy - ppy*sx);

    px += qdt_2m * Ex;
    py += qdt_2m * Ey;
    pz += qdt_2m * Ez;

    ptrs.p1[n] = px; ptrs.p2[n] = py; ptrs.p3[n] = pz;
    Scalar gamma = std::sqrt(Scalar(1) + px*px + py*py + pz*pz);
    ptrs.E[n] = gamma;

    // 3. Position update
    Scalar old_x, old_y, old_z;
    local_to_cartesian_impl(mp, tri_idx, layer_idx, l1, l2, zeta,
                            old_x, old_y, old_z);

    Scalar vx = px / gamma, vy = py / gamma, vz = pz / gamma;
    Scalar new_x = old_x + vx * Scalar(dt);
    Scalar new_y = old_y + vy * Scalar(dt);
    Scalar new_z = old_z + vz * Scalar(dt);

    int new_tri, new_layer;
    Scalar new_l1, new_l2, new_zeta;
    if (!cartesian_to_local_impl(mp, new_x, new_y, new_z,
                                 new_tri, new_layer, new_l1, new_l2,
                                 new_zeta, tri_idx)) {
      ptrs.cell[n] = empty_cell;
      continue;
    }

    // 4. Current deposition
    if (!check_flag(ptrs.flag[n], PtcFlag::ignore_current)) {
      Scalar l_old[3] = {l1, l2, l3};
      Scalar l_new[3] = {new_l1, new_l2, Scalar(1) - new_l1 - new_l2};
      Scalar zeta_new_in_old;
      if (new_tri == tri_idx && new_layer == layer_idx) {
        zeta_new_in_old = new_zeta;
      } else {
        Scalar r_new = std::sqrt(new_x*new_x + new_y*new_y + new_z*new_z);
        Scalar dr_old = mp.radii[layer_idx + 1] - mp.radii[layer_idx];
        zeta_new_in_old = (r_new - mp.radii[layer_idx]) / dr_old;
        Scalar ri = Scalar(1) / r_new;
        mp.compute_barycentric(tri_idx, new_x*ri, new_y*ri, new_z*ri,
                               l_new[0], l_new[1], l_new[2]);
      }
      int dep_tri, dep_layer;
      Scalar q_over_dt = q * ptrs.weight[n] / Scalar(dt);
      deposit_current(mp, tri_idx, layer_idx,
                      l_old, zeta, l_new, zeta_new_in_old,
                      q_over_dt, J_e, dep_tri, dep_layer);
    }

    // 5. Charge density deposit at new position
    {
      Scalar l_new[3] = {new_l1, new_l2, Scalar(1) - new_l1 - new_l2};
      Scalar q_w = q * ptrs.weight[n];
      deposit_rho(mp, new_tri, new_layer, l_new, new_zeta, q_w, rho);
    }

    // 6. Store new position
    ptrs.x1[n] = new_l1;
    ptrs.x2[n] = new_l2;
    ptrs.x3[n] = new_zeta;
    ptrs.cell[n] = prism_cell_encode(new_tri, new_layer, N_r);
  }
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

  size_t idx = m_particles.number();
  if (idx >= m_particles.size()) return -1;

  auto ptrs = m_particles.get_host_ptrs();
  ptrs.x1[idx] = l1; ptrs.x2[idx] = l2; ptrs.x3[idx] = zeta;
  ptrs.p1[idx] = px; ptrs.p2[idx] = py; ptrs.p3[idx] = pz;
  ptrs.E[idx] = std::sqrt(Scalar(1) + px*px + py*py + pz*pz);
  ptrs.weight[idx] = weight;
  ptrs.cell[idx] = prism_cell_encode(tri_idx, layer_idx, m_mesh.m_N_r);
  ptrs.flag[idx] = flag;
  ptrs.id[idx] = idx;

  m_particles.set_num(idx + 1);
  return static_cast<int>(idx);
}

// =========================================================================
// Remove dead particles
// =========================================================================

void prismatic_ptc_updater::remove_dead_particles() {
  auto ptrs = m_particles.get_host_ptrs();
  size_t num = m_particles.number();
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
  m_particles.set_num(write);
}

}  // namespace Aperture
