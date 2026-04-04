#include "systems/prismatic/prismatic_ptc_updater.h"
#include "systems/prismatic/prismatic_deposit.h"
#include "framework/environment.h"
#include "utils/logger.h"
#include "utils/util_functions.h"
#include <cmath>

namespace Aperture {

prismatic_ptc_updater::prismatic_ptc_updater(prismatic_mesh& mesh,
                                             buffer<Scalar>& E_e,
                                             buffer<Scalar>& B_f,
                                             buffer<Scalar>& J_e)
    : m_mesh(mesh), m_E_e(E_e), m_B_f(B_f), m_J_e(J_e) {}

void prismatic_ptc_updater::init() {
  int max_ptc = 100000;
  sim_env().params().get_value("max_ptc_num", max_ptc);
  sim_env().params().get_value("q_e", m_charge_e);
  sim_env().params().get_value("m_e", m_mass_e);

  m_particles.resize(max_ptc);
  m_particles.set_memtype(MemType::host_only);
  m_particles.init();

  Logger::print_info("Prismatic particle updater initialized: max_ptc={}",
                     max_ptc);
}

void prismatic_ptc_updater::update(double dt, uint32_t step) {
  auto ptrs = m_particles.get_host_ptrs();
  size_t num = m_particles.number();
  int N_r = m_mesh.m_N_r;

  // Get field data pointers and mesh ptrs
  const Scalar* E_e = m_E_e.host_ptr();
  const Scalar* B_f = m_B_f.host_ptr();
  Scalar* J_e = m_J_e.host_ptr();
  auto mp = m_mesh.host_ptrs();

  for (size_t n = 0; n < num; n++) {
    if (ptrs.cell[n] == empty_cell) continue;

    // Decode current position
    int tri_idx, layer_idx;
    prism_cell_decode(ptrs.cell[n], N_r, tri_idx, layer_idx);
    Scalar l1 = ptrs.x1[n], l2 = ptrs.x2[n];
    Scalar l3 = 1.0f - l1 - l2;
    Scalar zeta = ptrs.x3[n];

    // -----------------------------------------------------------
    // 1. Interpolate E, B at particle position
    // -----------------------------------------------------------
    Scalar l[3] = {l1, l2, l3};
    Scalar Ex, Ey, Ez, Bx, By, Bz;
    interpolate_fields(mp, tri_idx, layer_idx, l, zeta,
                       E_e, B_f, Ex, Ey, Ez, Bx, By, Bz);

    // -----------------------------------------------------------
    // 2. Boris push (Cartesian momentum)
    // -----------------------------------------------------------
    // Determine charge and mass from species flag
    int sp = get_ptc_type(ptrs.flag[n]);
    Scalar q = (sp == (int)PtcType::positron) ? -m_charge_e : m_charge_e;
    Scalar m = m_mass_e;
    Scalar qdt_over_2m = q * dt / (2.0f * m);

    // Half-acceleration with E
    Scalar px = ptrs.p1[n] + qdt_over_2m * Ex;
    Scalar py = ptrs.p2[n] + qdt_over_2m * Ey;
    Scalar pz = ptrs.p3[n] + qdt_over_2m * Ez;

    // Rotation with B
    Scalar gamma_mid = std::sqrt(1.0f + px*px + py*py + pz*pz);
    Scalar tx = qdt_over_2m * Bx / gamma_mid;
    Scalar ty = qdt_over_2m * By / gamma_mid;
    Scalar tz = qdt_over_2m * Bz / gamma_mid;
    Scalar t2 = tx*tx + ty*ty + tz*tz;
    Scalar sx = 2.0f * tx / (1.0f + t2);
    Scalar sy = 2.0f * ty / (1.0f + t2);
    Scalar sz = 2.0f * tz / (1.0f + t2);

    // p' = p + p x t
    Scalar ppx = px + (py*tz - pz*ty);
    Scalar ppy = py + (pz*tx - px*tz);
    Scalar ppz = pz + (px*ty - py*tx);

    // p = p_minus + p' x s
    px += (ppy*sz - ppz*sy);
    py += (ppz*sx - ppx*sz);
    pz += (ppx*sy - ppy*sx);

    // Second half-acceleration with E
    px += qdt_over_2m * Ex;
    py += qdt_over_2m * Ey;
    pz += qdt_over_2m * Ez;

    // Update stored momentum
    ptrs.p1[n] = px;
    ptrs.p2[n] = py;
    ptrs.p3[n] = pz;
    Scalar gamma = std::sqrt(1.0f + px*px + py*py + pz*pz);
    ptrs.E[n] = gamma;

    // -----------------------------------------------------------
    // 3. Position update: compute new 3D position
    // -----------------------------------------------------------
    Scalar old_x, old_y, old_z;
    local_to_cartesian(tri_idx, layer_idx, l1, l2, zeta,
                       old_x, old_y, old_z);

    Scalar vx = px / gamma;
    Scalar vy = py / gamma;
    Scalar vz = pz / gamma;

    Scalar new_x = old_x + vx * dt;
    Scalar new_y = old_y + vy * dt;
    Scalar new_z = old_z + vz * dt;

    // Convert new position to local coordinates
    int new_tri, new_layer;
    Scalar new_l1, new_l2, new_zeta;
    if (!cartesian_to_local(new_x, new_y, new_z,
                            new_tri, new_layer, new_l1, new_l2, new_zeta,
                            tri_idx)) {
      // Particle left the domain — mark for removal
      ptrs.cell[n] = empty_cell;
      continue;
    }

    // -----------------------------------------------------------
    // 4. Current deposition
    // -----------------------------------------------------------
    if (!check_flag(ptrs.flag[n], PtcFlag::ignore_current)) {
      // Compute old and new coordinates in the starting prism
      Scalar l_old[3] = {l1, l2, l3};
      Scalar l_new[3] = {new_l1, new_l2, 1.0f - new_l1 - new_l2};

      // If the particle stayed in the same triangle, deposit directly.
      // Otherwise, we need to express the new position in the old triangle's
      // coordinates for the multi-prism deposit_current to work.
      Scalar zeta_new_in_old;
      if (new_tri == tri_idx && new_layer == layer_idx) {
        zeta_new_in_old = new_zeta;
      } else {
        // Express the new position's radius in the old layer's zeta
        Scalar r_new = std::sqrt(new_x*new_x + new_y*new_y + new_z*new_z);
        Scalar dr_old = m_mesh.radii[layer_idx + 1] - m_mesh.radii[layer_idx];
        zeta_new_in_old = (r_new - m_mesh.radii[layer_idx]) / dr_old;

        // Express new angular position in old triangle's barycentric coords
        Scalar r_inv = 1.0f / r_new;
        Scalar nsx = new_x * r_inv, nsy = new_y * r_inv, nsz = new_z * r_inv;
        mp.compute_barycentric(tri_idx, nsx, nsy, nsz,
                               l_new[0], l_new[1], l_new[2]);
      }

      int dep_tri, dep_layer;
      Scalar q_over_dt = q * ptrs.weight[n] / dt;
      deposit_current(mp, tri_idx, layer_idx,
                      l_old, zeta, l_new, zeta_new_in_old,
                      q_over_dt, J_e, dep_tri, dep_layer);
    }

    // -----------------------------------------------------------
    // 5. Store new position
    // -----------------------------------------------------------
    ptrs.x1[n] = new_l1;
    ptrs.x2[n] = new_l2;
    ptrs.x3[n] = new_zeta;
    ptrs.cell[n] = prism_cell_encode(new_tri, new_layer, N_r);
  }
}

bool prismatic_ptc_updater::cartesian_to_local(
    Scalar x, Scalar y, Scalar z,
    int& tri_idx, int& layer_idx,
    Scalar& l1, Scalar& l2, Scalar& zeta,
    int tri_hint) const {
  Scalar r = std::sqrt(x*x + y*y + z*z);
  layer_idx = m_mesh.find_radial_layer(r);
  if (layer_idx < 0) return false;

  zeta = m_mesh.compute_zeta(layer_idx, r);

  // Project to unit sphere and find triangle
  Scalar r_inv = 1.0f / r;
  Scalar sx = x * r_inv, sy = y * r_inv, sz = z * r_inv;
  tri_idx = m_mesh.find_triangle(sx, sy, sz, tri_hint);

  Scalar l3;
  m_mesh.compute_barycentric(tri_idx, sx, sy, sz, l1, l2, l3);
  return true;
}

void prismatic_ptc_updater::local_to_cartesian(
    int tri_idx, int layer_idx,
    Scalar l1, Scalar l2, Scalar zeta,
    Scalar& x, Scalar& y, Scalar& z) const {
  Scalar l3 = 1.0f - l1 - l2;
  int v0 = m_mesh.tri_verts[tri_idx * 3 + 0];
  int v1 = m_mesh.tri_verts[tri_idx * 3 + 1];
  int v2 = m_mesh.tri_verts[tri_idx * 3 + 2];

  // Angular position on unit sphere
  Scalar sx = l1 * m_mesh.sphere_vx[v0] + l2 * m_mesh.sphere_vx[v1] +
              l3 * m_mesh.sphere_vx[v2];
  Scalar sy = l1 * m_mesh.sphere_vy[v0] + l2 * m_mesh.sphere_vy[v1] +
              l3 * m_mesh.sphere_vy[v2];
  Scalar sz = l1 * m_mesh.sphere_vz[v0] + l2 * m_mesh.sphere_vz[v1] +
              l3 * m_mesh.sphere_vz[v2];

  // Normalize to unit sphere (barycentric interpolation doesn't preserve |v|=1)
  Scalar s_inv = 1.0f / std::sqrt(sx*sx + sy*sy + sz*sz);
  sx *= s_inv; sy *= s_inv; sz *= s_inv;

  // Radial position
  Scalar r = m_mesh.radii[layer_idx] +
             zeta * (m_mesh.radii[layer_idx + 1] - m_mesh.radii[layer_idx]);

  x = r * sx;
  y = r * sy;
  z = r * sz;
}

int prismatic_ptc_updater::add_particle(Scalar x, Scalar y, Scalar z,
                                        Scalar px, Scalar py, Scalar pz,
                                        Scalar weight, uint32_t flag) {
  int tri_idx, layer_idx;
  Scalar l1, l2, zeta;
  if (!cartesian_to_local(x, y, z, tri_idx, layer_idx, l1, l2, zeta))
    return -1;

  size_t idx = m_particles.number();
  if (idx >= m_particles.size()) return -1;

  auto ptrs = m_particles.get_host_ptrs();
  ptrs.x1[idx] = l1;
  ptrs.x2[idx] = l2;
  ptrs.x3[idx] = zeta;
  ptrs.p1[idx] = px;
  ptrs.p2[idx] = py;
  ptrs.p3[idx] = pz;
  ptrs.E[idx] = std::sqrt(1.0f + px*px + py*py + pz*pz);
  ptrs.weight[idx] = weight;
  ptrs.cell[idx] = prism_cell_encode(tri_idx, layer_idx, m_mesh.m_N_r);
  ptrs.flag[idx] = flag;
  ptrs.id[idx] = idx;

  m_particles.set_num(idx + 1);
  return static_cast<int>(idx);
}

void prismatic_ptc_updater::remove_dead_particles() {
  // Simple compaction: swap dead particles with the last live one
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
