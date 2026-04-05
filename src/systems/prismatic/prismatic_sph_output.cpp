#include "systems/prismatic/prismatic_sph_output.h"
#include "systems/prismatic/prismatic_deposit.h"
#include "framework/environment.h"
#include "utils/hdf_wrapper.h"
#include "utils/logger.h"
#include <cmath>
#include <cstdio>
#include <filesystem>

namespace Aperture {

prismatic_sph_output::prismatic_sph_output(prismatic_mesh& mesh)
    : m_mesh(mesh) {}

void prismatic_sph_output::register_data_components() {
  m_E = sim_env().register_data<prismatic_edge_field>("E", m_mesh);
  m_B = sim_env().register_data<prismatic_face_field>("B", m_mesh);
  sim_env().get_data_optional("J", m_J);
  sim_env().get_data_optional("rho", m_rho);
}

void prismatic_sph_output::init() {
  sim_env().params().get_value("sph_N_theta", m_N_theta);
  sim_env().params().get_value("sph_N_phi", m_N_phi);
  sim_env().params().get_value("fld_output_interval", m_output_interval);
  sim_env().params().get_value("output_dir", m_output_dir);

  std::filesystem::create_directories(m_output_dir);

  int N_ang = m_N_theta * m_N_phi;
  int N_total = N_ang * (m_mesh.m_N_r + 1);
  m_Br.resize(N_total); m_Bth.resize(N_total); m_Bph.resize(N_total);
  m_Er.resize(N_total); m_Eth.resize(N_total); m_Eph.resize(N_total);
  if (m_J != nullptr) { m_Jr.resize(N_total); m_Jth.resize(N_total); m_Jph.resize(N_total); }
  if (m_rho != nullptr) { m_rho_grid.resize(N_total); }

  precompute_grid();
  write_grid_info();

  m_time = 0.0;
  Logger::print_info("Spherical output initialized: {}x{}x{} = {} points",
                     m_N_theta, m_N_phi, m_mesh.m_N_r + 1, N_total);
}

void prismatic_sph_output::precompute_grid() {
  int N_ang = m_N_theta * m_N_phi;
  m_grid.resize(N_ang);
  auto mp = m_mesh.host_ptrs();

  int tri_hint = 0;
  for (int it = 0; it < m_N_theta; it++) {
    // theta in [0, pi], including poles
    Scalar theta = Scalar(M_PI) * it / (m_N_theta - 1);
    Scalar sin_th = std::sin(theta);
    Scalar cos_th = std::cos(theta);

    for (int ip = 0; ip < m_N_phi; ip++) {
      Scalar phi = Scalar(2.0 * M_PI) * ip / m_N_phi;
      Scalar sx = sin_th * std::cos(phi);
      Scalar sy = sin_th * std::sin(phi);
      Scalar sz = cos_th;

      int tri = mp.find_triangle(sx, sy, sz, tri_hint);
      tri_hint = tri;

      auto& pt = m_grid[it * m_N_phi + ip];
      pt.tri_idx = tri;
      mp.compute_barycentric(tri, sx, sy, sz, pt.l[0], pt.l[1], pt.l[2]);
      pt.sx = sx;
      pt.sy = sy;
      pt.sz = sz;
    }
  }

  Logger::print_info("  Precomputed {} angular grid points", N_ang);
}

void prismatic_sph_output::write_grid_info() {
  std::string filename = m_output_dir + "/sph_grid.h5";
  auto file = hdf_create(filename);

  file.write(m_N_theta, "N_theta");
  file.write(m_N_phi, "N_phi");
  file.write(m_mesh.m_N_r, "N_r");

  // Write theta and phi arrays
  std::vector<Scalar> theta(m_N_theta), phi(m_N_phi);
  for (int i = 0; i < m_N_theta; i++)
    theta[i] = Scalar(M_PI) * i / (m_N_theta - 1);
  for (int i = 0; i < m_N_phi; i++)
    phi[i] = Scalar(2.0 * M_PI) * i / m_N_phi;
  file.write(theta.data(), m_N_theta, "theta");
  file.write(phi.data(), m_N_phi, "phi");
  file.write(m_mesh.radii.host_ptr(), m_mesh.m_N_r + 1, "radii");

  file.close();
  Logger::print_info("  Grid info written to {}", filename);
}

void prismatic_sph_output::update(double dt, uint32_t step) {
  m_time += dt;
  if (step % m_output_interval != 0) return;

  // Sync E and B from device to host (no-op for host-only buffers).
  // J and rho are deposited on the host by the particle updater,
  // so they must NOT be overwritten from device.
  m_E->data().copy_to_host();
  m_B->data().copy_to_host();

  write_snapshot(step, m_time);
}

void prismatic_sph_output::write_snapshot(uint32_t step, double time) {
  auto mp = m_mesh.host_ptrs();
  const Scalar* E_e = m_E->host_ptr();
  const Scalar* B_f = m_B->host_ptr();
  int N_ang = m_N_theta * m_N_phi;
  int N_shells = m_mesh.m_N_r + 1;

  // For each shell k and angular point, interpolate the fields.
  // On a shell boundary (zeta = 0 of layer k, or equivalently zeta = 1
  // of layer k-1), we use the layer starting at that shell.  For the
  // outermost shell (k = N_r), use layer N_r - 1 with zeta = 1.
  for (int k = 0; k < N_shells; k++) {
    int layer = (k < m_mesh.m_N_r) ? k : m_mesh.m_N_r - 1;
    Scalar zeta = (k < m_mesh.m_N_r) ? Scalar(0) : Scalar(1);

    for (int ia = 0; ia < N_ang; ia++) {
      auto& pt = m_grid[ia];
      int idx = k * N_ang + ia;

      Scalar iEx, iEy, iEz, iBx, iBy, iBz;
      interpolate_fields(mp, pt.tri_idx, layer, pt.l, zeta,
                         E_e, B_f, iEx, iEy, iEz, iBx, iBy, iBz);

      // Convert Cartesian (Bx, By, Bz) to spherical (Br, Bth, Bph)
      Scalar sx = pt.sx, sy = pt.sy, sz = pt.sz;
      // r_hat = (sx, sy, sz)  (unit sphere position IS the radial unit vector)
      // theta_hat = (cos_th*cos_phi, cos_th*sin_phi, -sin_th)
      // phi_hat = (-sin_phi, cos_phi, 0)
      Scalar cos_th = sz;
      Scalar sin_th = std::sqrt(sx * sx + sy * sy);
      Scalar cos_phi, sin_phi;
      if (sin_th > Scalar(1e-10)) {
        cos_phi = sx / sin_th;
        sin_phi = sy / sin_th;
      } else {
        cos_phi = Scalar(1);
        sin_phi = Scalar(0);
      }

      m_Br[idx]  = iBx * sx + iBy * sy + iBz * sz;
      m_Bth[idx] = iBx * cos_th * cos_phi + iBy * cos_th * sin_phi - iBz * sin_th;
      m_Bph[idx] = -iBx * sin_phi + iBy * cos_phi;

      m_Er[idx]  = iEx * sx + iEy * sy + iEz * sz;
      m_Eth[idx] = iEx * cos_th * cos_phi + iEy * cos_th * sin_phi - iEz * sin_th;
      m_Eph[idx] = -iEx * sin_phi + iEy * cos_phi;

      // J: same Whitney 1-form interpolation as E (both are edge 1-cochains)
      if (m_J != nullptr) {
        Scalar iJx, iJy, iJz, dummy1, dummy2, dummy3;
        interpolate_fields(mp, pt.tri_idx, layer, pt.l, zeta,
                           m_J->host_ptr(), B_f, iJx, iJy, iJz,
                           dummy1, dummy2, dummy3);
        m_Jr[idx]  = iJx * sx + iJy * sy + iJz * sz;
        m_Jth[idx] = iJx * cos_th * cos_phi + iJy * cos_th * sin_phi - iJz * sin_th;
        m_Jph[idx] = -iJx * sin_phi + iJy * cos_phi;
      }

      // rho: Whitney 0-form interpolation (barycentric on vertices)
      if (m_rho != nullptr) {
        const Scalar* rho_data = m_rho->host_ptr();
        Scalar rho_val = Scalar(0);
        Scalar phi_hat[2] = {Scalar(1) - zeta, zeta};
        int shells[2] = {layer, layer + 1};
        if (k == m_mesh.m_N_r) { shells[0] = m_mesh.m_N_r - 1; shells[1] = m_mesh.m_N_r; }
        for (int lev = 0; lev < 2; lev++) {
          for (int vi = 0; vi < 3; vi++) {
            int sv = mp.tri_verts[pt.tri_idx * 3 + vi];
            int v_idx = shells[lev] * mp.N_vert_s + sv;
            rho_val += rho_data[v_idx] * pt.l[vi] * phi_hat[lev];
          }
        }
        m_rho_grid[idx] = rho_val;
      }
    }
  }

  // Write to HDF5
  char fname[256];
  std::snprintf(fname, sizeof(fname), "%s/sph_%06u.h5",
                m_output_dir.c_str(), step);
  auto file = hdf_create(std::string(fname));

  int N_total = N_shells * N_ang;
  file.write(m_Br.data(), N_total, "Br");
  file.write(m_Bth.data(), N_total, "Bth");
  file.write(m_Bph.data(), N_total, "Bph");
  file.write(m_Er.data(), N_total, "Er");
  file.write(m_Eth.data(), N_total, "Eth");
  file.write(m_Eph.data(), N_total, "Eph");
  if (m_J != nullptr) {
    file.write(m_Jr.data(), N_total, "Jr");
    file.write(m_Jth.data(), N_total, "Jth");
    file.write(m_Jph.data(), N_total, "Jph");
  }
  if (m_rho != nullptr) {
    file.write(m_rho_grid.data(), N_total, "rho");
  }
  file.write(static_cast<int>(step), "step");
  file.write(time, "time");
  file.close();

  Logger::print_info("Spherical snapshot: step={}, time={:.4f}", step, time);
}

}  // namespace Aperture
