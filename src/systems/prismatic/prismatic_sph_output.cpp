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

  // Match the metric used by compute_metric() on the mesh and by the GR
  // solver.  When the mesh is flat, √γ is r² sinθ; when it is Kerr-Schild,
  // √γ = sinθ √(Σ(Σ+2r)).  Both are evaluated on the fly in write_snapshot.
  sim_env().params().get_value("bh_spin", m_spin);
  sim_env().params().get_value("use_flat_metric", m_use_flat_metric);

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

  // Sync all fields from device to host (no-op for host-only buffers)
  m_E->data().copy_to_host();
  m_B->data().copy_to_host();
  if (m_J != nullptr) m_J->data().copy_to_host();
  if (m_rho != nullptr) m_rho->data().copy_to_host();

  write_snapshot(step, m_time);
}

void prismatic_sph_output::write_snapshot(uint32_t step, double time) {
  auto mp = m_mesh.host_ptrs();
  const Scalar* B_f = m_B->host_ptr();
  int N_ang = m_N_theta * m_N_phi;
  int N_shells = m_mesh.m_N_r + 1;

  // Whitney interpolation expects primal 1-cochains (edge circulations).
  // The GR solver stores D̃[e] = dual 2-cochain; convert to the primal
  // 1-cochain D_primal[e] = hodge1_inv[e] · D̃[e] before interpolating.
  // Flat solver already stores primal values, so this is a no-op copy.
  const int N_edges = m_mesh.m_N_edges;
  std::vector<Scalar> E_primal(N_edges);
  {
    const Scalar* E_raw = m_E->host_ptr();
    if (m_E->edge_kind() == EdgeCochainKind::dual_2) {
      const Scalar* h1inv = m_mesh.hodge1_inv.host_ptr();
      for (int e = 0; e < N_edges; e++) E_primal[e] = h1inv[e] * E_raw[e];
    } else {
      for (int e = 0; e < N_edges; e++) E_primal[e] = E_raw[e];
    }
  }
  const Scalar* E_e = E_primal.data();

  // J is stored under the same convention as E (primal for flat, dual for GR).
  std::vector<Scalar> J_primal;
  const Scalar* J_e = nullptr;
  if (m_J != nullptr) {
    const Scalar* J_raw = m_J->host_ptr();
    if (m_J->edge_kind() == EdgeCochainKind::dual_2) {
      J_primal.resize(N_edges);
      const Scalar* h1inv = m_mesh.hodge1_inv.host_ptr();
      for (int e = 0; e < N_edges; e++) J_primal[e] = h1inv[e] * J_raw[e];
      J_e = J_primal.data();
    } else {
      J_e = J_raw;
    }
  }

  // For each shell k and angular point, interpolate the fields.
  // On a shell boundary (zeta = 0 of layer k, or equivalently zeta = 1
  // of layer k-1), we use the layer starting at that shell.  For the
  // outermost shell (k = N_r), use layer N_r - 1 with zeta = 1.
  //
  // Output convention:
  //   Er / Eth / Eph  = D_i  (coord-basis COVARIANT, KS or flat) at (r_k, θ)
  //   Br / Bth / Bph  = B^i  (coord-basis CONTRAVARIANT) at (r_k, θ)
  //   Jr / Jth / Jph  = J_i  (coord-basis covariant, same form as D_i)
  //
  // Rationale: the prismatic mesh is embedded as a flat Euclidean
  // icosphere, so the Whitney reconstruction natively produces a flat
  // 1-form / 2-form.  Extracting Cartesian r̂/θ̂/φ̂ projections gives:
  //   - for the 1-form:    V_orth_r̂ = V_r_cov,  V_orth_θ̂ = V_θ_cov / r,
  //                        V_orth_φ̂ = V_φ_cov / (r sinθ)
  //   - for the 2-form:    V^i_flat_sph = (√γ_flat)⁻¹ · ε^{ijk} F_{jk}/2,
  //                        and V^i_flat_sph = B^i_KS · √γ_KS / √γ_flat
  // Multiplying the 1-form projections by (1, r, r sinθ) and the 2-form
  // projections by √γ_flat / √γ = (r² sinθ)/√γ · (appropriate per-axis
  // scale) converts to coord-basis components — independent of the metric
  // for D_i, metric-aware (via √γ) for B^i.  This is the natural form for
  // downstream visualization and for particle-interpolation consumers.
  for (int k = 0; k < N_shells; k++) {
    int layer = (k < m_mesh.m_N_r) ? k : m_mesh.m_N_r - 1;
    Scalar zeta = (k < m_mesh.m_N_r) ? Scalar(0) : Scalar(1);
    Scalar r = m_mesh.radii[k];
    Scalar r_sq = r * r;

    for (int ia = 0; ia < N_ang; ia++) {
      auto& pt = m_grid[ia];
      int idx = k * N_ang + ia;

      Scalar iEx, iEy, iEz, iBx, iBy, iBz;
      interpolate_fields(mp, pt.tri_idx, layer, pt.l, zeta,
                         E_e, B_f, iEx, iEy, iEz, iBx, iBy, iBz);

      // Cartesian unit-vector frame at the grid point.
      Scalar sx = pt.sx, sy = pt.sy, sz = pt.sz;
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

      // Cartesian r̂ / θ̂ / φ̂ projections (flat-orthonormal components).
      Scalar B_or = iBx * sx + iBy * sy + iBz * sz;
      Scalar B_ot = iBx * cos_th * cos_phi + iBy * cos_th * sin_phi -
                    iBz * sin_th;
      Scalar B_op = -iBx * sin_phi + iBy * cos_phi;
      Scalar E_or = iEx * sx + iEy * sy + iEz * sz;
      Scalar E_ot = iEx * cos_th * cos_phi + iEy * cos_th * sin_phi -
                    iEz * sin_th;
      Scalar E_op = -iEx * sin_phi + iEy * cos_phi;

      // √γ at (r_k, θ) for the background metric.
      Scalar sqrt_gamma;
      if (m_use_flat_metric) {
        sqrt_gamma = r_sq * sin_th;
      } else {
        Scalar Sigma = r_sq + m_spin * m_spin * cos_th * cos_th;
        sqrt_gamma = sin_th * std::sqrt(Sigma * (Sigma + Scalar(2) * r));
      }
      Scalar sg_safe = (sqrt_gamma > Scalar(1e-20)) ? sqrt_gamma
                                                    : Scalar(1e-20);
      Scalar inv_sg = Scalar(1) / sg_safe;

      // D_i (coord-basis covariant) — metric-independent:
      //   D_r = E_or,  D_θ = E_ot · r,  D_φ = E_op · r sinθ.
      m_Er[idx]  = E_or;
      m_Eth[idx] = E_ot * r;
      m_Eph[idx] = E_op * r * sin_th;

      // B^i (coord-basis contravariant):
      //   B^i = V^i_flat_sph · √γ_flat / √γ
      //       = (B_ortho_i / flat_scale_i) · (r² sinθ) / √γ
      // which simplifies to:
      //   B^r = B_or · r² sinθ / √γ
      //   B^θ = B_ot · r  sinθ / √γ
      //   B^φ = B_op · r       / √γ.
      m_Br[idx]  = B_or * r_sq * sin_th * inv_sg;
      m_Bth[idx] = B_ot * r    * sin_th * inv_sg;
      m_Bph[idx] = B_op * r             * inv_sg;

      // J: same Whitney 1-form interpolation as E.
      if (m_J != nullptr) {
        Scalar iJx, iJy, iJz, dummy1, dummy2, dummy3;
        interpolate_fields(mp, pt.tri_idx, layer, pt.l, zeta,
                           J_e, B_f, iJx, iJy, iJz,
                           dummy1, dummy2, dummy3);
        Scalar J_or = iJx * sx + iJy * sy + iJz * sz;
        Scalar J_ot = iJx * cos_th * cos_phi + iJy * cos_th * sin_phi -
                      iJz * sin_th;
        Scalar J_op = -iJx * sin_phi + iJy * cos_phi;
        m_Jr[idx]  = J_or;
        m_Jth[idx] = J_ot * r;
        m_Jph[idx] = J_op * r * sin_th;
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
