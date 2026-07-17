#pragma once

// Volume-fill injection shared by main.cpp and streaming_test.cpp
// (previously copy-pasted in both).  Reads the same config keys and
// preserves the original behavior: electron-only, radially outward
// momentum ptc_pr, ptc_per_cell particles uniformly placed in every
// prism, fixed seed.  Used by the streaming/GCA test problems — the
// magnetosphere driver uses prismatic_surface_injector instead.

#include "framework/environment.h"
#include "systems/prismatic/prismatic_deposit.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_ptc_updater.h"
#include "utils/logger.h"
#include <random>

namespace Aperture {

template <typename UpdaterPtr>
inline void fill_volume_radial_beam(const prismatic_mesh& mesh,
                                    UpdaterPtr updater) {
  int ptc_per_cell = sim_env().params().get_as<int64_t>("ptc_per_cell", 10);
  double ptc_pr = sim_env().params().get_as<double>("ptc_pr", 10.0);
  bool use_gca = sim_env().params().get_as<bool>("use_gca", false);
  auto mp = mesh.host_ptrs();
  int N_r = mesh.m_N_r;

  // E/B are registered by the solver/updater; fetched once (the old
  // copy re-registered them inside the loop).
  nonown_ptr<prismatic_edge_field> E_data;
  nonown_ptr<prismatic_face_field> B_data;
  sim_env().get_data("E", E_data);
  sim_env().get_data("B", B_data);

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  int count = 0;
  for (int k = 0; k < N_r; k++) {
    for (int t = 0; t < mesh.m_N_tri; t++) {
      for (int ip = 0; ip < ptc_per_cell; ip++) {
        // Uniform barycentric (triangle fold) + zeta within layer
        double u = dist(rng), v = dist(rng);
        if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }
        Scalar l1 = u, l2 = v, l3 = 1.0 - u - v;
        Scalar zeta = dist(rng);

        int v0 = mp.tri_verts[t*3], v1 = mp.tri_verts[t*3+1],
            v2 = mp.tri_verts[t*3+2];
        Scalar sx = l1*mp.sphere_vx[v0] + l2*mp.sphere_vx[v1] +
                    l3*mp.sphere_vx[v2];
        Scalar sy = l1*mp.sphere_vy[v0] + l2*mp.sphere_vy[v1] +
                    l3*mp.sphere_vy[v2];
        Scalar sz = l1*mp.sphere_vz[v0] + l2*mp.sphere_vz[v1] +
                    l3*mp.sphere_vz[v2];
        Scalar sn = 1.0f / std::sqrt(sx*sx + sy*sy + sz*sz);
        sx *= sn; sy *= sn; sz *= sn;

        size_t idx = updater->particles()->number();
        if (idx >= updater->particles()->size()) break;
        auto ptrs = updater->particles()->get_host_ptrs();
        ptrs.x1[idx] = l1; ptrs.x2[idx] = l2; ptrs.x3[idx] = zeta;

        if (use_gca) {
          // Decompose the radial momentum into (u_par, mu) using the
          // local B from the already-initialized field data.
          Scalar ll[3] = {l1, l2, l3};
          Scalar iEx, iEy, iEz, iBx, iBy, iBz;
          interpolate_fields(mp, t, k, ll, zeta,
                             E_data->host_ptr(), B_data->host_ptr(),
                             iEx, iEy, iEz, iBx, iBy, iBz);
          Scalar Bmag = std::sqrt(iBx*iBx + iBy*iBy + iBz*iBz);
          Scalar bx_l = iBx/Bmag, by_l = iBy/Bmag, bz_l = iBz/Bmag;

          Scalar px = ptc_pr * sx, py = ptc_pr * sy, pz = ptc_pr * sz;
          Scalar u_par = px*bx_l + py*by_l + pz*bz_l;
          Scalar u_perp_sq = ptc_pr*ptc_pr - u_par*u_par;
          if (u_perp_sq < 0) u_perp_sq = 0;
          // For E << B: kappa ~ 1, so mu ~ m u_perp^2 / (2B)
          Scalar mu = Scalar(1.0) * u_perp_sq / (Scalar(2) * Bmag);

          ptrs.p1[idx] = u_par;
          ptrs.p2[idx] = mu;
          ptrs.p3[idx] = std::sqrt(u_perp_sq);  // u_perp for reconstruction
          ptrs.E[idx] = std::sqrt(1.0f + ptc_pr*ptc_pr);
        } else {
          Scalar px = ptc_pr * sx, py = ptc_pr * sy, pz = ptc_pr * sz;
          ptrs.p1[idx] = px; ptrs.p2[idx] = py; ptrs.p3[idx] = pz;
          ptrs.E[idx] = std::sqrt(1.0f + ptc_pr*ptc_pr);
        }

        ptrs.weight[idx] = 1.0;
        ptrs.cell[idx] = prism_cell_encode(t, k, mesh.m_N_tri);
        ptrs.flag[idx] = gen_ptc_type_flag(PtcType::electron);
        ptrs.id[idx] = idx;
        updater->particles()->set_num(idx + 1);
        count++;
      }
    }
  }
  updater->particles()->copy_to_device();
  Logger::print_info("Volume-filled {} particles ({} per cell, {} prisms)",
                     count, ptc_per_cell, mesh.m_N_tri * N_r);
}

}  // namespace Aperture
