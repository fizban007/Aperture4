#include "framework/environment.h"
#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/prismatic_data_exporter.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_ptc_updater.h"
#include "systems/prismatic/prismatic_sph_output.h"
#include "systems/prismatic/prismatic_deposit.h"
#include "utils/util_functions.h"
#include <random>

using namespace Aperture;

int main(int argc, char* argv[]) {
  auto& env = sim_environment::instance(&argc, &argv);

  int L = env.params().get_as<int64_t>("subdivision_level", 3);
  int N_r = env.params().get_as<int64_t>("N_r", 50);
  double r_min = env.params().get_as<double>("r_min", 1.0);
  double r_max = env.params().get_as<double>("r_max", 20.0);

  prismatic_mesh mesh;
  mesh.build(L, N_r, r_min, r_max);
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  mesh.copy_to_device();
#endif

  env.register_system<dec_field_solver_t>(mesh);
  env.register_system<prismatic_data_exporter>(mesh);
  auto updater = env.register_system<prismatic_ptc_updater_t>(mesh);
  env.register_system<prismatic_sph_output>(mesh);

  env.init();

  // ================================================================
  // Volume-fill injection: place particles in every prism
  // ================================================================
  bool fill_volume = env.params().get_as<bool>("fill_volume", false);
  if (fill_volume) {
    int ptc_per_cell = env.params().get_as<int64_t>("ptc_per_cell", 10);
    double ptc_pr = env.params().get_as<double>("ptc_pr", 10.0);
    auto mp = mesh.host_ptrs();

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    int count = 0;
    for (int k = 0; k < N_r; k++) {
      for (int t = 0; t < mesh.m_N_tri; t++) {
        for (int ip = 0; ip < ptc_per_cell; ip++) {
          // Random barycentric coordinates
          double u = dist(rng), v = dist(rng);
          if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }
          Scalar l1 = u, l2 = v, l3 = 1.0 - u - v;

          // Random zeta within layer
          Scalar zeta = dist(rng);

          // Compute 3D position
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

          Scalar r = mp.radii[k] + zeta * (mp.radii[k+1] - mp.radii[k]);
          Scalar x = r * sx, y = r * sy, z = r * sz;

          // Write directly to particle arrays
          size_t idx = updater->particles()->number();
          if (idx >= updater->particles()->size()) break;
          auto ptrs = updater->particles()->get_host_ptrs();
          ptrs.x1[idx] = l1; ptrs.x2[idx] = l2; ptrs.x3[idx] = zeta;

          bool use_gca = env.params().get_as<bool>("use_gca", false);
          if (use_gca) {
            // Compute local B at this position to decompose momentum
            Scalar ll[3] = {l1, l2, l3};
            Scalar iEx, iEy, iEz, iBx, iBy, iBz;
            // Use the field solver's B field (already initialized)
            auto E_data = env.register_data<prismatic_edge_field>("E", mesh);
            auto B_data = env.register_data<prismatic_face_field>("B", mesh);
            interpolate_fields(mp, t, k, ll, zeta,
                               E_data->host_ptr(), B_data->host_ptr(),
                               iEx, iEy, iEz, iBx, iBy, iBz);
            Scalar Bmag = std::sqrt(iBx*iBx + iBy*iBy + iBz*iBz);
            Scalar bx_l = iBx/Bmag, by_l = iBy/Bmag, bz_l = iBz/Bmag;

            // Radial momentum: p = ptc_pr * r_hat
            Scalar px = ptc_pr * sx, py = ptc_pr * sy, pz = ptc_pr * sz;

            // u_par = p · b
            Scalar u_par = px*bx_l + py*by_l + pz*bz_l;
            // u_perp² = |p|² - u_par²
            Scalar u_perp_sq = ptc_pr*ptc_pr - u_par*u_par;
            if (u_perp_sq < 0) u_perp_sq = 0;
            // mu = m * u_perp² / (2 * B * kappa)
            // For E<<B: kappa ≈ 1, so mu ≈ m * u_perp² / (2B)
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

  Logger::print_info("Before run: {} particles", updater->particles()->number());

  env.run();
  return 0;
}
