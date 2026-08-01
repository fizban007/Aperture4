// Trajectory-level pusher test on a FROZEN static dipole background.
//
// Quantifies the accuracy of the production particle path (gather +
// hybrid pusher kernel) for trapped dipole orbits — the motion that
// dominates the neutron-star application.  The DEC field solver is
// registered only to allocate field data and set the initial analytic
// dipole (E = 0); it is never stepped, so the particles move through a
// static discrete background with no self-consistent feedback.
//
// A deterministic trapped ensemble is seeded near the equatorial
// mid-shell with gyro-radius = orbit_rgyro_frac * (local horizontal
// edge length) and fixed pitch angle.  The branch under test is chosen
// with the standard config keys:
//   use_gca = false                      -> pure Boris
//   use_gca = true, gca_switch_omegac ~ 0 -> pure GCA
//   use_recovery_gather = true/false      -> recovery vs primal B-gather
//
// Output: orbit_%06d.csv per dump (id, cell, flag, Cartesian x/y/z via
// the kernel's own decode, p1..p3, E).  For gca_state particles the
// momentum slots are (u_par, mu, u_perp).  Analysis compares against a
// fine-dt reference integration in the exact dipole field.
//
// Single rank only.

#include "framework/environment.h"
#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/prismatic_data_exporter.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_ptc_updater.h"
#include "systems/prismatic/prismatic_ptc_update_kernel.hpp"
#include <cmath>
#include <cstdio>
#include <mpi.h>
#include <string>

using namespace Aperture;

namespace {

// Exact aligned dipole (m = Bp zhat), matching set_initial_dipole.
void dipole_B(double Bp, double x, double y, double z, double& Bx,
              double& By, double& Bz) {
  double r2 = x * x + y * y + z * z;
  double r = std::sqrt(r2);
  double ir3 = 1.0 / (r2 * r);
  double mdotr = z / r;  // m_hat . r_hat
  Bx = Bp * ir3 * (3.0 * mdotr * x / r);
  By = Bp * ir3 * (3.0 * mdotr * y / r);
  Bz = Bp * ir3 * (3.0 * mdotr * z / r - 1.0);
}

}  // namespace

int main(int argc, char* argv[]) {
  auto& env = sim_environment::instance(&argc, &argv);

  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_size != 1) {
    Logger::print_err("ptc_orbit_test is single-rank only");
    MPI_Finalize();
    return 1;
  }

  int L = env.params().get_as<int64_t>("subdivision_level", 4);
  int N_r = env.params().get_as<int64_t>("N_r", 16);
  double r_min = env.params().get_as<double>("r_min", 1.0);
  double r_max = env.params().get_as<double>("r_max", 2.0);
  double Bp = env.params().get_as<double>("Bp", 1.0);
  double dt = env.params().get_as<double>("dt", 0.05);
  int max_steps = env.params().get_as<int64_t>("max_steps", 2000);
  int dump_interval = env.params().get_as<int64_t>("dump_interval", 20);
  int n_ptc = env.params().get_as<int64_t>("orbit_n_ptc", 64);
  double rg_frac = env.params().get_as<double>("orbit_rgyro_frac", 0.25);
  double pitch_deg = env.params().get_as<double>("orbit_pitch_deg", 60.0);
  std::string out_dir =
      env.params().get_as<std::string>("output_dir", "Data/");
  if (!out_dir.empty() && out_dir.back() != '/') out_dir += '/';

  prismatic_mesh mesh;
  mesh.build(L, N_r, r_min, r_max);
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  mesh.copy_to_device();
#endif

  auto updater =
      env.register_system<prismatic_ptc_updater_t>(mesh, nullptr, nullptr);
  auto solver =
      env.register_system<dec_field_solver_t>(mesh, nullptr, nullptr);
  env.register_system<prismatic_data_exporter>(mesh, nullptr, nullptr);

  env.init();
  solver->set_initial_dipole();  // frozen background: solver never steps

  // ---- Deterministic trapped ensemble ----------------------------------
  // Mid-shell radii, golden-angle azimuths, near-equatorial latitudes,
  // pitch angle pitch_deg w.r.t. the local exact B, gyro-phase from the
  // particle index.  r_gyro = u_perp / |B| = rg_frac * h(r0).
  auto mph = mesh.host_ptrs();
  const double pitch = pitch_deg * M_PI / 180.0;
  for (int i = 0; i < n_ptc; ++i) {
    double fi = (n_ptc > 1) ? double(i) / double(n_ptc - 1) : 0.5;
    double r0 = 1.3 + 0.3 * fi;
    double phi0 = 2.399963229728653 * i;  // golden angle
    double th0 = 0.5 * M_PI + 0.12 * std::sin(2.71 * i + 0.4);

    double x = r0 * std::sin(th0) * std::cos(phi0);
    double y = r0 * std::sin(th0) * std::sin(phi0);
    double z = r0 * std::cos(th0);

    double Bx, By, Bz;
    dipole_B(Bp, x, y, z, Bx, By, Bz);
    double Bn = std::sqrt(Bx * Bx + By * By + Bz * Bz);
    double bx = Bx / Bn, by = By / Bn, bz = Bz / Bn;

    // Median horizontal edge length on the shell containing r0.
    int k0 = 0;
    while (k0 + 1 < mesh.m_N_r && mesh.radii[k0 + 1] < r0) ++k0;
    // crude but deterministic: use edge 0..N_edge_s median-ish proxy =
    // arc of edge 0 at this radius (edges are near-uniform).
    double h_loc = mesh.radii[k0] * mesh.sph_edge_alpha[0];

    double u_perp = rg_frac * h_loc * Bn;
    double u_par = u_perp / std::tan(pitch);

    // perpendicular basis: e1 = b x a (a = any non-parallel), gyro-phase
    double ax = std::sin(1.3 * i), ay = std::cos(0.7 * i + 1.1), az = 0.5;
    double e1x = by * az - bz * ay, e1y = bz * ax - bx * az,
           e1z = bx * ay - by * ax;
    double e1n = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
    e1x /= e1n; e1y /= e1n; e1z /= e1n;
    double e2x = by * e1z - bz * e1y, e2y = bz * e1x - bx * e1z,
           e2z = bx * e1y - by * e1x;
    double ph = 0.9 * i + 0.2;
    double ux = u_perp * (std::cos(ph) * e1x + std::sin(ph) * e2x) +
                u_par * bx;
    double uy = u_perp * (std::cos(ph) * e1y + std::sin(ph) * e2y) +
                u_par * by;
    double uz = u_perp * (std::cos(ph) * e1z + std::sin(ph) * e2z) +
                u_par * bz;

    if (updater->add_particle(Scalar(x), Scalar(y), Scalar(z), Scalar(ux),
                              Scalar(uy), Scalar(uz), Scalar(1.0),
                              0u) < 0) {
      Logger::print_err("seeding particle {} failed (outside mesh?)", i);
    }
  }
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  updater->particles()->copy_to_device();
#endif
  auto ptc = updater->particles();
  Logger::print_info("ptc_orbit_test: seeded {} particles", ptc->number());

  // ---- Manual loop: particles only, fields frozen ----------------------
  auto dump = [&](uint32_t step) {
    size_t num = ptc->number();
    ptc->copy_to_host();
    char fname[512];
    std::snprintf(fname, sizeof(fname), "%sorbit_%06u.csv", out_dir.c_str(),
                  step);
    std::FILE* f = std::fopen(fname, "w");
    if (!f) {
      Logger::print_err("cannot open {}", fname);
      return;
    }
    std::fprintf(f, "# t=%.10g\n", double(step) * dt);
    std::fprintf(f, "id,cell,flag,x,y,z,p1,p2,p3,E\n");
    for (size_t n = 0; n < num; ++n) {
      uint32_t cell = ptc->cell[n];
      if (cell == empty_cell) {
        std::fprintf(f, "%llu,%u,%u,nan,nan,nan,nan,nan,nan,nan\n",
                     (unsigned long long)ptc->id[n], cell, ptc->flag[n]);
        continue;
      }
      int layer = int(cell) / mesh.m_N_tri;
      int tri = int(cell) % mesh.m_N_tri;
      Scalar x, y, z;
      local_to_cartesian_impl(mph, tri, layer, ptc->x1[n], ptc->x2[n],
                              ptc->x3[n], x, y, z);
      std::fprintf(f, "%llu,%u,%u,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                   (unsigned long long)ptc->id[n], cell, ptc->flag[n],
                   double(x), double(y), double(z), double(ptc->p1[n]),
                   double(ptc->p2[n]), double(ptc->p3[n]),
                   double(ptc->E[n]));
    }
    std::fclose(f);
  };

  for (int step = 0; step < max_steps; ++step) {
    if (step % dump_interval == 0) dump(step);
    updater->update(dt, uint32_t(step));
  }
  dump(uint32_t(max_steps));
  Logger::print_info("ptc_orbit_test: done ({} steps, dt = {})", max_steps,
                     dt);
  return 0;
}
