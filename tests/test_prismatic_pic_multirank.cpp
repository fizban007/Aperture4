// Multi-process acceptance test for distributed prismatic PIC (Phase 6).
//
// Deterministic full-loop run: dipole IC, a fixed lattice of seeded
// particles (identical global set at any rank count; each rank keeps
// its owned ones), N steps of push -> Whitney deposit -> J reduction ->
// migration -> field update, then global snapshot + sph output.
//
// Run single-process and under mpirun -n 20/40 with the same config
// (different output_dir) and compare:
//   - exporter step files / sph files: equal to FP-reordering tolerance
//     (the deposit allreduce and atomics reorder float sums);
//   - LIVE particle counts: printed, must match across rank counts;
//   - OWNERSHIP: every live particle sits on the rank owning its cell
//     (fails if migration misroutes or drops particles).
//
// Reference config: tests/config_prismatic_pic_multirank.toml (holds
// the expected baseline numbers).  Use distinct output_dir per rank
// count, then diff the HDF5 datasets.
//
//   ./test_prismatic_pic_multirank -c tests/config_prismatic_pic_multirank.toml
//   mpirun -n 20 ./test_prismatic_pic_multirank -c cfg.toml

#include "framework/environment.h"
#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_data_exporter.h"
#include "systems/prismatic/prismatic_field_replicator.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_mpi_comm.h"
#include "systems/prismatic/prismatic_ptc_updater.h"
#include "systems/prismatic/prismatic_sph_output.h"
#include <cmath>
#include <cstdio>
#include <mpi.h>

using namespace Aperture;

int main(int argc, char* argv[]) {
  auto& env = sim_environment::instance(&argc, &argv);

  int world_size = 1, world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_size > 1 && world_size % 20 != 0) {
    if (world_rank == 0)
      std::fprintf(stderr, "SKIP: needs 1 or 20*K ranks (got %d)\n",
                   world_size);
    MPI_Finalize();
    return 0;
  }

  int L = env.params().get_as<int64_t>("subdivision_level", 2);
  int N_r = env.params().get_as<int64_t>("N_r", 8);
  double r_min = env.params().get_as<double>("r_min", 1.0);
  double r_max = env.params().get_as<double>("r_max", 2.0);
  int seed_stride = env.params().get_as<int64_t>("seed_stride", 4);
  double p0 = env.params().get_as<double>("seed_p0", 1.0);

  prismatic_mesh mesh;
  mesh.build(L, N_r, r_min, r_max);
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  mesh.copy_to_device();
#endif

  prismatic_mpi_comm mcomm;
  icosphere_topology topo;
  prismatic_partition part;
  prismatic_mesh_partition mpart;
  const prismatic_mesh_partition* mp = nullptr;
  const prismatic_mpi_comm* pc = nullptr;
  if (world_size > 1) {
    mcomm = prismatic_mpi_comm::create(MPI_COMM_WORLD, world_size / 20);
    topo = icosphere_topology::build_from_mesh(mesh);
    part = prismatic_partition::combined(mesh.m_L, mesh.m_N_r,
                                         mcomm.n_radial_ranks(),
                                         mcomm.radial_rank(),
                                         mcomm.angular_rank());
    part.set_topology(&topo);
    mpart = prismatic_mesh_partition::build(part, topo);
    mp = &mpart;
    pc = &mcomm;
    env.register_system<prismatic_field_replicator_t>(mesh, mp, pc);
  }
  auto updater = env.register_system<prismatic_ptc_updater_t>(mesh, mp, pc);
  auto solver = env.register_system<dec_field_solver_t>(mesh, pc);
  env.register_system<prismatic_data_exporter>(mesh, mp, pc);
  env.register_system<prismatic_sph_output>(mesh, mp, pc);

  env.init();
  solver->set_initial_dipole();

  // Deterministic particle lattice: every seed_stride-th sphere tri on
  // every layer, at the prism center, alternating e-/e+, tangential
  // momentum of magnitude p0 with a deterministic per-cell variation.
  auto mph = mesh.host_ptrs();
  long n_seeded_local = 0;
  for (int k = 0; k < mesh.m_N_r; ++k) {
    double r_c = 0.5 * (mesh.radii[k] + mesh.radii[k + 1]);
    for (int t = 0; t < mesh.m_N_tri; t += seed_stride) {
      double cx = 0, cy = 0, cz = 0;
      for (int vi = 0; vi < 3; ++vi) {
        int sv = mph.tri_verts[t * 3 + vi];
        cx += mph.sphere_vx[sv];
        cy += mph.sphere_vy[sv];
        cz += mph.sphere_vz[sv];
      }
      double norm = std::sqrt(cx * cx + cy * cy + cz * cz);
      cx /= norm; cy /= norm; cz /= norm;
      // Tangential direction ~ zhat x rhat, with a fallback near poles.
      double tx = -cy, ty = cx, tz = 0;
      double tn = std::sqrt(tx * tx + ty * ty);
      if (tn < 1e-6) { tx = 1; ty = 0; tn = 1; }
      tx /= tn; ty /= tn;
      double phase = 0.37 * t + 0.61 * k;
      double px = p0 * (tx + 0.1 * std::sin(phase));
      double py = p0 * (ty + 0.1 * std::cos(phase));
      double pz = p0 * 0.1 * std::sin(0.53 * phase);
      uint32_t flag =
          ((t / seed_stride + k) % 2 == 0)
              ? 0u
              : set_ptc_type_flag(0u, PtcType::positron);
      if (updater->add_particle(Scalar(cx * r_c), Scalar(cy * r_c),
                                Scalar(cz * r_c), Scalar(px), Scalar(py),
                                Scalar(pz), Scalar(1.0), flag) >= 0) {
        n_seeded_local++;
      }
    }
  }
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  updater->particles()->copy_to_device();
#endif
  long n_seeded = n_seeded_local;
  MPI_Allreduce(MPI_IN_PLACE, &n_seeded, 1, MPI_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  if (world_rank == 0)
    std::printf("SEEDED %ld particles (%d ranks)\n", n_seeded, world_size);

  env.run();

  // Post-run: live count, ownership audit, energy sum.
  auto ptc = updater->particles();
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  ptc->copy_to_host();
#endif
  auto hp = ptc->get_host_ptrs();
  const int N_tri = mesh.m_N_tri;
  const int tris_per_face = N_tri / 20;
  const int K = world_size > 1 ? world_size / 20 : 1;
  const int base = mesh.m_N_r / K;
  const int rem = mesh.m_N_r - base * K;
  const int me = world_size > 1
                     ? mcomm.radial_rank() * 20 + mcomm.angular_rank()
                     : 0;
  long live = 0, misowned = 0;
  double esum = 0;
  for (size_t n = 0; n < ptc->number(); ++n) {
    if (hp.cell[n] == empty_cell) continue;
    live++;
    esum += double(hp.E[n]) * double(hp.weight[n]);
    if (world_size > 1) {
      if (prism_migrate_dest(hp.cell[n], N_tri, tris_per_face, base, rem,
                             me) >= 0)
        misowned++;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &live, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &misowned, 1, MPI_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &esum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  bool ok = misowned == 0;
  if (world_rank == 0) {
    std::printf("LIVE %ld\nMISOWNED %ld\nESUM %.9e\n", live, misowned,
                esum);
    std::printf("%s\n", ok ? "ALL PASS" : "FAILURE");
  }

  if (world_size > 1) MPI_Finalize();
  return ok ? 0 : 1;
}
