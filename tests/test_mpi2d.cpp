#include "core/particles.h"
#include "framework/environment.h"
#include "framework/config.h"
#include "systems/grid.h"
#include "systems/domain_comm.h"
#include "utils/logger.h"

using namespace Aperture;

int
main(int argc, char* argv[]) {
  sim_environment env(&argc, &argv);
  typedef Config<2> Conf;

  env.params().add("log_level", int64_t(LogLevel::debug));
  env.params().add("N", std::vector<int64_t>({10, 10}));
  env.params().add("guard", std::vector<int64_t>({2, 2}));
  env.params().add("nodes", std::vector<int64_t>({2, 2}));
  env.params().add("lower", std::vector<double>({1.0, 2.0}));
  env.params().add("size", std::vector<double>({100.0, 10.0}));
  env.params().add("periodic_boundary", std::vector<bool>({true, true}));

  auto comm = env.register_system<domain_comm<Conf>>(env);
  auto grid = env.register_system<grid_t<Conf>>(env, *comm);
  comm->resize_buffers(*grid);

  particles_t ptc(100, MemType::device_managed);
  photons_t ph(100, MemType::device_managed);
  int N1 = grid->dims[0];
  if (comm->rank() == 0) {
    ptc.append_dev({0.5, 0.5, 0.5}, {1.0, 0.0, 0.0}, 1 + 7 * N1);
    ptc.append_dev({0.5, 0.5, 0.5}, {2.0, 0.0, 0.0}, (N1 - 1) + 3 * N1);
    ptc.append_dev({0.5, 0.5, 0.5}, {3.0, 1.0, 0.0}, 1 + (N1 - 1) * N1);
    ptc.append_dev({0.5, 0.5, 0.5}, {4.0, -1.0, 0.0}, (N1 - 1) + 0 * N1);
    ph.append_dev({0.1, 0.2, 0.3}, {1.0, 1.0, 1.0}, 2 + 8 * N1, 0.0);
  }
  comm->send_particles(ptc, *grid);
  comm->send_particles(ph, *grid);
  ptc.sort_by_cell(grid->size());
  ph.sort_by_cell(grid->size());

  Logger::print_debug_all("Rank {} has {} particles:",
                          comm->rank(), ptc.number());
  Logger::print_debug_all(
      "Rank {} has {} photons:", comm->rank(), ph.number());
  for (unsigned int i = 0; i < ptc.number(); i++) {
    auto c = ptc.cell[i];
    Logger::print_debug_all("cell {}, {}", c % N1, c / N1);
  }

  // multi_array<Scalar> v(mesh.extent());
  // v.assign_dev(env.domain_info().rank);
  // env.send_array_guard_cells(v);
  // v.copy_to_host();

  // for (int n = 0; n < env.domain_info().size; n++) {
  //   MPI_Barrier(MPI_COMM_WORLD);
  //   if (n == env.domain_info().rank) {
  //     std::cout << "This is the initial content from rank " << n << std::endl;
  //     for (int j = 0; j < mesh.dims[1]; j++) {
  //       for (int i = 0; i < mesh.dims[0]; i++) {
  //         std::cout << v(i, j) << " ";
  //       }
  //       std::cout << std::endl;
  //     }
  //   }
  // }

  // env.send_add_array_guard_cells(v);
  // env.send_add_array_guard_cells_single_dir(v, 0, -1);
  // env.send_add_array_guard_cells_single_dir(v, 0, 1);

  // v.copy_to_host();

  // for (int n = 0; n < env.domain_info().size; n++) {
  //   MPI_Barrier(MPI_COMM_WORLD);
  //   if (n == env.domain_info().rank) {
  //     std::cout << "This is the content from rank " << n << std::endl;
  //     for (int j = 0; j < mesh.dims[1]; j++) {
  //       for (int i = 0; i < mesh.dims[0]; i++) {
  //         std::cout << v(i, j) << " ";
  //       }
  //       std::cout << std::endl;
  //     }
  //   }
  // }

  return 0;
}
