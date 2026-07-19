// Device-path test for prismatic_ptc_injector: exercises the GPU exec
// policy (thrust exclusive scan, device rng_t, adapter plumbing) that the
// host-side test_prismatic_injector.cpp cannot reach.

#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_ptc_injector.hpp"
#include "systems/prismatic/prismatic_ptc_mesh_local.h"

#include "catch2/catch_all.hpp"

using namespace Aperture;

TEST_CASE("Prismatic injector device path: uniform pair injection",
          "[prismatic][injector][gpu]") {
  prismatic_mesh mesh;
  mesh.build(2, 4, 1.0, 2.0);
  mesh.copy_to_device();

  prismatic_particle_data ptc(200000, MemType::host_device);
  rng_states_t<exec_tags::device> states(42);
  states.init();

  icosphere_topology topo = icosphere_topology::build_from_mesh(mesh);
  auto part = prismatic_partition::single_rank(2, 4);
  part.set_topology(&topo);
  auto mpart = prismatic_mesh_partition::build(part, topo);
  prismatic_ptc_mesh_local lmesh;
  lmesh.build(mesh, mpart, nullptr, MemType::host_device);
  lmesh.copy_to_device();

  prismatic_ptc_injector<prismatic_exec_policy_gpu> injector(lmesh, ptc,
                                                             states);

  const int ppc = 6;
  injector.inject_pairs(
      [] LAMBDA(int tri, int k, auto& mp) { return k < 2; },
      [ppc] LAMBDA(int tri, int k, auto& mp) { return ppc; },
      [] LAMBDA(auto& x_global, rand_state& state, PtcType type) {
        return vec_t<Scalar, 3>(rng_gaussian<Scalar>(state, 0.1), 0.0, 0.0);
      },
      [] LAMBDA(auto& x_global, PtcType type) { return Scalar(1.0); });

  int N_inject_cells = mesh.m_N_tri * 2;
  REQUIRE(ptc.number() == size_t(N_inject_cells * ppc));

  ptc.copy_to_host();
  auto ptrs = ptc.get_host_ptrs();
  int n_electron = 0, n_positron = 0;
  for (size_t i = 0; i < ptc.number(); i++) {
    REQUIRE(ptrs.cell[i] != empty_cell);
    int tri, k;
    prism_cell_decode(ptrs.cell[i], mesh.m_N_tri, tri, k);
    REQUIRE(k < 2);
    REQUIRE(ptrs.x1[i] + ptrs.x2[i] <= 1.0f);
    REQUIRE(ptrs.E[i] >= 1.0f);
    auto type = get_ptc_type(ptrs.flag[i]);
    if (type == (int)PtcType::electron) n_electron++;
    if (type == (int)PtcType::positron) n_positron++;
  }
  REQUIRE(n_electron == N_inject_cells * ppc / 2);
  REQUIRE(n_positron == N_inject_cells * ppc / 2);
}
