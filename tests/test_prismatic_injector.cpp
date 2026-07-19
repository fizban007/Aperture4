// Tests for prismatic_ptc_injector (systems/prismatic/prismatic_ptc_injector.hpp)
// — the port of the base functor-driven ptc_injector onto the prismatic
// mesh.  Uses the direct-dependency constructor (no sim_environment).

#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_mesh_ptrs.h"
#include "systems/prismatic/prismatic_ptc_injector.hpp"
#include "systems/prismatic/prismatic_ptc_mesh_local.h"

#include "catch2/catch_all.hpp"
#include <cmath>

using namespace Aperture;

namespace {

struct injector_fixture {
  prismatic_mesh mesh;
  prismatic_particle_data ptc;
  rng_states_t<prismatic_exec_policy_host::exec_tag> states;
  icosphere_topology topo;
  prismatic_mesh_partition mpart;
  prismatic_ptc_mesh_local lmesh;   // identity (single-rank) local mesh

  injector_fixture(int L = 2, int N_r = 4)
      : mesh(),
        ptc(200000, MemType::host_only),
        states(42) {
    mesh.build(L, N_r, 1.0, 2.0);
    states.init();
    topo = icosphere_topology::build_from_mesh(mesh);
    auto part = prismatic_partition::single_rank(L, N_r);
    part.set_topology(&topo);
    mpart = prismatic_mesh_partition::build(part, topo);
    lmesh.build(mesh, mpart);
  }
};

}  // namespace

TEST_CASE("Prismatic injector: uniform volume pair injection",
          "[prismatic][injector]") {
  injector_fixture fx;
  prismatic_ptc_injector<prismatic_exec_policy_host> injector(fx.lmesh, fx.ptc,
                                                              fx.states);

  const int ppc = 6;  // per cell, must be even (pairs)
  injector.inject_pairs(
      [] LAMBDA(int tri, int k, auto& mp) { return true; },
      [ppc] LAMBDA(int tri, int k, auto& mp) { return ppc; },
      [] LAMBDA(auto& x_global, rand_state& state, PtcType type) {
        return vec_t<Scalar, 3>(0.0, 0.0, 0.0);
      },
      [] LAMBDA(auto& x_global, PtcType type) { return Scalar(1.0); });

  int N_cells = fx.mesh.m_N_tri * fx.mesh.m_N_r;
  REQUIRE(fx.ptc.number() == size_t(N_cells * ppc));

  auto ptrs = fx.ptc.get_host_ptrs();
  auto mp = fx.mesh.host_ptrs();
  int n_electron = 0, n_positron = 0;
  for (size_t i = 0; i < fx.ptc.number(); i++) {
    REQUIRE(ptrs.cell[i] != empty_cell);
    int tri, k;
    prism_cell_decode(ptrs.cell[i], fx.mesh.m_N_tri, tri, k);
    REQUIRE(tri < fx.mesh.m_N_tri);
    REQUIRE(k < fx.mesh.m_N_r);
    // Barycentric sample inside the simplex, zeta inside the layer
    REQUIRE(ptrs.x1[i] >= 0.0f);
    REQUIRE(ptrs.x2[i] >= 0.0f);
    REQUIRE(ptrs.x1[i] + ptrs.x2[i] <= 1.0f);
    REQUIRE(ptrs.x3[i] >= 0.0f);
    REQUIRE(ptrs.x3[i] <= 1.0f);
    REQUIRE(ptrs.weight[i] == Scalar(1.0));
    REQUIRE(ptrs.E[i] == Scalar(1.0));  // cold injection
    auto type = get_ptc_type(ptrs.flag[i]);
    if (type == (int)PtcType::electron) n_electron++;
    if (type == (int)PtcType::positron) n_positron++;
  }
  // Exact pairwise charge balance
  REQUIRE(n_electron == N_cells * ppc / 2);
  REQUIRE(n_positron == N_cells * ppc / 2);
}

TEST_CASE("Prismatic injector: criteria restricts to surface shell",
          "[prismatic][injector]") {
  injector_fixture fx;
  prismatic_ptc_injector<prismatic_exec_policy_host> injector(fx.lmesh, fx.ptc,
                                                              fx.states);

  const int ppc = 2;
  injector.inject_pairs(
      [] LAMBDA(int tri, int k, auto& mp) { return k == 0; },
      [ppc] LAMBDA(int tri, int k, auto& mp) { return ppc; },
      [] LAMBDA(auto& x_global, rand_state& state, PtcType type) {
        // Radial kick using the sampled position
        Scalar r = math::sqrt(x_global.dot(x_global));
        return vec_t<Scalar, 3>(x_global[0] / r, x_global[1] / r,
                                x_global[2] / r);
      },
      [] LAMBDA(auto& x_global, PtcType type) { return Scalar(2.0); });

  REQUIRE(fx.ptc.number() == size_t(fx.mesh.m_N_tri * ppc));

  auto ptrs = fx.ptc.get_host_ptrs();
  for (size_t i = 0; i < fx.ptc.number(); i++) {
    int tri, k;
    prism_cell_decode(ptrs.cell[i], fx.mesh.m_N_tri, tri, k);
    REQUIRE(k == 0);
    REQUIRE(ptrs.weight[i] == Scalar(2.0));
    // Momentum is the radial unit vector -> E = sqrt(2)
    REQUIRE(ptrs.E[i] == Catch::Approx(std::sqrt(2.0)).epsilon(1e-5));
  }

  // Second injection appends after the first
  injector.inject_pairs(
      [] LAMBDA(int tri, int k, auto& mp) { return k == 1; },
      [ppc] LAMBDA(int tri, int k, auto& mp) { return ppc; },
      [] LAMBDA(auto& x_global, rand_state& state, PtcType type) {
        return vec_t<Scalar, 3>(0.0, 0.0, 0.0);
      },
      [] LAMBDA(auto& x_global, PtcType type) { return Scalar(1.0); });
  REQUIRE(fx.ptc.number() == size_t(2 * fx.mesh.m_N_tri * ppc));
}

TEST_CASE("Prismatic injector: validator marks rejected slots inert",
          "[prismatic][injector]") {
  injector_fixture fx;
  prismatic_ptc_injector<prismatic_exec_policy_host> injector(fx.lmesh, fx.ptc,
                                                              fx.states);

  const int ppc = 4;
  injector.inject_pairs(
      [] LAMBDA(int tri, int k, auto& mp) { return true; },
      [ppc] LAMBDA(int tri, int k, auto& mp) { return ppc; },
      [] LAMBDA(auto& x_global, rand_state& state, PtcType type) {
        return vec_t<Scalar, 3>(0.0, 0.0, 0.0);
      },
      [] LAMBDA(auto& x_global, PtcType type) { return Scalar(1.0); },
      0,
      // Reject placements in the upper hemisphere
      [] LAMBDA(int cell, auto& x_global) { return x_global[2] <= 0.0; });

  int N_cells = fx.mesh.m_N_tri * fx.mesh.m_N_r;
  // Slots are reserved for every candidate; rejected ones are inert.
  REQUIRE(fx.ptc.number() == size_t(N_cells * ppc));

  auto ptrs = fx.ptc.get_host_ptrs();
  size_t live = 0;
  for (size_t i = 0; i < fx.ptc.number(); i++) {
    if (ptrs.cell[i] == empty_cell) continue;
    live++;
    int tri, k;
    prism_cell_decode(ptrs.cell[i], fx.mesh.m_N_tri, tri, k);
    auto mp = fx.mesh.host_ptrs();
    auto x = prism_position(mp, tri, k, ptrs.x1[i], ptrs.x2[i], ptrs.x3[i]);
    REQUIRE(x[2] <= 0.0);
  }
  // Roughly half the sphere accepted (both pair members share a position)
  REQUIRE(live > size_t(N_cells * ppc) / 4);
  REQUIRE(live < size_t(3 * N_cells * ppc) / 4);
}
