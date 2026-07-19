#include "catch2/catch_all.hpp"
#include "systems/prismatic/dec_solver_dist.h"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_partition.h"
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

using namespace Aperture;

namespace {

using policy = prismatic_exec_policy_host;
using core_t = dec_solver_dist<policy>;

constexpr int TL = 2;      // subdivision level
constexpr int TN_r = 8;    // radial layers
constexpr double TDT = 0.01;

// -------------------------------------------------------------------------
// Global reference stepper: a literal transcription of dec_field_solver's
// update_explicit kernels (Faraday, Ampere, damping, PEC) over the global
// flat CSR.  The dist core must reproduce this on owned cells.
// -------------------------------------------------------------------------
struct global_ref {
  const prismatic_mesh* mesh = nullptr;
  std::vector<Scalar> E, B, J;

  void init(const prismatic_mesh& m) {
    mesh = &m;
    E.resize(m.m_N_edges);
    B.resize(m.m_N_faces);
    J.resize(m.m_N_edges);
    for (int e = 0; e < m.m_N_edges; e++) {
      E[e] = std::sin(Scalar(0.013) * e) + Scalar(0.37);
      J[e] = Scalar(0.05) * std::sin(Scalar(0.011) * e + Scalar(0.3));
    }
    for (int f = 0; f < m.m_N_faces; f++)
      B[f] = std::cos(Scalar(0.007) * f) - Scalar(0.21);
  }

  void faraday(double dt) {
    auto const& m = *mesh;
    const int* rp = m.d1_row_ptr.host_ptr();
    const int* ci = m.d1_col_idx.host_ptr();
    const Scalar* v = m.d1_val.host_ptr();
    for (int f = 0; f < m.m_N_faces; f++) {
      Scalar curl_E = Scalar(0);
      for (int j = rp[f]; j < rp[f + 1]; j++) curl_E += v[j] * E[ci[j]];
      B[f] -= dt * curl_E;
    }
  }

  void ampere(double dt) {
    auto const& m = *mesh;
    const int* rp = m.d1t_row_ptr.host_ptr();
    const int* ci = m.d1t_col_idx.host_ptr();
    const Scalar* v = m.d1t_val.host_ptr();
    const Scalar* h2 = m.hodge2.host_ptr();
    const Scalar* h1i = m.hodge1_inv.host_ptr();
    for (int e = 0; e < m.m_N_edges; e++) {
      Scalar curl_H = Scalar(0);
      for (int j = rp[e]; j < rp[e + 1]; j++) {
        int f = ci[j];
        curl_H += v[j] * h2[f] * B[f];
      }
      E[e] += dt * h1i[e] * (curl_H - J[e]);
    }
  }

  void compute_rhs(const std::vector<Scalar>& Ein,
                   const std::vector<Scalar>& Bin, std::vector<Scalar>& dE,
                   std::vector<Scalar>& dB) const {
    auto const& m = *mesh;
    const int* rp = m.d1_row_ptr.host_ptr();
    const int* ci = m.d1_col_idx.host_ptr();
    const Scalar* v = m.d1_val.host_ptr();
    for (int f = 0; f < m.m_N_faces; f++) {
      Scalar curl_E = Scalar(0);
      for (int j = rp[f]; j < rp[f + 1]; j++) curl_E += v[j] * Ein[ci[j]];
      dB[f] = -curl_E;
    }
    const int* trp = m.d1t_row_ptr.host_ptr();
    const int* tci = m.d1t_col_idx.host_ptr();
    const Scalar* tv = m.d1t_val.host_ptr();
    const Scalar* h2 = m.hodge2.host_ptr();
    const Scalar* h1i = m.hodge1_inv.host_ptr();
    for (int e = 0; e < m.m_N_edges; e++) {
      Scalar curl_H = Scalar(0);
      for (int j = trp[e]; j < trp[e + 1]; j++) {
        int f = tci[j];
        curl_H += tv[j] * h2[f] * Bin[f];
      }
      dE[e] = h1i[e] * (curl_H - J[e]);
    }
  }

  void damping_on(std::vector<Scalar>& Ex, std::vector<Scalar>& Bx, double dt,
                  int len, Scalar coef, Scalar expnt) const {
    auto const& m = *mesh;
    int k_start = m.m_N_r - len;
    if (k_start < 1) k_start = 1;
    const int* erl = m.edge_radial_layer.host_ptr();
    const int* frl = m.face_radial_layer.host_ptr();
    for (int e = 0; e < m.m_N_edges; e++) {
      int k = erl[e];
      if (k >= k_start) {
        Scalar ramp = Scalar(k - k_start + 1) / Scalar(len);
        Ex[e] *= std::exp(-coef * std::pow(ramp, expnt) * Scalar(dt));
      }
    }
    for (int f = 0; f < m.m_N_faces; f++) {
      int k = frl[f];
      if (k >= k_start) {
        Scalar ramp = Scalar(k - k_start + 1) / Scalar(len);
        Bx[f] *= std::exp(-coef * std::pow(ramp, expnt) * Scalar(dt));
      }
    }
  }
  void damping(double dt, int len, Scalar coef, Scalar expnt) {
    damping_on(E, B, dt, len, coef, expnt);
  }

  // Same index arithmetic as dec_field_solver::apply_pec_bc.
  void pec_on(std::vector<Scalar>& Ex, std::vector<Scalar>& Bx) const {
    auto const& m = *mesh;
    for (int e = 0; e < m.m_N_edge_s; e++) {
      Ex[e] = Scalar(0);
      Ex[m.m_N_r * m.m_N_edge_s + e] = Scalar(0);
    }
    for (int t = 0; t < m.m_N_tri; t++) {
      Bx[t] = Scalar(0);
      Bx[m.m_N_r * m.m_N_tri + t] = Scalar(0);
    }
  }
  void pec() { pec_on(E, B); }

  // Literal transcription of dec_field_solver::update_semi_implicit.
  void semi_step(double dt, Scalar beta, int iters, int damp_len,
                 Scalar damp_coef, Scalar damp_exp) {
    Scalar alpha = Scalar(1) - beta;
    std::vector<Scalar> dE(E.size()), dB(B.size());
    std::vector<Scalar> dE2(E.size()), dB2(B.size());
    std::vector<Scalar> tmpE(E.size()), tmpB(B.size());
    compute_rhs(E, B, dE, dB);
    for (size_t e = 0; e < E.size(); e++) tmpE[e] = E[e] + dt * dE[e];
    for (size_t f = 0; f < B.size(); f++) tmpB[f] = B[f] + dt * dB[f];
    damping_on(tmpE, tmpB, dt, damp_len, damp_coef, damp_exp);
    pec_on(tmpE, tmpB);
    for (int it = 0; it < iters; it++) {
      compute_rhs(tmpE, tmpB, dE2, dB2);
      for (size_t e = 0; e < E.size(); e++)
        tmpE[e] = E[e] + dt * (alpha * dE[e] + beta * dE2[e]);
      for (size_t f = 0; f < B.size(); f++)
        tmpB[f] = B[f] + dt * (alpha * dB[f] + beta * dB2[f]);
      damping_on(tmpE, tmpB, dt, damp_len, damp_coef, damp_exp);
      pec_on(tmpE, tmpB);
    }
    E = tmpE;
    B = tmpB;
    pec_on(E, B);
  }
};

// -------------------------------------------------------------------------
// One rank's solver instance + local field state.
// -------------------------------------------------------------------------
struct rank_ctx {
  prismatic_partition part;
  std::unique_ptr<prismatic_mesh_partition> mp;
  core_t core;
  buffer<Scalar> E, B, J, B0;
  // Scratch for the semi-implicit path.
  buffer<Scalar> tmpE, tmpB, dE, dB, dE2, dB2;
};

rank_ctx make_ctx(const prismatic_mesh& mesh, const icosphere_topology& topo,
                  prismatic_partition part, const global_ref& g0,
                  halo_depth depth = halo_depth::solver) {
  rank_ctx c;
  part.set_topology(&topo);
  c.part = part;
  c.mp = std::make_unique<prismatic_mesh_partition>(
      prismatic_mesh_partition::build(part, topo, depth));
  c.core.build(mesh, *c.mp);
  auto alloc = [](buffer<Scalar>& b, int n) {
    b.set_memtype(MemType::host_only);
    b.resize(n);
    b.assign(Scalar(0));
  };
  alloc(c.E, c.core.n_edges_local());
  alloc(c.J, c.core.n_edges_local());
  alloc(c.B, c.core.n_faces_local());
  alloc(c.B0, c.core.n_faces_local());
  alloc(c.tmpE, c.core.n_edges_local());
  alloc(c.dE, c.core.n_edges_local());
  alloc(c.dE2, c.core.n_edges_local());
  alloc(c.tmpB, c.core.n_faces_local());
  alloc(c.dB, c.core.n_faces_local());
  alloc(c.dB2, c.core.n_faces_local());
  c.core.edge_from_global(g0.E.data(), c.E);
  c.core.edge_from_global(g0.J.data(), c.J);
  c.core.face_from_global(g0.B.data(), c.B);
  return c;
}

// -------------------------------------------------------------------------
// Lockstep halo exchange across in-process "ranks" — same pairing
// semantics as in_process_halo_backend::exchange_all, but resolved per
// sub-communicator: angular peer_rank is the peer's ANGULAR rank
// (ico-face index for legacy identity partitions, path-ordered rank
// for canonical unit partitions — both stored in part.angular_rank),
// radial peer_rank is the peer's radial rank within the same angular
// rank.
// -------------------------------------------------------------------------
void exchange(std::vector<rank_ctx>& ranks, cochain_type ct,
              const std::function<Scalar*(rank_ctx&)>& base) {
  auto find = [&](int ang, int rad) -> rank_ctx* {
    for (auto& r : ranks)
      if (r.part.angular_rank == ang && r.part.radial_rank == rad)
        return &r;
    return nullptr;
  };
  auto peer_entry_for = [](const halo_plan& p, int rank) {
    for (auto const& pe : p.peers)
      if (pe.peer_rank == rank) return &pe;
    return static_cast<const halo_plan::peer_entry*>(nullptr);
  };

  // Angular axis first, then radial — the production exchanger order
  // (radial forwarding of angular ghosts under pic-depth plans; for
  // solver-depth plans the order is immaterial).
  for (auto& a : ranks) {
    auto const& pa = a.mp->angular_plan_local(ct);
    for (auto const& pe : pa.peers) {
      rank_ctx* b = find(pe.peer_rank, a.part.radial_rank);
      REQUIRE(b != nullptr);
      auto const* peb =
          peer_entry_for(b->mp->angular_plan_local(ct), a.part.angular_rank);
      REQUIRE(peb != nullptr);
      REQUIRE(pe.recv_global_idx.size() == peb->send_global_idx.size());
      Scalar* ba = base(a);
      Scalar* bb = base(*b);
      for (size_t i = 0; i < pe.recv_global_idx.size(); ++i)
        ba[pe.recv_global_idx[i]] = bb[peb->send_global_idx[i]];
    }
  }
  // Radial axis.
  for (auto& a : ranks) {
    auto const& pa = a.mp->radial_plan_local(ct);
    for (auto const& pe : pa.peers) {
      rank_ctx* b = find(a.part.angular_rank, pe.peer_rank);
      REQUIRE(b != nullptr);
      auto const* peb =
          peer_entry_for(b->mp->radial_plan_local(ct), a.part.radial_rank);
      REQUIRE(peb != nullptr);
      REQUIRE(pe.recv_global_idx.size() == peb->send_global_idx.size());
      Scalar* ba = base(a);
      Scalar* bb = base(*b);
      for (size_t i = 0; i < pe.recv_global_idx.size(); ++i)
        ba[pe.recv_global_idx[i]] = bb[peb->send_global_idx[i]];
    }
  }
}

using field_of = std::function<buffer<Scalar>&(rank_ctx&)>;

void exchange_edge_field(std::vector<rank_ctx>& ranks, const field_of& get) {
  exchange(ranks, cochain_type::h_edge,
           [&](rank_ctx& c) { return get(c).host_ptr(); });
  exchange(ranks, cochain_type::v_edge,
           [&](rank_ctx& c) { return get(c).host_ptr() + c.core.e_split(); });
}
void exchange_face_field(std::vector<rank_ctx>& ranks, const field_of& get) {
  exchange(ranks, cochain_type::tri_face,
           [&](rank_ctx& c) { return get(c).host_ptr(); });
  exchange(ranks, cochain_type::rect_face,
           [&](rank_ctx& c) { return get(c).host_ptr() + c.core.b_split(); });
}
void exchange_E(std::vector<rank_ctx>& ranks) {
  exchange_edge_field(ranks, [](rank_ctx& c) -> buffer<Scalar>& { return c.E; });
}
void exchange_B(std::vector<rank_ctx>& ranks) {
  exchange_face_field(ranks, [](rank_ctx& c) -> buffer<Scalar>& { return c.B; });
}

// Gather every rank's owned cells into global arrays and return the max
// abs difference against the reference, normalized by the reference's
// max magnitude.
Scalar compare_owned(std::vector<rank_ctx>& ranks,
                     const std::vector<Scalar>& E_ref,
                     const std::vector<Scalar>& B_ref) {
  std::vector<Scalar> Eg(E_ref.size(), Scalar(0)), Bg(B_ref.size(), Scalar(0));
  for (auto& c : ranks) {
    c.core.edge_owned_to_global(c.E, Eg.data());
    c.core.face_owned_to_global(c.B, Bg.data());
  }
  Scalar scale = Scalar(0), diff = Scalar(0);
  for (size_t e = 0; e < E_ref.size(); e++) {
    scale = std::max(scale, std::abs(E_ref[e]));
    diff = std::max(diff, std::abs(Eg[e] - E_ref[e]));
  }
  for (size_t f = 0; f < B_ref.size(); f++) {
    scale = std::max(scale, std::abs(B_ref[f]));
    diff = std::max(diff, std::abs(Bg[f] - B_ref[f]));
  }
  return diff / scale;
}

// One full explicit step across all ranks with the plan's sync points:
// exchange E (h+v) -> Faraday -> exchange B (tri+rect) -> Ampere ->
// damping -> boundary.
void step_all(std::vector<rank_ctx>& ranks, double dt, int damp_len,
              Scalar damp_coef, Scalar damp_exp,
              const std::function<void(rank_ctx&)>& boundary) {
  exchange_E(ranks);
  for (auto& c : ranks) c.core.faraday(c.E, c.B, dt);
  exchange_B(ranks);
  for (auto& c : ranks) c.core.ampere(c.E, c.B, c.J, dt);
  for (auto& c : ranks)
    c.core.apply_damping(c.E, c.B, dt, damp_len, damp_coef, damp_exp);
  for (auto& c : ranks) boundary(c);
}

// One semi-implicit step across all ranks: the ghost refresh runs
// before the initial RHS and inside EVERY Picard iteration (the
// silent-drift gotcha called out in PHASE_4_1B_PLAN.md).
void step_all_semi(std::vector<rank_ctx>& ranks, double dt, Scalar beta,
                   int iters, int damp_len, Scalar damp_coef,
                   Scalar damp_exp) {
  Scalar alpha = Scalar(1) - beta;
  auto tmpE_of = [](rank_ctx& c) -> buffer<Scalar>& { return c.tmpE; };
  auto tmpB_of = [](rank_ctx& c) -> buffer<Scalar>& { return c.tmpB; };

  exchange_E(ranks);
  exchange_B(ranks);
  for (auto& c : ranks) {
    c.core.compute_rhs(c.E, c.B, c.J, c.dE, c.dB);
    c.core.euler_predict(c.E, c.dE, c.tmpE, c.B, c.dB, c.tmpB, dt);
    c.core.apply_damping(c.tmpE, c.tmpB, dt, damp_len, damp_coef, damp_exp);
    c.core.apply_pec_bc(c.tmpE, c.tmpB);
  }
  for (int it = 0; it < iters; it++) {
    exchange_edge_field(ranks, tmpE_of);
    exchange_face_field(ranks, tmpB_of);
    for (auto& c : ranks) {
      c.core.compute_rhs(c.tmpE, c.tmpB, c.J, c.dE2, c.dB2);
      c.core.picard_combine(c.E, c.dE, c.dE2, c.tmpE, c.B, c.dB, c.dB2,
                            c.tmpB, dt, alpha, beta);
      c.core.apply_damping(c.tmpE, c.tmpB, dt, damp_len, damp_coef, damp_exp);
      c.core.apply_pec_bc(c.tmpE, c.tmpB);
    }
  }
  for (auto& c : ranks) {
    // Copy-back (ghosts included; they are refreshed before any read).
    for (int e = 0; e < c.core.n_edges_local(); e++) c.E[e] = c.tmpE[e];
    for (int f = 0; f < c.core.n_faces_local(); f++) c.B[f] = c.tmpB[f];
    c.core.apply_pec_bc(c.E, c.B);
  }
}

// Canonical A x K unit decomposition (Phase 7A) — path-ordered angular
// ranks via prismatic_partition::combined.
std::vector<rank_ctx> make_partitioned_units(const prismatic_mesh& mesh,
                                             const icosphere_topology& topo,
                                             const global_ref& g0, int A,
                                             int K) {
  std::vector<rank_ctx> ranks;
  ranks.reserve(A * K);
  for (int w = 0; w < A * K; w++)
    ranks.push_back(make_ctx(
        mesh, topo, prismatic_partition::combined(TL, TN_r, A, K, w), g0));
  return ranks;
}

std::vector<rank_ctx> make_partitioned(const prismatic_mesh& mesh,
                                       const icosphere_topology& topo,
                                       const global_ref& g0,
                                       int n_radial, bool angular) {
  std::vector<rank_ctx> ranks;
  if (angular && n_radial > 1) {
    for (int r = 0; r < n_radial; r++)
      for (int f = 0; f < 20; f++)
        ranks.push_back(make_ctx(
            mesh, topo,
            prismatic_partition::combined_ico_face(TL, TN_r, n_radial, r, f), g0));
  } else if (angular) {
    for (int f = 0; f < 20; f++)
      ranks.push_back(make_ctx(
          mesh, topo, prismatic_partition::ico_face_angular(TL, TN_r, f), g0));
  } else {
    for (int r = 0; r < n_radial; r++)
      ranks.push_back(make_ctx(
          mesh, topo, prismatic_partition::radial_slab(TL, TN_r, n_radial, r),
          g0));
  }
  return ranks;
}

}  // namespace

// =========================================================================
// B1 acceptance: the distributed core, driven with lockstep halo
// exchanges, reproduces the global explicit update on every owned cell
// for all partition shapes.
// =========================================================================
TEST_CASE("dec_dist: distributed explicit update matches global solver",
          "[prismatic][dec_dist]") {
  const int n_steps = 10;
  const int damp_len = 3;
  const Scalar damp_coef = 0.5, damp_exp = 3.0;

  prismatic_mesh mesh;
  mesh.build(TL, TN_r, 1.0, 2.0);
  auto topo = icosphere_topology::build_from_mesh(mesh);

  global_ref ref;
  ref.init(mesh);
  global_ref g0 = ref;  // pristine IC for seeding the rank contexts

  // Advance the reference.
  for (int s = 0; s < n_steps; s++) {
    ref.faraday(TDT);
    ref.ampere(TDT);
    ref.damping(TDT, damp_len, damp_coef, damp_exp);
    ref.pec();
  }

  auto pec_boundary = [](rank_ctx& c) { c.core.apply_pec_bc(c.E, c.B); };

  auto run_and_check = [&](std::vector<rank_ctx> ranks, Scalar tol) {
    for (int s = 0; s < n_steps; s++)
      step_all(ranks, TDT, damp_len, damp_coef, damp_exp, pec_boundary);
    Scalar rel = compare_owned(ranks, ref.E, ref.B);
    INFO("n_ranks = " << ranks.size() << ", max rel diff = " << rel);
    REQUIRE(rel < tol);
  };

  SECTION("single rank (bit-compatible path)") {
    std::vector<rank_ctx> ranks;
    ranks.push_back(
        make_ctx(mesh, topo, prismatic_partition::single_rank(TL, TN_r), g0));
    run_and_check(std::move(ranks), Scalar(1e-6));
  }
  SECTION("20-rank angular") {
    run_and_check(make_partitioned(mesh, topo, g0, 1, true), Scalar(2e-5));
  }
  SECTION("radial slabs, K = 3") {
    run_and_check(make_partitioned(mesh, topo, g0, 3, false), Scalar(2e-5));
  }
  SECTION("combined 20 x 4") {
    run_and_check(make_partitioned(mesh, topo, g0, 4, true), Scalar(2e-5));
  }
  // Phase 7A canonical unit partitions (path-ordered angular ranks,
  // generic plan builder).
  SECTION("canonical A=4 x K=2 (single Frontier node shape)") {
    run_and_check(make_partitioned_units(mesh, topo, g0, 4, 2), Scalar(2e-5));
  }
  SECTION("canonical A=20 x K=2") {
    run_and_check(make_partitioned_units(mesh, topo, g0, 20, 2),
                  Scalar(2e-5));
  }
  SECTION("canonical A=80 (sub-face quarter units, m=1)") {
    run_and_check(make_partitioned_units(mesh, topo, g0, 80, 1),
                  Scalar(2e-5));
  }
  // Phase 7B: the solver over PIC-DEPTH layouts (larger ghost sets, one
  // layout per cochain for a PIC run) must still reproduce the global
  // update — ghosts are exact copies regardless of set size.
  SECTION("canonical A=4 x K=2, pic-depth layouts") {
    std::vector<rank_ctx> ranks;
    for (int w = 0; w < 8; w++)
      ranks.push_back(make_ctx(mesh, topo,
                               prismatic_partition::combined(TL, TN_r, 4, 2, w),
                               g0, halo_depth::pic));
    run_and_check(std::move(ranks), Scalar(2e-5));
  }
  SECTION("canonical A=80, pic-depth layouts") {
    std::vector<rank_ctx> ranks;
    for (int w = 0; w < 80; w++)
      ranks.push_back(make_ctx(mesh, topo,
                               prismatic_partition::combined(TL, TN_r, 80, 1, w),
                               g0, halo_depth::pic));
    run_and_check(std::move(ranks), Scalar(2e-5));
  }
}

// =========================================================================
// Semi-implicit path: distributed predictor-corrector with per-Picard-
// iteration ghost refresh matches the global transcription.
// =========================================================================
TEST_CASE("dec_dist: distributed semi-implicit update matches global solver",
          "[prismatic][dec_dist]") {
  const int n_steps = 6;
  const int iters = 4;
  const Scalar beta = 0.55;
  const int damp_len = 3;
  const Scalar damp_coef = 0.5, damp_exp = 3.0;

  prismatic_mesh mesh;
  mesh.build(TL, TN_r, 1.0, 2.0);
  auto topo = icosphere_topology::build_from_mesh(mesh);

  global_ref ref;
  ref.init(mesh);
  global_ref g0 = ref;
  for (int s = 0; s < n_steps; s++)
    ref.semi_step(TDT, beta, iters, damp_len, damp_coef, damp_exp);

  auto run_and_check = [&](std::vector<rank_ctx> ranks, Scalar tol) {
    for (int s = 0; s < n_steps; s++)
      step_all_semi(ranks, TDT, beta, iters, damp_len, damp_coef, damp_exp);
    Scalar rel = compare_owned(ranks, ref.E, ref.B);
    INFO("n_ranks = " << ranks.size() << ", max rel diff = " << rel);
    REQUIRE(rel < tol);
  };

  SECTION("single rank") {
    std::vector<rank_ctx> ranks;
    ranks.push_back(
        make_ctx(mesh, topo, prismatic_partition::single_rank(TL, TN_r), g0));
    run_and_check(std::move(ranks), Scalar(1e-6));
  }
  SECTION("combined 20 x 4") {
    run_and_check(make_partitioned(mesh, topo, g0, 4, true), Scalar(2e-5));
  }
  SECTION("canonical A=4 x K=2") {
    run_and_check(make_partitioned_units(mesh, topo, g0, 4, 2), Scalar(2e-5));
  }
  SECTION("canonical A=80 (sub-face quarter units, m=1)") {
    run_and_check(make_partitioned_units(mesh, topo, g0, 80, 1),
                  Scalar(2e-5));
  }
}

// =========================================================================
// Inner rotating-dipole BC: the quadrature path (which reaches global
// mesh geometry through the l2g maps) must agree between a single-rank
// core and a fully-partitioned run.
// =========================================================================
TEST_CASE("dec_dist: inner dipole BC consistent across partitioning",
          "[prismatic][dec_dist]") {
  const int n_steps = 5;
  const int damp_len = 3;
  const Scalar damp_coef = 0.5, damp_exp = 3.0;

  prismatic_mesh mesh;
  mesh.build(TL, TN_r, 1.0, 2.0);
  auto topo = icosphere_topology::build_from_mesh(mesh);

  global_ref g0;
  g0.init(mesh);

  dec_inner_bc_params par;
  par.Bp = 100.0;
  par.Omega = 0.25;
  par.obliquity = 0.3;
  par.use_deutsch = false;
  par.overwrite_b = true;

  double time = 0.0;
  auto inner_boundary = [&](rank_ctx& c) {
    // Leapfrog staggering as in update_explicit: E at t+dt, B at t+dt/2.
    c.core.apply_inner_bc(c.E, c.B, c.B0, par, time + TDT, time + 0.5 * TDT);
  };

  auto run = [&](std::vector<rank_ctx> ranks) {
    time = 0.0;
    for (int s = 0; s < n_steps; s++) {
      step_all(ranks, TDT, damp_len, damp_coef, damp_exp, inner_boundary);
      time += TDT;
    }
    std::vector<Scalar> Eg(mesh.m_N_edges, Scalar(0)),
        Bg(mesh.m_N_faces, Scalar(0));
    for (auto& c : ranks) {
      c.core.edge_owned_to_global(c.E, Eg.data());
      c.core.face_owned_to_global(c.B, Bg.data());
    }
    return std::make_pair(std::move(Eg), std::move(Bg));
  };

  std::vector<rank_ctx> single;
  single.push_back(
      make_ctx(mesh, topo, prismatic_partition::single_rank(TL, TN_r), g0));
  auto single_result = run(std::move(single));
  const std::vector<Scalar>& E1 = single_result.first;
  const std::vector<Scalar>& B1 = single_result.second;

  auto check_against_single = [&](std::vector<rank_ctx> ranks,
                                  const char* label) {
    auto [EN, BN] = run(std::move(ranks));
    Scalar scale = Scalar(0), diff = Scalar(0);
    for (int e = 0; e < mesh.m_N_edges; e++) {
      scale = std::max(scale, std::abs(E1[e]));
      diff = std::max(diff, std::abs(EN[e] - E1[e]));
    }
    for (int f = 0; f < mesh.m_N_faces; f++) {
      scale = std::max(scale, std::abs(B1[f]));
      diff = std::max(diff, std::abs(BN[f] - B1[f]));
    }
    INFO(label << ": max rel diff vs 1 rank = " << diff / scale);
    REQUIRE(diff / scale < Scalar(2e-5));
  };

  check_against_single(make_partitioned(mesh, topo, g0, 4, true),
                       "legacy 20 x 4");
  check_against_single(make_partitioned_units(mesh, topo, g0, 4, 2),
                       "canonical 4 x 2");
  check_against_single(make_partitioned_units(mesh, topo, g0, 80, 1),
                       "canonical 80 x 1");
}
