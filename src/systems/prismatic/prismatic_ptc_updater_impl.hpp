#pragma once

#include "systems/prismatic/prismatic_ptc_updater.h"
#include "systems/prismatic/prismatic_ptc_update_kernel.hpp"
#include "core/particles_functions.h"
#include "framework/environment.h"
#include "utils/logger.h"
#include <cmath>

namespace Aperture {

template <typename ExecPolicy>
prismatic_ptc_updater<ExecPolicy>::prismatic_ptc_updater(
    prismatic_mesh& mesh, const prismatic_mesh_partition* mp,
    const prismatic_mpi_comm* comm)
    : m_mesh(mesh), m_mp(mp), m_comm(comm) {
  m_distributed = mp != nullptr && comm != nullptr && !comm->is_single_rank();
}

template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::register_data_components() {
  auto mem = ExecPolicy::data_mem_type();
  if (m_distributed) {
    // Local solver-shared fields: totals in, deposits out.  Idempotent
    // with the solver's registrations (same names, same bundle).
    m_E = sim_env().template register_data<prismatic_edge_field>(
        "E", *m_mp, mem);
    m_B = sim_env().template register_data<prismatic_face_field>(
        "B", *m_mp, mem);
    m_J = sim_env().template register_data<prismatic_edge_field>(
        "J", *m_mp, mem);
    m_rho = sim_env().template register_data<prismatic_vertex_field>(
        "rho", *m_mp, mem);
    m_rho_abs = sim_env().template register_data<prismatic_vertex_field>(
        "rho_abs", *m_mp, mem);
    m_gamma_wsum = sim_env().template register_data<prismatic_vertex_field>(
        "gamma_wsum", *m_mp, mem);
  } else {
    m_E = sim_env().template register_data<prismatic_edge_field>("E", m_mesh, mem);
    m_B = sim_env().template register_data<prismatic_face_field>("B", m_mesh, mem);
    m_J = sim_env().template register_data<prismatic_edge_field>("J", m_mesh, mem);
    m_rho = sim_env().template register_data<prismatic_vertex_field>("rho", m_mesh, mem);
    m_rho_abs = sim_env().template register_data<prismatic_vertex_field>(
        "rho_abs", m_mesh, mem);
    m_gamma_wsum = sim_env().template register_data<prismatic_vertex_field>(
        "gamma_wsum", m_mesh, mem);
  }
  m_J->set_edge_kind(EdgeCochainKind::dual_2);

  int max_ptc = 100000;
  sim_env().params().get_value("max_ptc_num", max_ptc);
  m_ptc = sim_env().template register_data<prismatic_particle_data>(
      "particles", max_ptc, mem);

  size_t seed = default_random_seed;
  sim_env().params().get_value("random_seed", seed);
  m_rng_states =
      sim_env()
          .template register_data<rng_states_t<typename ExecPolicy::exec_tag>>(
              "rng_states", seed);
  m_rng_states->skip_output(true);
  m_rng_states->include_in_snapshot(true);

  // -----------------------------------------------------------------------
  // The local particle mesh is built HERE (registration time), not in
  // init(): the injector systems run their init() BEFORE ours (they are
  // registered first so injection precedes the push in a step) and need
  // ptc_mesh() there.
  // Distributed: the shared pic-depth bundle (canonical comm required).
  // Single-rank: an identity bundle — identity tables and maps,
  // bit-exact with the old global path.
  // -----------------------------------------------------------------------
  sim_env().params().get_value("use_recovery_gather", m_use_recovery_gather);
  if (m_use_recovery_gather) {
    // Weights are per SPHERE vertex (replicated data) — built globally,
    // then remapped into the local mesh below.
    m_recovery.build(m_mesh);
  }
  const prismatic_mesh_partition* mp = m_mp;
  if (m_distributed) {
    if (!m_comm->canonical_rank_order()) {
      Logger::print_err(
          "prismatic_ptc_updater (7C) requires a canonical A*K comm "
          "(prismatic_mpi_comm::create(world, A, K))");
      std::abort();
    }
    if (m_mp->depth() != halo_depth::pic) {
      Logger::print_err(
          "prismatic_ptc_updater requires a pic-depth mesh_partition "
          "(prismatic_mesh_partition::build(part, topo, halo_depth::pic))");
      std::abort();
    }
  } else {
    m_topo_own = icosphere_topology::build_from_mesh(m_mesh);
    auto part = prismatic_partition::single_rank(m_mesh.m_L, m_mesh.m_N_r);
    part.set_topology(&m_topo_own);
    m_mp_own = prismatic_mesh_partition::build(part, m_topo_own);
    mp = &m_mp_own;
  }
  m_lmesh.build(m_mesh, *mp,
                m_use_recovery_gather ? &m_recovery : nullptr,
                ExecPolicy::data_mem_type());
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  m_lmesh.copy_to_device();
#endif
  if (m_use_recovery_gather) {
    m_Bv.set_memtype(ExecPolicy::data_mem_type());
    m_Bv.resize(size_t(3) * m_lmesh.host_ptrs().N_verts);
    m_Bv.assign(Scalar(0));
  }
}

template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::init() {
  sim_env().params().get_value("q_e", m_charge_e);
  sim_env().params().get_value("m_e", m_mass_e);
  sim_env().params().get_value("sort_interval", m_sort_interval);
  sim_env().params().get_value("use_gca", m_use_gca);
  sim_env().params().get_value("include_curvature", m_include_curvature);
  sim_env().params().get_value("gca_switch_omegac_dt", m_gca_switch_wc);
  sim_env().params().get_value("gca_zero_mu_on_capture", m_gca_zero_mu);
  sim_env().params().get_value("ptc_absorb_radius", m_absorb_radius);
  sim_env().params().get_value("deposit_diagnostics", m_deposit_diagnostics);
  if (m_absorb_radius > Scalar(0)) {
    Logger::print_info("Particle absorption radius: {}", m_absorb_radius);
  }

  if (m_distributed) {
    bool halo_device_direct = true;
    sim_env().params().get_value("halo_device_direct", halo_device_direct);
    m_ex.init(*m_mp, *m_comm, halo_device_direct);
    m_world_rank = m_comm->world_rank();
    m_world_size = m_comm->world_size();

    m_mig_count.set_memtype(ExecPolicy::data_mem_type());
    m_mig_cursor.set_memtype(ExecPolicy::data_mem_type());
    m_mig_count.resize(m_world_size);
    m_mig_cursor.resize(m_world_size);
    // Non-zero initial capacity so host_ptr() is valid even on steps
    // with nothing to send (MPI gets zero counts but a real pointer).
    for (auto& b : m_snd_s) {
      b.set_memtype(ExecPolicy::data_mem_type());
      b.resize(16);
    }
    for (auto* b : {&m_snd_cell, &m_snd_flag}) {
      b->set_memtype(ExecPolicy::data_mem_type());
      b->resize(16);
    }
    m_snd_id.set_memtype(ExecPolicy::data_mem_type());
    m_snd_id.resize(16);
    Logger::print_info(
        "Distributed particle updater: rank {} owns {} tris (of {} local), "
        "layers [{}, {}) at k0 = {}",
        m_world_rank, m_lmesh.n_tri_own(), m_lmesh.n_tri_local(),
        m_lmesh.lay_own_lo(), m_lmesh.lay_own_hi(), m_lmesh.k0());

    // 7D memory audit: the per-rank footprint must scale ~ 1/(A*K) plus
    // the O(4^L) replicated angular-table constant (weak-scaling smoke
    // checks this across A*K shapes).
    auto lp_sz = m_lmesh.host_ptrs();
    const size_t local_3d_bytes =
        sizeof(Scalar) * (size_t(m_E->data().size()) + m_B->data().size() +
                          m_J->data().size() + 3 * m_rho->data().size() +
                          m_Bv.size()) +
        sizeof(int) * (size_t(lp_sz.N_r + 1) *
                           (lp_sz.N_edge_s + lp_sz.N_tri + lp_sz.N_vert_s) +
                       size_t(lp_sz.N_r) * (lp_sz.N_vert_s + lp_sz.N_edge_s));
    const size_t angular_bytes =
        size_t(m_mesh.m_N_tri) * 3 * sizeof(int) * 4 +
        size_t(m_mesh.m_N_vert_s) * 5 * sizeof(Scalar) +
        size_t(m_mesh.m_N_tri) * sizeof(double) +
        size_t(m_mesh.m_N_edge_s) * (2 * sizeof(double) + 4 * sizeof(int)) +
        size_t(m_mesh.m_N_vert_s) * sizeof(double);
    Logger::print_info(
        "Per-rank footprint: local 3D (fields+maps+Bv) ~ {:.1f} kB, "
        "replicated angular tables ~ {:.1f} kB, particle buffer {:.1f} MB",
        local_3d_bytes / 1.0e3, angular_bytes / 1.0e3,
        double(m_ptc->size()) * (8 * sizeof(Scalar) + 2 * 4 + 8) / 1.0e6);
  }

  Logger::print_info("Prismatic particle updater initialized: {} particles",
                     m_ptc->size());
}

// ===========================================================================
// Field sync point: E/B pic halos + the recovery Bv (owned-slot fit +
// 3-component vertex halo exchange).  Collective; idempotent per step.
// ===========================================================================
template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::sync_fields(uint32_t step) {
  if (step == m_synced_step) return;
  m_synced_step = step;

  if (m_distributed) {
    m_ex.exchange_edge(m_E->data(), m_E->split());
    m_ex.exchange_face(m_B->data(), m_B->split());
  }

  if (m_use_recovery_gather) {
    auto lmp = m_lmesh.get_ptrs(typename ExecPolicy::exec_tag{});
    const int n_shell_own = lmp.shell_own_hi - lmp.shell_own_lo;
    const int n_own_slots = n_shell_own * lmp.n_vert_s_own;
    ExecPolicy::launch(
        [lmp, n_own_slots] LAMBDA(auto B_f, auto Bv) {
          ExecPolicy::loop(0, n_own_slots, [&] LAMBDA(int i) {
            const int k = lmp.shell_own_lo + i / lmp.n_vert_s_own;
            const int s = i % lmp.n_vert_s_own;
            compute_vertex_B_local(lmp, B_f, Bv, k, s);
          });
        },
        m_B->data(), m_Bv);
    ExecPolicy::sync();
    if (m_distributed) {
      const int stride = m_lmesh.host_ptrs().N_verts;
      for (int c = 0; c < 3; ++c) {
        m_ex.exchange_vertex(m_Bv, c * stride);
      }
    }
  }
}

template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::update(double dt, uint32_t step) {
  sync_fields(step);

  auto lmp = m_lmesh.get_ptrs(typename ExecPolicy::exec_tag{});
  const int N_tri = lmp.N_tri;
  size_t num = m_ptc->number();
  Scalar charge_e = m_charge_e;
  Scalar mass_e = m_mass_e;

  // Clear rho, J, and the diagnostic deposits before deposit (full local
  // arrays — ghost slots absorb off-rank stencil ends until reduce()).
  ExecPolicy::launch(
      [N_edges = int(m_J->data().size()), N_verts = int(m_rho->data().size())]
      LAMBDA(auto J_e, auto rho, auto rho_abs, auto gamma_wsum) {
        ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
          J_e[e] = Scalar(0);
        });
        ExecPolicy::loop(0, N_verts, [&] LAMBDA(int v) {
          rho[v] = Scalar(0);
          rho_abs[v] = Scalar(0);
          gamma_wsum[v] = Scalar(0);
        });
      },
      m_J->data(), m_rho->data(), m_rho_abs->data(), m_gamma_wsum->data());

  const Scalar* Bv_rec = nullptr;
  if (m_use_recovery_gather) {
    Bv_rec = m_Bv.dev_ptr() != nullptr ? m_Bv.dev_ptr() : m_Bv.host_ptr();
  }

  // Particle update loop
  bool use_gca = m_use_gca;
  bool include_curvature = m_include_curvature;
  Scalar absorb_r = m_absorb_radius;
  bool dep_diag = m_deposit_diagnostics;
  Scalar gca_wc = m_gca_switch_wc;
  bool gca_zero_mu = m_gca_zero_mu;
  ExecPolicy::launch(
      [num, N_tri, charge_e, mass_e, dt, lmp, use_gca, include_curvature,
       Bv_rec, absorb_r, dep_diag, gca_wc, gca_zero_mu]
      LAMBDA(auto ptc, auto E_e, auto B_f, auto J_e, auto rho,
             auto rho_abs, auto gamma_wsum) {
        ExecPolicy::loop(0, (int)num, [&] LAMBDA(int n) {
          if (ptc.cell[n] == empty_cell) return;
          int sp = get_ptc_type(ptc.flag[n]);
          Scalar q = (sp == (int)PtcType::positron) ? -charge_e : charge_e;
          update_single_particle(lmp, N_tri, ptc, n, E_e, B_f, J_e, rho,
                                 q, mass_e, Scalar(dt),
                                 use_gca, include_curvature, Bv_rec,
                                 absorb_r,
                                 dep_diag ? (Scalar*)rho_abs : nullptr,
                                 dep_diag ? (Scalar*)gamma_wsum : nullptr,
                                 gca_wc, gca_zero_mu);
        });
      },
      *m_ptc, m_E->data(), m_B->data(), m_J->data(), m_rho->data(),
      m_rho_abs->data(), m_gamma_wsum->data());

  ExecPolicy::sync();

  if (m_distributed) {
    // Fold ghost deposits into their owners (radial round first — the
    // corner relay), then refresh the ghost slots of the deposits the
    // next step's injector criteria read through their stencils.
    m_ex.reduce_edge(m_J->data(), m_J->split());
    m_ex.exchange_edge(m_J->data(), m_J->split());
    m_ex.reduce_vertex(m_rho->data());
    if (m_deposit_diagnostics) {
      m_ex.reduce_vertex(m_rho_abs->data());
      m_ex.exchange_vertex(m_rho_abs->data());
      m_ex.reduce_vertex(m_gamma_wsum->data());
    }

    migrate();
  }

  // Periodically sort particles by cell for GPU cache efficiency
  if (m_sort_interval > 0 && step % m_sort_interval == 0) {
    size_t max_cell = m_lmesh.max_cell();
    ptc_sort_by_cell(typename ExecPolicy::exec_tag{}, *m_ptc, max_cell);
  }
}

// =========================================================================
// Migration (plan F7): particles whose LOCAL cell left the owned region
// move to the owning world rank = rad·A + ang (angular rank from the
// per-tri table, radial from the global slab map).  The wire carries
// GLOBAL cell ids; pack translates local→global on the device, unpack
// global→local on the host via tri_g2l.  Device-side pack (two kernels:
// count, then place at per-destination cursors), host-staged
// MPI_Alltoallv per component, arrivals appended at the end of the
// particle array.  Sent particles become empty slots, compacted by the
// periodic sort.
// =========================================================================
template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::migrate() {
  const size_t num = m_ptc->number();
  const int ws = m_world_size;
  auto lmp = m_lmesh.get_ptrs(typename ExecPolicy::exec_tag{});

  // Pass 1: count leavers per destination.
  m_mig_count.assign(0);
  ExecPolicy::launch(
      [num, lmp] LAMBDA(auto ptc, auto count) {
        ExecPolicy::loop(0, (int)num, [&] LAMBDA(int n) {
          if (ptc.cell[n] == empty_cell) return;
          int dest = lmp.migrate_dest(ptc.cell[n]);
          if (dest >= 0) atomic_add(&count[dest], 1);
        });
      },
      *m_ptc, m_mig_count);
  ExecPolicy::sync();
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  m_mig_count.copy_to_host();
#endif

  // Host: exclusive scan -> send offsets; exchange counts.
  std::vector<int> snd_cnt(ws), snd_off(ws), rcv_cnt(ws), rcv_off(ws);
  int n_send = 0;
  for (int r = 0; r < ws; ++r) {
    snd_cnt[r] = m_mig_count[r];
    snd_off[r] = n_send;
    n_send += snd_cnt[r];
  }
  MPI_Alltoall(snd_cnt.data(), 1, MPI_INT, rcv_cnt.data(), 1, MPI_INT,
               MPI_COMM_WORLD);
  int n_recv = 0;
  for (int r = 0; r < ws; ++r) {
    rcv_off[r] = n_recv;
    n_recv += rcv_cnt[r];
  }

  // Pass 2: pack leavers at per-destination cursors and vacate them.
  // Cells are translated to the GLOBAL wire encoding here.
  if (int(m_snd_cell.size()) < n_send) {
    const size_t cap = size_t(n_send) * 2;
    for (auto& b : m_snd_s) b.resize(cap);
    m_snd_cell.resize(cap);
    m_snd_flag.resize(cap);
    m_snd_id.resize(cap);
  }
  if (n_send > 0) {
    for (int r = 0; r < ws; ++r) m_mig_cursor[r] = snd_off[r];
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    m_mig_cursor.copy_to_device();
#endif
    ExecPolicy::launch(
        [num, lmp] LAMBDA(auto ptc, auto cursor, auto sx1, auto sx2,
                          auto sx3, auto sp1, auto sp2, auto sp3,
                          auto sE, auto sw, auto scell, auto sflag,
                          auto sid) {
          ExecPolicy::loop(0, (int)num, [&] LAMBDA(int n) {
            if (ptc.cell[n] == empty_cell) return;
            int dest = lmp.migrate_dest(ptc.cell[n]);
            if (dest < 0) return;
            int slot = atomic_add(&cursor[dest], 1);
            sx1[slot] = ptc.x1[n];
            sx2[slot] = ptc.x2[n];
            sx3[slot] = ptc.x3[n];
            sp1[slot] = ptc.p1[n];
            sp2[slot] = ptc.p2[n];
            sp3[slot] = ptc.p3[n];
            sE[slot] = ptc.E[n];
            sw[slot] = ptc.weight[n];
            scell[slot] = lmp.wire_cell(ptc.cell[n]);
            sflag[slot] = ptc.flag[n];
            sid[slot] = ptc.id[n];
            ptc.cell[n] = empty_cell;
          });
        },
        *m_ptc, m_mig_cursor, m_snd_s[0], m_snd_s[1], m_snd_s[2],
        m_snd_s[3], m_snd_s[4], m_snd_s[5], m_snd_s[6], m_snd_s[7],
        m_snd_cell, m_snd_flag, m_snd_id);
    ExecPolicy::sync();
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    for (auto& b : m_snd_s) b.copy_to_host(0, n_send);
    m_snd_cell.copy_to_host(0, n_send);
    m_snd_flag.copy_to_host(0, n_send);
    m_snd_id.copy_to_host(0, n_send);
#endif
  }

  // Exchange.  (Component-wise Alltoallv; counts are identical.  Sized
  // at least 1 so .data() is a real pointer under zero counts.)
  const int rcv_cap = n_recv > 0 ? n_recv : 1;
  for (auto* v : {&m_rcv_cell, &m_rcv_flag}) v->resize(rcv_cap);
  for (auto& v : m_rcv_s) v.resize(rcv_cap);
  m_rcv_id.resize(rcv_cap);
  const MPI_Datatype st = mpi_scalar_type();
  for (int c = 0; c < 8; ++c) {
    MPI_Alltoallv(m_snd_s[c].host_ptr(), snd_cnt.data(), snd_off.data(), st,
                  m_rcv_s[c].data(), rcv_cnt.data(), rcv_off.data(), st,
                  MPI_COMM_WORLD);
  }
  MPI_Alltoallv(m_snd_cell.host_ptr(), snd_cnt.data(), snd_off.data(),
                MPI_UINT32_T, m_rcv_cell.data(), rcv_cnt.data(),
                rcv_off.data(), MPI_UINT32_T, MPI_COMM_WORLD);
  MPI_Alltoallv(m_snd_flag.host_ptr(), snd_cnt.data(), snd_off.data(),
                MPI_UINT32_T, m_rcv_flag.data(), rcv_cnt.data(),
                rcv_off.data(), MPI_UINT32_T, MPI_COMM_WORLD);
  MPI_Alltoallv(m_snd_id.host_ptr(), snd_cnt.data(), snd_off.data(),
                MPI_UINT64_T, m_rcv_id.data(), rcv_cnt.data(),
                rcv_off.data(), MPI_UINT64_T, MPI_COMM_WORLD);

  if (n_recv == 0) return;

  // Append arrivals at the end of the particle array, translating the
  // GLOBAL wire cells to this rank's local encoding (host-side, per
  // arrival — cheap at migration counts).
  if (num + n_recv > m_ptc->size()) {
    Logger::print_err(
        "prismatic_ptc_updater::migrate: particle buffer overflow "
        "({} + {} arrivals > {})",
        num, n_recv, m_ptc->size());
    std::abort();
  }
  const auto& g2l = m_lmesh.tri_g2l();
  const int N_tri_glob = m_lmesh.n_tri_global();
  const int n_tri_loc = m_lmesh.n_tri_local();
  const int k0 = m_lmesh.k0();
  for (int i = 0; i < n_recv; ++i) {
    const uint32_t wc = m_rcv_cell[i];
    const int glay = int(wc) / N_tri_glob;
    const int gtri = int(wc) - glay * N_tri_glob;
    const int ltri = g2l[gtri];
    const int llay = glay - k0;
    if (ltri < 0 || llay < 0) {
      Logger::print_err(
          "prismatic_ptc_updater::migrate: arrival misrouted (global cell "
          "{} not in this rank's halo)",
          wc);
      std::abort();
    }
    m_rcv_cell[i] = uint32_t(llay * n_tri_loc + ltri);
  }

  auto hp = m_ptc->get_host_ptrs();
  const Scalar* rs[8] = {m_rcv_s[0].data(), m_rcv_s[1].data(),
                         m_rcv_s[2].data(), m_rcv_s[3].data(),
                         m_rcv_s[4].data(), m_rcv_s[5].data(),
                         m_rcv_s[6].data(), m_rcv_s[7].data()};
  Scalar* ds[8] = {hp.x1, hp.x2, hp.x3, hp.p1, hp.p2, hp.p3, hp.E,
                   hp.weight};
  for (int c = 0; c < 8; ++c) {
    std::copy(rs[c], rs[c] + n_recv, ds[c] + num);
  }
  std::copy(m_rcv_cell.begin(), m_rcv_cell.begin() + n_recv, hp.cell + num);
  std::copy(m_rcv_flag.begin(), m_rcv_flag.begin() + n_recv, hp.flag + num);
  std::copy(m_rcv_id.begin(), m_rcv_id.begin() + n_recv, hp.id + num);
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  m_ptc->x1.copy_to_device(num, n_recv);
  m_ptc->x2.copy_to_device(num, n_recv);
  m_ptc->x3.copy_to_device(num, n_recv);
  m_ptc->p1.copy_to_device(num, n_recv);
  m_ptc->p2.copy_to_device(num, n_recv);
  m_ptc->p3.copy_to_device(num, n_recv);
  m_ptc->E.copy_to_device(num, n_recv);
  m_ptc->weight.copy_to_device(num, n_recv);
  m_ptc->cell.copy_to_device(num, n_recv);
  m_ptc->flag.copy_to_device(num, n_recv);
  m_ptc->id.copy_to_device(num, n_recv);
#endif
  m_ptc->add_num(n_recv);
}

template <typename ExecPolicy>
int prismatic_ptc_updater<ExecPolicy>::add_particle(
    Scalar x, Scalar y, Scalar z, Scalar px, Scalar py, Scalar pz,
    Scalar weight, uint32_t flag) {
  auto lmp = m_lmesh.host_ptrs();
  int tri_idx, layer_idx;
  Scalar l1, l2, zeta;
  if (!cartesian_to_local_impl(lmp, x, y, z, tri_idx, layer_idx, l1, l2,
                               zeta))
    return -1;
  // Only the owner of the target cell creates the particle (callers add
  // globally; each rank keeps its own).  Single-rank owns everything.
  if (!lmp.owns_cell(tri_idx, layer_idx)) return -1;

  size_t idx = m_ptc->number();
  if (idx >= m_ptc->size()) return -1;

  auto ptrs = m_ptc->get_host_ptrs();
  ptrs.x1[idx] = l1; ptrs.x2[idx] = l2; ptrs.x3[idx] = zeta;
  ptrs.p1[idx] = px; ptrs.p2[idx] = py; ptrs.p3[idx] = pz;
  ptrs.E[idx] = std::sqrt(Scalar(1) + px*px + py*py + pz*pz);
  ptrs.weight[idx] = weight;
  ptrs.cell[idx] = uint32_t(layer_idx * lmp.N_tri + tri_idx);
  ptrs.flag[idx] = flag;
  ptrs.id[idx] = idx;
  m_ptc->set_num(idx + 1);
  return static_cast<int>(idx);
}

}  // namespace Aperture
