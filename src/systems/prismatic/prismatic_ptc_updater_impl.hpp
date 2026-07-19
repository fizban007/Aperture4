#pragma once

#include "systems/prismatic/prismatic_ptc_updater.h"
#include "systems/prismatic/prismatic_ptc_update_kernel.hpp"
#include "systems/prismatic/prismatic_field_replicator.h"
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
    // Kernel-facing GLOBAL replicas (E/B maintained by the replicator,
    // deposits reduced by this system at the end of each update).
    m_E = sim_env().template register_data<prismatic_edge_field>(
        "E_ptc", m_mesh, mem);
    m_B = sim_env().template register_data<prismatic_face_field>(
        "B_ptc", m_mesh, mem);
    m_J = sim_env().template register_data<prismatic_edge_field>(
        "J_ptc", m_mesh, mem);
    m_rho = sim_env().template register_data<prismatic_vertex_field>(
        "rho_ptc", m_mesh, mem);
    m_rho_abs = sim_env().template register_data<prismatic_vertex_field>(
        "rho_abs_ptc", m_mesh, mem);
    m_gamma_wsum = sim_env().template register_data<prismatic_vertex_field>(
        "gamma_wsum_ptc", m_mesh, mem);
    m_J->set_edge_kind(EdgeCochainKind::dual_2);
    m_J->skip_output(true);
    m_rho->skip_output(true);
    m_rho_abs->skip_output(true);
    m_gamma_wsum->skip_output(true);
    // Solver-facing LOCAL fields (idempotent with the solver's / the
    // replicator's registrations, so system order does not matter).
    m_J_loc = sim_env().template register_data<prismatic_edge_field>(
        "J", *m_mp, mem);
    m_J_loc->set_edge_kind(EdgeCochainKind::dual_2);
    m_rho_loc = sim_env().template register_data<prismatic_vertex_field>(
        "rho", *m_mp, mem);
    m_rho_abs_loc = sim_env().template register_data<prismatic_vertex_field>(
        "rho_abs", *m_mp, mem);
    m_gw_loc = sim_env().template register_data<prismatic_vertex_field>(
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
  sim_env().params().get_value("use_recovery_gather", m_use_recovery_gather);
  sim_env().params().get_value("ptc_absorb_radius", m_absorb_radius);
  sim_env().params().get_value("deposit_diagnostics", m_deposit_diagnostics);
  if (m_absorb_radius > Scalar(0)) {
    Logger::print_info("Particle absorption radius: {}", m_absorb_radius);
  }
  if (m_use_recovery_gather) {
    m_recovery.build(m_mesh);
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    m_recovery.copy_to_device();
#endif
  }

  if (m_distributed) {
    // The replicator owns the field-sync maps and refreshes E_ptc/B_ptc
    // at the start of every step; it must be registered (first).
    auto rep = sim_env().get_system("prismatic_field_replicator");
    if (rep == nullptr) {
      Logger::print_err(
          "prismatic_ptc_updater (distributed) requires "
          "prismatic_field_replicator registered before all particle "
          "systems");
      std::abort();
    }
    m_sync = &dynamic_cast<prismatic_field_replicator<ExecPolicy>&>(*rep)
                  .sync();

    // Cell ownership: owned tris are the rank's contiguous ico-face
    // range (tri indices are grouped per ico-face, 4^L each); owned
    // layers are the radial slab.  The slab map (base/rem) mirrors
    // prismatic_partition::radial_slab.
    if (m_comm->canonical_rank_order()) {
      Logger::print_err(
          "Phase-6 particle systems support only the legacy 20xK "
          "identity comm (create(world, K)); the generalized A*K comm "
          "lands for particles in Phase 7C");
      std::abort();
    }
    const int tris_per_face = m_mesh.m_N_tri / 20;
    m_tri_lo = m_comm->angular_rank() * tris_per_face;
    m_tri_hi = m_tri_lo + tris_per_face;
    const int K = m_comm->n_radial_ranks();
    m_slab_base = m_mesh.m_N_r / K;
    m_slab_rem = m_mesh.m_N_r - m_slab_base * K;
    auto slab_lo = [&](int r) {
      return r * m_slab_base + (r < m_slab_rem ? r : m_slab_rem);
    };
    m_layer_lo = slab_lo(m_comm->radial_rank());
    m_layer_hi = slab_lo(m_comm->radial_rank() + 1);
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
        "Distributed particle updater: rank {} owns tris [{}, {}), layers "
        "[{}, {})",
        m_world_rank, m_tri_lo, m_tri_hi, m_layer_lo, m_layer_hi);
  }

  Logger::print_info("Prismatic particle updater initialized: {} particles",
                     m_ptc->size());
}

template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::update(double dt, uint32_t step) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  int N_tri = mp.N_tri;
  size_t num = m_ptc->number();
  Scalar charge_e = m_charge_e;
  Scalar mass_e = m_mass_e;

  // Clear rho, J, and the diagnostic deposits before deposit
  ExecPolicy::launch(
      [N_edges = mp.N_edges, N_verts = mp.N_verts]
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

  // Refresh the recovery vertex field from the current B cochain (one
  // fitted B vector per mesh vertex; see prismatic_vertex_recovery.h).
  const Scalar* Bv_rec = nullptr;
  if (m_use_recovery_gather) {
    auto rp = m_recovery.get_ptrs(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [mp, rp] LAMBDA(auto B_f) {
          ExecPolicy::loop(0, rp.N_verts, [&] LAMBDA(int vi) {
            rp.compute_vertex_B(mp, B_f, vi);
          });
        },
        m_B->data());
    ExecPolicy::sync();
    Bv_rec = rp.Bv;
  }

  // Particle update loop
  bool use_gca = m_use_gca;
  bool include_curvature = m_include_curvature;
  Scalar absorb_r = m_absorb_radius;
  bool dep_diag = m_deposit_diagnostics;
  Scalar gca_wc = m_gca_switch_wc;
  bool gca_zero_mu = m_gca_zero_mu;
  ExecPolicy::launch(
      [num, N_tri, charge_e, mass_e, dt, mp, use_gca, include_curvature,
       Bv_rec, absorb_r, dep_diag, gca_wc, gca_zero_mu]
      LAMBDA(auto ptc, auto E_e, auto B_f, auto J_e, auto rho,
             auto rho_abs, auto gamma_wsum) {
        ExecPolicy::loop(0, (int)num, [&] LAMBDA(int n) {
          if (ptc.cell[n] == empty_cell) return;
          int sp = get_ptc_type(ptc.flag[n]);
          Scalar q = (sp == (int)PtcType::positron) ? -charge_e : charge_e;
          update_single_particle(mp, N_tri, ptc, n, E_e, B_f, J_e, rho,
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
    // Sum the deposit replicas across ranks (each global slot gets
    // contributions only from ranks whose particles touched it) and
    // pull this rank's owned+ghost slots into the local fields the
    // solver / sph output consume.  The replicas end globally summed
    // on every rank, which is what the injectors read next step.
    m_sync->reduce_edge(m_J->data(), m_J_loc->data());
    m_sync->reduce_vertex(m_rho->data(), m_rho_loc->data());
    if (m_deposit_diagnostics) {
      m_sync->reduce_vertex(m_rho_abs->data(), m_rho_abs_loc->data());
      m_sync->reduce_vertex(m_gamma_wsum->data(), m_gw_loc->data());
    }

    migrate();
  }

  // Periodically sort particles by cell for GPU cache efficiency
  if (m_sort_interval > 0 && step % m_sort_interval == 0) {
    size_t max_cell = m_mesh.m_N_tri * m_mesh.m_N_r;
    ptc_sort_by_cell(typename ExecPolicy::exec_tag{}, *m_ptc, max_cell);
  }
}

// =========================================================================
// Migration: particles whose cell left the owned (ico-face x slab)
// region move to the owning rank.  Device-side pack (two kernels:
// count, then place at per-destination cursors), host-staged
// MPI_Alltoallv per component, arrivals appended at the end of the
// particle array.  Sent particles become empty slots, compacted by the
// periodic sort.
// =========================================================================
template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::migrate() {
  const size_t num = m_ptc->number();
  const int N_tri = m_mesh.m_N_tri;
  const int tris_per_face = N_tri / 20;
  const int base = m_slab_base, rem = m_slab_rem;
  const int me = m_world_rank, ws = m_world_size;

  // Pass 1: count leavers per destination.
  m_mig_count.assign(0);
  ExecPolicy::launch(
      [num, N_tri, tris_per_face, base, rem, me]
      LAMBDA(auto ptc, auto count) {
        ExecPolicy::loop(0, (int)num, [&] LAMBDA(int n) {
          if (ptc.cell[n] == empty_cell) return;
          int dest = prism_migrate_dest(ptc.cell[n], N_tri, tris_per_face,
                                        base, rem, me);
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
        [num, N_tri, tris_per_face, base, rem, me]
        LAMBDA(auto ptc, auto cursor, auto sx1, auto sx2,
               auto sx3, auto sp1, auto sp2, auto sp3,
               auto sE, auto sw, auto scell, auto sflag,
               auto sid) {
          ExecPolicy::loop(0, (int)num, [&] LAMBDA(int n) {
            if (ptc.cell[n] == empty_cell) return;
            int dest = prism_migrate_dest(ptc.cell[n], N_tri, tris_per_face,
                                          base, rem, me);
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
            scell[slot] = ptc.cell[n];
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

  // Append arrivals at the end of the particle array.
  if (num + n_recv > m_ptc->size()) {
    Logger::print_err(
        "prismatic_ptc_updater::migrate: particle buffer overflow "
        "({} + {} arrivals > {})",
        num, n_recv, m_ptc->size());
    std::abort();
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
  std::copy(m_rcv_cell.begin(), m_rcv_cell.end(), hp.cell + num);
  std::copy(m_rcv_flag.begin(), m_rcv_flag.end(), hp.flag + num);
  std::copy(m_rcv_id.begin(), m_rcv_id.end(), hp.id + num);
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
  auto mp = m_mesh.host_ptrs();
  int tri_idx, layer_idx;
  Scalar l1, l2, zeta;
  if (!cartesian_to_local_impl(mp, x, y, z, tri_idx, layer_idx, l1, l2, zeta))
    return -1;
  // Distributed: only the owner of the target cell creates the particle
  // (callers add globally; each rank keeps its own).
  if (m_distributed &&
      (tri_idx < m_tri_lo || tri_idx >= m_tri_hi ||
       layer_idx < m_layer_lo || layer_idx >= m_layer_hi))
    return -1;

  size_t idx = m_ptc->number();
  if (idx >= m_ptc->size()) return -1;

  auto ptrs = m_ptc->get_host_ptrs();
  ptrs.x1[idx] = l1; ptrs.x2[idx] = l2; ptrs.x3[idx] = zeta;
  ptrs.p1[idx] = px; ptrs.p2[idx] = py; ptrs.p3[idx] = pz;
  ptrs.E[idx] = std::sqrt(Scalar(1) + px*px + py*py + pz*pz);
  ptrs.weight[idx] = weight;
  ptrs.cell[idx] = prism_cell_encode(tri_idx, layer_idx, m_mesh.m_N_tri);
  ptrs.flag[idx] = flag;
  ptrs.id[idx] = idx;
  m_ptc->set_num(idx + 1);
  return static_cast<int>(idx);
}

}  // namespace Aperture
