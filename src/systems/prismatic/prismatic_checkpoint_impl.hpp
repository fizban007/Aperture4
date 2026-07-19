#pragma once

#include "systems/prismatic/prismatic_checkpoint.h"
#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/prismatic_data_exporter.h"
#include "systems/prismatic/prismatic_ptc_updater.h"
#include "systems/prismatic/prismatic_sph_output.h"
#include "framework/environment.h"
#include "utils/hdf_wrapper.h"
#include "utils/logger.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <mpi.h>
#include <string>
#include <vector>

namespace Aperture {

namespace {
constexpr const char* ckpt_file_name = "checkpoint.h5";

// get_system returns a nonown_ptr whose get() is const; downcast
// through the mutable operator->.
template <typename S>
S* get_system_as(const std::string& name) {
  auto ptr = sim_env().get_system(name);
  return ptr == nullptr ? nullptr : dynamic_cast<S*>(ptr.operator->());
}
}  // namespace

template <typename ExecPolicy>
prismatic_checkpointer<ExecPolicy>::prismatic_checkpointer(
    const prismatic_mesh& mesh, const prismatic_mesh_partition* mp,
    const prismatic_mpi_comm* comm)
    : m_mesh(mesh), m_mp(mp), m_comm(comm) {
  m_distributed = mp != nullptr && comm != nullptr && !comm->is_single_rank();
}

template <typename ExecPolicy>
void prismatic_checkpointer<ExecPolicy>::register_data_components() {
  // Nothing: every dataset the checkpointer touches is registered by
  // the solver/updater and fetched by name in init().  Re-registering
  // here with different sizing is the known framework mine.
}

template <typename ExecPolicy>
void prismatic_checkpointer<ExecPolicy>::init() {
  sim_env().params().get_value("checkpoint_interval", m_interval);
  sim_env().params().get_value("checkpoint_keep", m_keep);
  std::string output_dir = "Data";
  sim_env().params().get_value("output_dir", output_dir);
  m_dir = output_dir + "/ckpt";
  sim_env().params().get_value("checkpoint_dir", m_dir);
  sim_env().params().get_value("restart_from", m_restart_from);
  if (m_keep < 1) m_keep = 1;

  // Fingerprint values (validated loudly on load).  Radii come from the
  // mesh itself so the comparison is independent of config defaults.
  m_r_min = double(m_mesh.radii[0]);
  m_r_max = double(m_mesh.radii[m_mesh.m_N_r]);
  m_dt = sim_env().params().get_as<double>("dt", 0.01);

  sim_env().get_data("Edelta", m_Edelta);
  sim_env().get_data("Bdelta", m_Bdelta);
  sim_env().get_data_optional("J", m_J);
  sim_env().get_data_optional("rho", m_rho);
  sim_env().get_data_optional("rho_abs", m_rho_abs);
  sim_env().get_data_optional("gamma_wsum", m_gamma_wsum);
  sim_env().get_data_optional("particles", m_ptc);
  if (m_ptc != nullptr) {
    sim_env().get_data_optional("rng_states", m_rng);
    m_updater = get_system_as<prismatic_ptc_updater<ExecPolicy>>(
        "prismatic_ptc_updater");
    if (m_updater == nullptr) {
      Logger::print_err(
          "prismatic_checkpointer: 'particles' data exists but no "
          "prismatic_ptc_updater<ExecPolicy> system found (policy "
          "mismatch?)");
      std::abort();
    }
  }
  m_solver = get_system_as<dec_field_solver<ExecPolicy>>("dec_field_solver");
  if (m_solver == nullptr) {
    Logger::print_err(
        "prismatic_checkpointer: no dec_field_solver<ExecPolicy> system "
        "registered");
    std::abort();
  }

  if (m_distributed) {
    m_world_rank = m_comm->world_rank();
    m_world_size = m_comm->world_size();
    // Owned runs for the global cochain datasets (exporter pattern).
    auto const& L_he = m_mp->layout(cochain_type::h_edge);
    auto const& L_ve = m_mp->layout(cochain_type::v_edge);
    auto const& L_tri = m_mp->layout(cochain_type::tri_face);
    auto const& L_rect = m_mp->layout(cochain_type::rect_face);
    const size_t n_h_glob = size_t(m_mesh.m_N_r + 1) * m_mesh.m_N_edge_s;
    const size_t n_tri_glob = size_t(m_mesh.m_N_r + 1) * m_mesh.m_N_tri;
    append_owned_runs(L_he, 0, 0, m_E_runs);
    append_owned_runs(L_ve, L_he.local_size(), n_h_glob, m_E_runs);
    append_owned_runs(L_tri, 0, 0, m_B_runs);
    append_owned_runs(L_rect, L_tri.local_size(), n_tri_glob, m_B_runs);
    append_owned_runs(m_mp->layout(cochain_type::vertex), 0, 0, m_V_runs);
  }

  // SIGUSR1 graceful stop: the run loop calls this with the
  // post-increment (step, time) — exactly the resume point.
  sim_env().register_force_snapshot(
      [this](uint32_t step, double time) { write_checkpoint(step, time); });

  if (m_interval > 0) {
    Logger::print_info(
        "Checkpointing every {} steps to {} (keep {} generations)",
        m_interval, m_dir, m_keep);
  }
  m_time = 0.0;
}

template <typename ExecPolicy>
void prismatic_checkpointer<ExecPolicy>::update(double dt, uint32_t step) {
  m_time += dt;
  if (m_interval > 0 && step > 0 && step % m_interval == 0) {
    // Registered LAST: the state right now is the start-of-step-(s+1)
    // state; m_time = (step + 1) * dt.
    write_checkpoint(step + 1, m_time);
  }
}

// ===========================================================================
// Writer (plan steps 1 + 3).
// ===========================================================================
template <typename ExecPolicy>
void prismatic_checkpointer<ExecPolicy>::write_checkpoint(
    uint32_t resume_step, double resume_time) {
  namespace fs = std::filesystem;
  const auto t0 = std::chrono::steady_clock::now();
  int phys_rank = 0;
  if (m_distributed) MPI_Comm_rank(MPI_COMM_WORLD, &phys_rank);

  const std::string tmp_dir = m_dir + "/tmp";
  if (phys_rank == 0) {
    std::error_code ec;
    fs::remove_all(tmp_dir, ec);
    fs::create_directories(tmp_dir);
  }
  if (m_distributed) MPI_Barrier(MPI_COMM_WORLD);

  auto file = hdf_create(tmp_dir + "/" + ckpt_file_name,
                         m_distributed ? H5CreateMode::trunc_parallel
                                       : H5CreateMode::trunc);

  // ---- Cochains: owned-runs collective writes of the global datasets.
  auto write_edge = [&](nonown_ptr<prismatic_edge_field>& f,
                        const char* name) {
    if (f == nullptr) return;
    f->data().copy_to_host();
    if (m_distributed) {
      file.write_parallel_runs(f->host_ptr(), f->data().size(),
                               size_t(m_mesh.m_N_edges), m_E_runs.mem_off,
                               m_E_runs.file_off, m_E_runs.len, name);
    } else {
      file.write(f->host_ptr(), size_t(m_mesh.m_N_edges), name);
    }
  };
  auto write_vert = [&](nonown_ptr<prismatic_vertex_field>& f,
                        const char* name) {
    if (f == nullptr) return;
    f->data().copy_to_host();
    if (m_distributed) {
      file.write_parallel_runs(f->host_ptr(), f->data().size(),
                               size_t(m_mesh.m_N_verts), m_V_runs.mem_off,
                               m_V_runs.file_off, m_V_runs.len, name);
    } else {
      file.write(f->host_ptr(), size_t(m_mesh.m_N_verts), name);
    }
  };
  write_edge(m_Edelta, "Edelta_e");
  m_Bdelta->data().copy_to_host();
  if (m_distributed) {
    file.write_parallel_runs(m_Bdelta->host_ptr(), m_Bdelta->data().size(),
                             size_t(m_mesh.m_N_faces), m_B_runs.mem_off,
                             m_B_runs.file_off, m_B_runs.len, "Bdelta_f");
  } else {
    file.write(m_Bdelta->host_ptr(), size_t(m_mesh.m_N_faces), "Bdelta_f");
  }
  // Not needed by the solver (recomputed every step), but the injector
  // criteria read LAST step's J / rho_abs — storing them makes the
  // first resumed step faithful (plan D2).
  write_edge(m_J, "J_e");
  write_vert(m_rho, "rho");
  write_vert(m_rho_abs, "rho_abs");
  write_vert(m_gamma_wsum, "gamma_wsum");

  // ---- Particles: live macros only, one concatenated global dataset,
  // GLOBAL cells widened to uint64 on the wire (plan D2/D3).
  uint64_t n_live = 0, ptc_total = 0;
  if (m_ptc != nullptr && m_updater != nullptr) {
    m_ptc->copy_to_host();
    const auto hp = m_ptc->get_host_ptrs();
    const auto& lmesh = m_updater->ptc_mesh();
    const auto& l2g = lmesh.tri_l2g_host();
    const int n_tri_loc = lmesh.n_tri_local();
    const int N_tri_glob = lmesh.n_tri_global();
    const int k0 = lmesh.k0();
    const size_t num = m_ptc->number();

    std::vector<Scalar> comps[8];
    std::vector<uint64_t> cells, ids;
    std::vector<uint32_t> flags;
    for (size_t n = 0; n < num; ++n) {
      if (hp.cell[n] == empty_cell) continue;
      comps[0].push_back(hp.x1[n]);
      comps[1].push_back(hp.x2[n]);
      comps[2].push_back(hp.x3[n]);
      comps[3].push_back(hp.p1[n]);
      comps[4].push_back(hp.p2[n]);
      comps[5].push_back(hp.p3[n]);
      comps[6].push_back(hp.E[n]);
      comps[7].push_back(hp.weight[n]);
      const int lay = int(hp.cell[n]) / n_tri_loc;
      const int tri = int(hp.cell[n]) - lay * n_tri_loc;
      cells.push_back(uint64_t(k0 + lay) * uint64_t(N_tri_glob) +
                      uint64_t(l2g[tri]));
      flags.push_back(hp.flag[n]);
      ids.push_back(hp.id[n]);
    }
    n_live = cells.size();

    uint64_t offset = 0;
    ptc_total = n_live;
    if (m_distributed) {
      MPI_Exscan(&n_live, &offset, 1, MPI_UINT64_T, MPI_SUM,
                 m_comm->world());
      if (m_world_rank == 0) offset = 0;
      MPI_Allreduce(&n_live, &ptc_total, 1, MPI_UINT64_T, MPI_SUM,
                    m_comm->world());
    }

    static const char* comp_names[8] = {"ptc_x1", "ptc_x2", "ptc_x3",
                                        "ptc_p1", "ptc_p2", "ptc_p3",
                                        "ptc_E",  "ptc_weight"};
    if (m_distributed) {
      for (int c = 0; c < 8; ++c) {
        file.write_parallel(comps[c].data(), n_live, ptc_total, offset,
                            n_live, 0, comp_names[c]);
      }
      file.write_parallel(cells.data(), n_live, ptc_total, offset, n_live,
                          0, "ptc_cell");
      file.write_parallel(flags.data(), n_live, ptc_total, offset, n_live,
                          0, "ptc_flag");
      file.write_parallel(ids.data(), n_live, ptc_total, offset, n_live, 0,
                          "ptc_id");
    } else {
      for (int c = 0; c < 8; ++c) {
        file.write(comps[c].data(), n_live, comp_names[c]);
      }
      file.write(cells.data(), n_live, "ptc_cell");
      file.write(flags.data(), n_live, "ptc_flag");
      file.write(ids.data(), n_live, "ptc_id");
    }

    // Per-WRITING-rank id counters (plan D4).
    auto& idc = m_ptc->ptc_id();
    idc.copy_to_host();
    const uint32_t id_ctr = idc[0];
    if (m_distributed) {
      file.write_parallel(&id_ctr, 1, size_t(m_world_size),
                          size_t(m_world_rank), 1, 0, "ptc_id_counter");
    } else {
      file.write(&id_ctr, 1, "ptc_id_counter");
    }

    // Per-WRITING-rank rng streams (plan D4): exact same-count restart.
    if (m_rng != nullptr) {
      m_rng->copy_to_host();
      const size_t n_states = m_rng->size();
      const uint64_t* raw =
          reinterpret_cast<const uint64_t*>(m_rng->states().host_ptr());
      const size_t n_u64 = n_states * 4;
      if (m_distributed) {
        file.write_parallel(raw, n_u64, n_u64 * m_world_size,
                            n_u64 * m_world_rank, n_u64, 0, "rng_states");
      } else {
        file.write(raw, n_u64, "rng_states");
      }
      file.write(int(n_states), "rng_n_states");
    }
  }
  file.write(ptc_total, "ptc_total");

  // ---- Resume point + config fingerprint (plan D2).
  file.write(resume_step, "resume_step");
  file.write(resume_time, "time");
  file.write(m_mesh.m_L, "L");
  file.write(m_mesh.m_N_r, "N_r");
  file.write(m_mesh.m_N_tri, "N_tri");
  file.write(m_r_min, "r_min");
  file.write(m_r_max, "r_max");
  file.write(m_dt, "dt");
  file.write(m_comm != nullptr ? m_comm->n_angular_ranks() : 1, "writer_A");
  file.write(m_comm != nullptr ? m_comm->n_radial_ranks() : 1, "writer_K");
  file.write(m_world_size, "writer_world");
  if (m_J != nullptr) {
    file.write(m_J->edge_kind() == EdgeCochainKind::dual_2 ? 1 : 0,
               "J_kind_dual2");
  }
  // Written LAST: a generation without it is incomplete (used by the
  // restart_from = auto scan; the atomic rename below is the primary
  // crash barrier).
  file.write(1, "complete");
  file.close();

  if (m_distributed) MPI_Barrier(MPI_COMM_WORLD);
  if (phys_rank == 0) rotate_generations(resume_step);
  if (m_distributed) MPI_Barrier(MPI_COMM_WORLD);

  const double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
  if (phys_rank == 0) {
    Logger::print_info(
        "Checkpoint written: resume step {}, time {:.4f}, {} macros, "
        "{:.1f} s",
        resume_step, resume_time, ptc_total, secs);
  }
}

// Rank 0 only: atomically promote tmp/ to ckpt_<step>/ and prune old
// generations beyond checkpoint_keep (plan D5).
template <typename ExecPolicy>
void prismatic_checkpointer<ExecPolicy>::rotate_generations(
    uint32_t resume_step) {
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path gen_dir = fs::path(m_dir) / ("ckpt_" +
                                              std::to_string(resume_step));
  fs::remove_all(gen_dir, ec);
  fs::rename(fs::path(m_dir) / "tmp", gen_dir, ec);
  if (ec) {
    Logger::print_err("Checkpoint rotation: rename to {} failed: {}",
                      gen_dir.string(), ec.message());
    return;
  }

  // Prune: keep the newest m_keep generations by step number.
  std::vector<std::pair<long, fs::path>> gens;
  for (auto const& e : fs::directory_iterator(m_dir, ec)) {
    const std::string base = e.path().filename().string();
    if (base.rfind("ckpt_", 0) != 0) continue;
    char* end = nullptr;
    const long step = std::strtol(base.c_str() + 5, &end, 10);
    if (end == nullptr || *end != '\0') continue;
    gens.emplace_back(step, e.path());
  }
  std::sort(gens.begin(), gens.end(),
            [](auto const& a, auto const& b) { return a.first > b.first; });
  for (size_t i = m_keep; i < gens.size(); ++i) {
    fs::remove_all(gens[i].second, ec);
    Logger::print_info("Checkpoint rotation: removed old generation {}",
                       gens[i].second.string());
  }
}

// ===========================================================================
// Reader (plan step 2): fingerprint validation, owned-runs field reads,
// chunked particle read routed through the migration machinery.
// ===========================================================================
template <typename ExecPolicy>
bool prismatic_checkpointer<ExecPolicy>::try_restart() {
  std::string src = m_restart_from;
  if (sim_env().is_restart()) src = sim_env().restart_file();
  if (src.empty()) return false;

  if (src == "auto") {
    src = find_latest_generation();
    if (src.empty()) {
      Logger::print_info(
          "restart_from = auto: no complete generation under {} — fresh "
          "start",
          m_dir);
      return false;
    }
  }
  // Accept either the generation directory or the file itself.
  if (src.size() < 3 || src.substr(src.size() - 3) != ".h5") {
    src += std::string("/") + ckpt_file_name;
  }
  load_generation(src);
  sim_env().finish_restart();
  return true;
}

template <typename ExecPolicy>
std::string prismatic_checkpointer<ExecPolicy>::find_latest_generation()
    const {
  namespace fs = std::filesystem;
  int phys_rank = 0;
  if (m_distributed) MPI_Comm_rank(MPI_COMM_WORLD, &phys_rank);

  std::string best;
  if (phys_rank == 0) {
    long best_step = -1;
    std::error_code ec;
    for (auto const& e : fs::directory_iterator(m_dir, ec)) {
      const std::string base = e.path().filename().string();
      if (base.rfind("ckpt_", 0) != 0) continue;
      char* end = nullptr;
      const long step = std::strtol(base.c_str() + 5, &end, 10);
      if (end == nullptr || *end != '\0' || step <= best_step) continue;
      const fs::path f = e.path() / ckpt_file_name;
      if (!fs::exists(f, ec)) continue;
      // Serial single-rank open just to verify the complete marker.
      H5File file(f.string(), H5OpenMode::read_only);
      const bool complete = file.exists("complete");
      file.close();
      if (!complete) {
        Logger::print_info("Skipping incomplete generation {}",
                           e.path().string());
        continue;
      }
      best_step = step;
      best = e.path().string();
    }
  }
  if (m_distributed) {
    int len = int(best.size());
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    best.resize(len);
    if (len > 0) MPI_Bcast(&best[0], len, MPI_CHAR, 0, MPI_COMM_WORLD);
  }
  return best;
}

template <typename ExecPolicy>
void prismatic_checkpointer<ExecPolicy>::load_generation(
    const std::string& path) {
  const auto t0 = std::chrono::steady_clock::now();
  Logger::print_info("Restarting from {}", path);

  H5File file(path, m_distributed ? H5OpenMode::read_parallel
                                  : H5OpenMode::read_only);

  // ---- Fingerprint (plan D2): loud validation.
  auto check_int = [&](const char* name, int expect) {
    const int got = file.read_scalar<int>(name);
    if (got != expect) {
      Logger::print_err(
          "Checkpoint fingerprint mismatch: {} = {} in checkpoint, {} in "
          "this run",
          name, got, expect);
      std::abort();
    }
  };
  auto check_dbl = [&](const char* name, double expect) {
    const double got = file.read_scalar<double>(name);
    if (got != expect) {
      Logger::print_err(
          "Checkpoint fingerprint mismatch: {} = {} in checkpoint, {} in "
          "this run",
          name, got, expect);
      std::abort();
    }
  };
  check_int("L", m_mesh.m_L);
  check_int("N_r", m_mesh.m_N_r);
  check_int("N_tri", m_mesh.m_N_tri);
  check_dbl("r_min", m_r_min);
  check_dbl("r_max", m_r_max);
  check_dbl("dt", m_dt);

  const int writer_A = file.read_scalar<int>("writer_A");
  const int writer_K = file.read_scalar<int>("writer_K");
  const int writer_world = file.read_scalar<int>("writer_world");
  const bool same_decomp =
      writer_world == m_world_size &&
      writer_A == (m_comm != nullptr ? m_comm->n_angular_ranks() : 1) &&
      writer_K == (m_comm != nullptr ? m_comm->n_radial_ranks() : 1);
  Logger::print_info(
      "Checkpoint written at A = {}, K = {}, world = {}; this run: A = {}, "
      "K = {}, world = {} ({})",
      writer_A, writer_K, writer_world,
      m_comm != nullptr ? m_comm->n_angular_ranks() : 1,
      m_comm != nullptr ? m_comm->n_radial_ranks() : 1, m_world_size,
      same_decomp ? "same decomposition" : "REDISTRIBUTING");

  const uint32_t resume_step = file.read_scalar<uint32_t>("resume_step");
  const double resume_time = file.read_scalar<double>("time");

  // ---- Cochains: owned slots (mirror of the owned-runs writes).
  auto read_edge = [&](nonown_ptr<prismatic_edge_field>& f,
                       const char* name) {
    if (f == nullptr || !file.exists(name)) return;
    if (m_distributed) {
      file.read_parallel_runs(f->host_ptr(), f->data().size(),
                              m_E_runs.mem_off, m_E_runs.file_off,
                              m_E_runs.len, name);
    } else {
      file.read_array(f->host_ptr(), f->data().size(), name);
    }
    f->data().copy_to_device();
  };
  auto read_vert = [&](nonown_ptr<prismatic_vertex_field>& f,
                       const char* name) {
    if (f == nullptr || !file.exists(name)) return;
    if (m_distributed) {
      file.read_parallel_runs(f->host_ptr(), f->data().size(),
                              m_V_runs.mem_off, m_V_runs.file_off,
                              m_V_runs.len, name);
    } else {
      file.read_array(f->host_ptr(), f->data().size(), name);
    }
    f->data().copy_to_device();
  };
  read_edge(m_Edelta, "Edelta_e");
  if (file.exists("Bdelta_f")) {
    if (m_distributed) {
      file.read_parallel_runs(m_Bdelta->host_ptr(), m_Bdelta->data().size(),
                              m_B_runs.mem_off, m_B_runs.file_off,
                              m_B_runs.len, "Bdelta_f");
    } else {
      file.read_array(m_Bdelta->host_ptr(), m_Bdelta->data().size(),
                      "Bdelta_f");
    }
    m_Bdelta->data().copy_to_device();
  }
  read_edge(m_J, "J_e");
  read_vert(m_rho, "rho");
  read_vert(m_rho_abs, "rho_abs");
  read_vert(m_gamma_wsum, "gamma_wsum");

  // Ghost slots + derived state: delta ghosts, totals (particles and
  // the injector consume totals), deposit ghosts (next-step injector
  // stencils), solver clock.
  m_solver->refresh_delta_ghosts();
  m_solver->refresh_total_fields();
  m_solver->set_time(resume_time);
  if (m_updater != nullptr) m_updater->refresh_deposit_ghosts();

  // ---- Particles: arbitrary contiguous chunk per rank, routed to the
  // owners via the migration machinery (plan D3 — this is what makes
  // different-rank-count restart nearly free).
  const uint64_t ptc_total = file.read_scalar<uint64_t>("ptc_total");
  if (ptc_total > 0) {
    if (m_updater == nullptr || m_ptc == nullptr) {
      Logger::print_err(
          "Checkpoint holds {} macros but this run has no particle "
          "updater — refusing to silently drop them",
          ptc_total);
      std::abort();
    }
    const uint64_t base = ptc_total / uint64_t(m_world_size);
    const uint64_t rem = ptc_total % uint64_t(m_world_size);
    const uint64_t r = uint64_t(m_world_rank);
    const uint64_t lo = r * base + std::min(r, rem);
    const uint64_t n_chunk = base + (r < rem ? 1 : 0);

    std::vector<Scalar> comps[8];
    std::vector<uint64_t> cells(n_chunk), ids(n_chunk);
    std::vector<uint32_t> flags(n_chunk);
    static const char* comp_names[8] = {"ptc_x1", "ptc_x2", "ptc_x3",
                                        "ptc_p1", "ptc_p2", "ptc_p3",
                                        "ptc_E",  "ptc_weight"};
    for (int c = 0; c < 8; ++c) {
      comps[c].resize(n_chunk);
      file.read_subset(comps[c].data(), n_chunk, comp_names[c], lo, n_chunk,
                       0);
    }
    file.read_subset(cells.data(), n_chunk, "ptc_cell", lo, n_chunk, 0);
    file.read_subset(flags.data(), n_chunk, "ptc_flag", lo, n_chunk, 0);
    file.read_subset(ids.data(), n_chunk, "ptc_id", lo, n_chunk, 0);

    const size_t n_before = m_ptc->number();
    m_updater->inject_wire_particles(comps, cells, flags, ids);
    uint64_t n_loaded = m_ptc->number() - n_before;
    if (m_distributed) {
      MPI_Allreduce(MPI_IN_PLACE, &n_loaded, 1, MPI_UINT64_T, MPI_SUM,
                    m_comm->world());
    }
    if (n_loaded != ptc_total) {
      Logger::print_err(
          "Checkpoint restore lost particles: {} stored, {} appended",
          ptc_total, n_loaded);
      std::abort();
    }

    // ---- id counters (plan D4): exact per-rank restore on the same
    // world size; on a different one, continue every rank above the
    // stored global max (ids only feed tracking).
    if (file.exists("ptc_id_counter")) {
      auto& idc = m_ptc->ptc_id();
      if (writer_world == m_world_size) {
        uint32_t v = 0;
        if (m_distributed) {
          file.read_subset(&v, 1, "ptc_id_counter", size_t(m_world_rank), 1,
                           0);
        } else {
          file.read_array(&v, 1, "ptc_id_counter");
        }
        idc[0] = v;
      } else {
        std::vector<uint32_t> all(writer_world);
        file.read_array(all.data(), all.size(), "ptc_id_counter");
        idc[0] = *std::max_element(all.begin(), all.end());
      }
      idc.copy_to_device();
    }

    // ---- rng streams (plan D4): bit-exact on the same world size and
    // state-pool size; otherwise keep the fresh init() seeding (log).
    if (m_rng != nullptr && file.exists("rng_states")) {
      const int stored_n = file.read_scalar<int>("rng_n_states");
      if (writer_world == m_world_size &&
          stored_n == int(m_rng->size())) {
        const size_t n_u64 = size_t(stored_n) * 4;
        uint64_t* raw =
            reinterpret_cast<uint64_t*>(m_rng->states().host_ptr());
        if (m_distributed) {
          file.read_subset(raw, n_u64, "rng_states", n_u64 * m_world_rank,
                           n_u64, 0);
        } else {
          file.read_array(raw, n_u64, "rng_states");
        }
        m_rng->copy_to_device();
        Logger::print_info("rng streams restored exactly (same rank count)");
      } else {
        Logger::print_info(
            "rng streams reseeded (decomposition changed: stored world {} "
            "x {} states, this run {} x {})",
            writer_world, stored_n, m_world_size, m_rng->size());
      }
    }
  }
  file.close();

  // ---- Seed the clocks (plan D2: every m_time integrates += dt).
  sim_env().set_step(resume_step);
  sim_env().set_time(resume_time);
  m_time = resume_time;
  if (auto* exp = get_system_as<prismatic_data_exporter>(
          "prismatic_data_exporter")) {
    exp->set_time(resume_time);
  }
  if (!m_distributed) {
    // sph output is single-rank only since 7D.
    if (auto* sph =
            get_system_as<prismatic_sph_output>("prismatic_sph_output")) {
      sph->set_time(resume_time);
    }
  }

  const double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
  Logger::print_info(
      "Restart complete: resuming at step {}, time {:.4f} ({} macros, "
      "{:.1f} s)",
      resume_step, resume_time, ptc_total, secs);
}

}  // namespace Aperture
