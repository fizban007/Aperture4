#include "systems/prismatic/prismatic_data_exporter.h"
#include "framework/environment.h"
#include "utils/hdf_wrapper.h"
#include "utils/logger.h"
#include <cstdio>
#include <filesystem>
#include <string>

namespace Aperture {

namespace {

// Compress a layout's owned set (locals 0..n_owned in ascending-global
// order) into contiguous runs for scattered hyperslab writes.
void append_owned_runs(const distributed_cochain_layout& L, size_t mem_base,
                       size_t file_base, std::vector<hsize_t>& mem_off,
                       std::vector<hsize_t>& file_off,
                       std::vector<hsize_t>& len) {
  const int n = L.owned_size();
  int l = 0;
  while (l < n) {
    const int g0 = L.to_global(l);
    int run = 1;
    while (l + run < n && L.to_global(l + run) == g0 + run) run++;
    mem_off.push_back(mem_base + l);
    file_off.push_back(file_base + g0);
    len.push_back(run);
    l += run;
  }
}

}  // namespace

prismatic_data_exporter::prismatic_data_exporter(
    const prismatic_mesh& mesh, const prismatic_mesh_partition* mp,
    const prismatic_mpi_comm* comm)
    : m_mesh(mesh), m_mp(mp), m_comm(comm) {
  m_distributed = mp != nullptr && comm != nullptr && !comm->is_single_rank();
}

void prismatic_data_exporter::register_data_components() {
  // Under a distributed solver the solver registered these local-sized;
  // register_data returns the existing components (the solver must be
  // registered first — init() verifies the sizes).
  m_E = sim_env().register_data<prismatic_edge_field>("E", m_mesh);
  m_B = sim_env().register_data<prismatic_face_field>("B", m_mesh);
}

void prismatic_data_exporter::init() {
  // Moment fields exist when a particle updater is registered (init
  // runs after every system's registration).
  sim_env().get_data_optional("J", m_J);
  sim_env().get_data_optional("rho", m_rho);
  sim_env().get_data_optional("rho_abs", m_rho_abs);
  sim_env().get_data_optional("gamma_wsum", m_gamma_wsum);

  sim_env().params().get_value("fld_output_interval", m_output_interval);
  sim_env().params().get_value("fld_output_radial_stride",
                               m_output_radial_stride);
  sim_env().params().get_value("fld_output_angular_stride",
                               m_output_angular_stride);
  sim_env().params().get_value("fld_output_aggregate", m_aggregate);
  sim_env().params().get_value("fld_output_angular_level", m_agg_level);
  sim_env().params().get_value("output_dir", m_output_dir);

  if (m_aggregate) {
    // Chain-map aggregation (F9): coarse level L - j, radial stride R.
    // Throws loudly on an invalid (j, R).
    m_agg.build(m_mesh, m_agg_level,
                m_output_radial_stride < 1 ? 1 : m_output_radial_stride);
    m_output_radial_stride = 1;   // disable the legacy sampling path
    m_output_angular_stride = 1;
    m_agg_E.resize(m_agg.n_h_c() + m_agg.n_v_c());
    m_agg_B.resize(m_agg.n_trif_c() + m_agg.n_rect_c());
    Logger::print_info(
        "Aggregated output: level {} (j={}) x radial stride {} -> "
        "{} coarse edges, {} coarse faces per snapshot",
        m_agg.L_out, m_agg.j, m_agg.R, m_agg_E.size(), m_agg_B.size());
  }

  if (m_output_radial_stride  < 1) m_output_radial_stride  = 1;
  if (m_output_angular_stride < 1) m_output_angular_stride = 1;

  if (m_distributed) {
    if (m_output_radial_stride > 1 || m_output_angular_stride > 1) {
      Logger::print_err(
          "prismatic_data_exporter: output downsampling is not supported "
          "in distributed mode; writing full snapshots");
      m_output_radial_stride = 1;
      m_output_angular_stride = 1;
    }
    // The solver must have registered the fields local-sized (solver
    // before exporter in main); a global-sized "E" here means the
    // registration order is wrong and every write would be garbage.
    auto const& L_he = m_mp->layout(cochain_type::h_edge);
    auto const& L_ve = m_mp->layout(cochain_type::v_edge);
    auto const& L_tri = m_mp->layout(cochain_type::tri_face);
    auto const& L_rect = m_mp->layout(cochain_type::rect_face);
    if (int(m_E->data().size()) != L_he.local_size() + L_ve.local_size() ||
        int(m_B->data().size()) != L_tri.local_size() + L_rect.local_size()) {
      Logger::print_err(
          "prismatic_data_exporter: field sizes are not local-sized; "
          "register the distributed solver BEFORE the exporter");
      std::abort();
    }
    // Owned runs for the combined global datasets.  Block bases: v
    // edges follow all h edges, rect faces follow all tri faces, in
    // both the local buffer (at the field's split()) and the global
    // ordering (at the global block sizes).
    const size_t n_h_glob = size_t(m_mesh.m_N_r + 1) * m_mesh.m_N_edge_s;
    const size_t n_tri_glob = size_t(m_mesh.m_N_r + 1) * m_mesh.m_N_tri;
    append_owned_runs(L_he, 0, 0, m_E_runs.mem_off, m_E_runs.file_off,
                      m_E_runs.len);
    append_owned_runs(L_ve, L_he.local_size(), n_h_glob, m_E_runs.mem_off,
                      m_E_runs.file_off, m_E_runs.len);
    append_owned_runs(L_tri, 0, 0, m_B_runs.mem_off, m_B_runs.file_off,
                      m_B_runs.len);
    append_owned_runs(L_rect, L_tri.local_size(), n_tri_glob,
                      m_B_runs.mem_off, m_B_runs.file_off, m_B_runs.len);
    if (m_rho != nullptr || m_rho_abs != nullptr || m_gamma_wsum != nullptr) {
      append_owned_runs(m_mp->layout(cochain_type::vertex), 0, 0,
                        m_V_runs.mem_off, m_V_runs.file_off, m_V_runs.len);
    }
    Logger::print_info(
        "Distributed exporter: {} + {} owned runs (E, B) per snapshot",
        m_E_runs.len.size(), m_B_runs.len.size());
  }

  // Create output directory
  std::filesystem::create_directories(m_output_dir);

  // Build kept-index arrays for structured downsampling. We iterate
  // over (radial layer, sphere sub-element) and apply each stride
  // independently, so the result is a structured subset of the prism
  // mesh — every R-th shell × every A-th sphere triangle / edge /
  // vertex. This is geometrically meaningful: with (R=2, A=4) the
  // result is exactly one refinement level coarser. The first
  // sub-element (t=0, e=0, s=0) is always retained, so the output
  // includes the inner shell.
  const int R = m_output_radial_stride;
  const int A = m_output_angular_stride;
  const int N_r       = m_mesh.m_N_r;
  const int N_tri     = m_mesh.m_N_tri;
  const int N_edge_s  = m_mesh.m_N_edge_s;
  const int N_vert_s  = m_mesh.m_N_vert_s;

  if ((R > 1 || A > 1) && !m_mesh.has_3d()) {
    Logger::print_err(
        "prismatic_data_exporter: legacy stride downsampling needs the "
        "full mesh build; writing full snapshots");
    m_output_radial_stride = m_output_angular_stride = 1;
  } else if (R > 1 || A > 1) {
    // ---- Faces: triangular shell faces, then rectangular faces ----
    // Triangular shell faces: (k, t) → k*N_tri + t, k ∈ [0, N_r], t ∈ [0, N_tri)
    for (int k = 0; k <= N_r; k += R) {
      for (int t = 0; t < N_tri; t += A) {
        m_out_face_idx.push_back(k * N_tri + t);
      }
    }
    // Rectangular faces: (k, e) → (N_r+1)*N_tri + k*N_edge_s + e,
    // k ∈ [0, N_r), e ∈ [0, N_edge_s)
    const int n_tri_total = (N_r + 1) * N_tri;
    for (int k = 0; k < N_r; k += R) {
      for (int e = 0; e < N_edge_s; e += A) {
        m_out_face_idx.push_back(n_tri_total + k * N_edge_s + e);
      }
    }

    // ---- Edges: horizontal edges, then vertical edges ----
    // Horizontal edges: k*N_edge_s + e
    for (int k = 0; k <= N_r; k += R) {
      for (int e = 0; e < N_edge_s; e += A) {
        m_out_edge_idx.push_back(k * N_edge_s + e);
      }
    }
    // Vertical edges: (N_r+1)*N_edge_s + k*N_vert_s + s
    const int n_h_edges = (N_r + 1) * N_edge_s;
    for (int k = 0; k < N_r; k += R) {
      for (int s = 0; s < N_vert_s; s += A) {
        m_out_edge_idx.push_back(n_h_edges + k * N_vert_s + s);
      }
    }

    m_out_E_buf.resize(m_out_edge_idx.size());
    m_out_B_buf.resize(m_out_face_idx.size());

    // ---- Collect unique vertices used by the kept faces / edges ----
    // Used for writing standalone vertex info to mesh.h5; downstream
    // tools can render the downsampled mesh without loading the full
    // vertex list.
    std::set<int> vert_set;
    for (int fi : m_out_face_idx) {
      if (fi < n_tri_total) {
        vert_set.insert(m_mesh.tri_face_v0[fi]);
        vert_set.insert(m_mesh.tri_face_v1[fi]);
        vert_set.insert(m_mesh.tri_face_v2[fi]);
      } else {
        int ri = fi - n_tri_total;
        vert_set.insert(m_mesh.rect_face_v0[ri]);
        vert_set.insert(m_mesh.rect_face_v1[ri]);
        vert_set.insert(m_mesh.rect_face_v2[ri]);
        vert_set.insert(m_mesh.rect_face_v3[ri]);
      }
    }
    for (int ei : m_out_edge_idx) {
      vert_set.insert(m_mesh.edge_v0[ei]);
      vert_set.insert(m_mesh.edge_v1[ei]);
    }
    m_out_vert_idx.assign(vert_set.begin(), vert_set.end());

    Logger::print_info(
        "Output downsample (radial={}, angular={}): writing "
        "{}/{} edges, {}/{} faces, {}/{} vertices per snapshot",
        R, A,
        static_cast<int>(m_out_edge_idx.size()), m_mesh.m_N_edges,
        static_cast<int>(m_out_face_idx.size()), m_mesh.m_N_faces,
        static_cast<int>(m_out_vert_idx.size()), m_mesh.m_N_verts);
  }

  // Write mesh file once.  Every rank holds the full global mesh
  // (build-then-extract-local, until 4.1a.4), so under MPI world rank 0
  // writes it alone with the ordinary serial path.
  int world_rank = 0;
  if (m_distributed) {
    world_rank = m_comm->world_rank();
  }
  if (world_rank == 0) write_mesh();

  m_time = 0.0;
}

void prismatic_data_exporter::update(double dt, uint32_t step) {
  m_time += dt;
  if (step % m_output_interval == 0) {
    write_snapshot(step, m_time);
  }
}

void prismatic_data_exporter::write_mesh() {
  std::string filename = m_output_dir + "/mesh.h5";
  auto file = hdf_create(filename);

  // Phase 7D: under a sphere-only mesh (distributed runs) no rank holds
  // the 3D per-cochain arrays; they are ANALYTIC in (sphere tables x
  // radii), so the mesh file carries the sphere-level data + radii +
  // parameters only, and post tools (python/sph_from_dump.py) recompute
  // whatever per-element geometry they need via the same formulas
  // (prismatic_mesh_geom.h).
  if (!m_mesh.has_3d()) {
    file.write(m_mesh.radii.host_ptr(), m_mesh.m_N_r + 1, "radii");
    file.write(m_mesh.sphere_vx.host_ptr(), m_mesh.m_N_vert_s, "sphere_vx");
    file.write(m_mesh.sphere_vy.host_ptr(), m_mesh.m_N_vert_s, "sphere_vy");
    file.write(m_mesh.sphere_vz.host_ptr(), m_mesh.m_N_vert_s, "sphere_vz");
    file.write(m_mesh.sphere_theta.host_ptr(), m_mesh.m_N_vert_s,
               "sphere_theta");
    file.write(m_mesh.sphere_phi.host_ptr(), m_mesh.m_N_vert_s,
               "sphere_phi");
    file.write(m_mesh.tri_verts.host_ptr(), m_mesh.m_N_tri * 3, "tri_verts");
    file.write(m_mesh.tri_edges_s.host_ptr(), m_mesh.m_N_tri * 3,
               "tri_edges_s");
    file.write(m_mesh.tri_edge_signs.host_ptr(), m_mesh.m_N_tri * 3,
               "tri_edge_signs");
    file.write(m_mesh.tri_neighbor.host_ptr(), m_mesh.m_N_tri * 3,
               "tri_neighbor");
    file.write(m_mesh.sphere_edge_v0.host_ptr(), m_mesh.m_N_edge_s,
               "sphere_edge_v0");
    file.write(m_mesh.sphere_edge_v1.host_ptr(), m_mesh.m_N_edge_s,
               "sphere_edge_v1");
    file.write(m_mesh.m_L, "L");
    file.write(m_mesh.m_N_r, "N_r");
    file.write(m_mesh.m_N_verts, "N_verts");
    file.write(m_mesh.m_N_edges, "N_edges");
    file.write(m_mesh.m_N_faces, "N_faces");
    file.write(m_mesh.m_N_tri, "N_tri");
    file.write(m_mesh.m_N_vert_s, "N_vert_s");
    file.write(m_mesh.m_N_edge_s, "N_edge_s");
    file.write(1, "sphere_only");
    file.close();
    Logger::print_info("Mesh written (sphere-level) to {}", filename);
    return;
  }

  // Write vertex positions.  Mesh stores (r, θ, φ); derive Cartesian
  // arrays locally for backwards compat with analysis/viz scripts that
  // read "vert_x/y/z", and also write the spherical coords directly.
  std::vector<Scalar> vert_x_cart(m_mesh.m_N_verts);
  std::vector<Scalar> vert_y_cart(m_mesh.m_N_verts);
  std::vector<Scalar> vert_z_cart(m_mesh.m_N_verts);
  for (int vi = 0; vi < m_mesh.m_N_verts; ++vi) {
    Scalar r = m_mesh.vert_r[vi];
    Scalar th = m_mesh.vert_theta[vi];
    Scalar ph = m_mesh.vert_phi[vi];
    Scalar sth = std::sin(th);
    vert_x_cart[vi] = r * sth * std::cos(ph);
    vert_y_cart[vi] = r * sth * std::sin(ph);
    vert_z_cart[vi] = r * std::cos(th);
  }
  file.write(vert_x_cart.data(), m_mesh.m_N_verts, "vert_x");
  file.write(vert_y_cart.data(), m_mesh.m_N_verts, "vert_y");
  file.write(vert_z_cart.data(), m_mesh.m_N_verts, "vert_z");
  file.write(m_mesh.vert_r.host_ptr(), m_mesh.m_N_verts, "vert_r");
  file.write(m_mesh.vert_theta.host_ptr(), m_mesh.m_N_verts, "vert_theta");
  file.write(m_mesh.vert_phi.host_ptr(), m_mesh.m_N_verts, "vert_phi");

  // Write edge endpoints
  file.write(m_mesh.edge_v0.host_ptr(), m_mesh.m_N_edges, "edge_v0");
  file.write(m_mesh.edge_v1.host_ptr(), m_mesh.m_N_edges, "edge_v1");
  file.write(m_mesh.edge_length.host_ptr(), m_mesh.m_N_edges, "edge_length");

  // Write face areas
  file.write(m_mesh.face_area.host_ptr(), m_mesh.m_N_faces, "face_area");

  // Write face vertex indices for visualization
  int n_tri_faces = m_mesh.m_N_tri * (m_mesh.m_N_r + 1);
  int n_rect_faces = m_mesh.m_N_edge_s * m_mesh.m_N_r;
  file.write(m_mesh.tri_face_v0.host_ptr(), n_tri_faces, "tri_face_v0");
  file.write(m_mesh.tri_face_v1.host_ptr(), n_tri_faces, "tri_face_v1");
  file.write(m_mesh.tri_face_v2.host_ptr(), n_tri_faces, "tri_face_v2");
  file.write(m_mesh.rect_face_v0.host_ptr(), n_rect_faces, "rect_face_v0");
  file.write(m_mesh.rect_face_v1.host_ptr(), n_rect_faces, "rect_face_v1");
  file.write(m_mesh.rect_face_v2.host_ptr(), n_rect_faces, "rect_face_v2");
  file.write(m_mesh.rect_face_v3.host_ptr(), n_rect_faces, "rect_face_v3");

  // Write boundary tags
  file.write(m_mesh.edge_boundary.host_ptr(), m_mesh.m_N_edges, "edge_boundary");
  file.write(m_mesh.face_boundary.host_ptr(), m_mesh.m_N_faces, "face_boundary");
  file.write(m_mesh.edge_radial_layer.host_ptr(), m_mesh.m_N_edges,
             "edge_radial_layer");
  file.write(m_mesh.face_radial_layer.host_ptr(), m_mesh.m_N_faces,
             "face_radial_layer");

  // Write Hodge star
  file.write(m_mesh.hodge1_inv.host_ptr(), m_mesh.m_N_edges, "hodge1_inv");
  file.write(m_mesh.hodge2.host_ptr(), m_mesh.m_N_faces, "hodge2");

  // Write radii
  file.write(m_mesh.radii.host_ptr(), m_mesh.m_N_r + 1, "radii");

  // Write sphere mesh data (needed for Whitney interpolation)
  file.write(m_mesh.sphere_vx.host_ptr(), m_mesh.m_N_vert_s, "sphere_vx");
  file.write(m_mesh.sphere_vy.host_ptr(), m_mesh.m_N_vert_s, "sphere_vy");
  file.write(m_mesh.sphere_vz.host_ptr(), m_mesh.m_N_vert_s, "sphere_vz");
  file.write(m_mesh.tri_verts.host_ptr(), m_mesh.m_N_tri * 3, "tri_verts");
  file.write(m_mesh.tri_edges_s.host_ptr(), m_mesh.m_N_tri * 3, "tri_edges_s");
  file.write(m_mesh.tri_edge_signs.host_ptr(), m_mesh.m_N_tri * 3, "tri_edge_signs");
  file.write(m_mesh.tri_neighbor.host_ptr(), m_mesh.m_N_tri * 3, "tri_neighbor");

  // Write incidence matrix d1 (face → edges) in CSR
  int d1_nnz = m_mesh.d1_row_ptr[m_mesh.m_N_faces];
  file.write(m_mesh.d1_row_ptr.host_ptr(), m_mesh.m_N_faces + 1, "d1_row_ptr");
  file.write(m_mesh.d1_col_idx.host_ptr(), d1_nnz, "d1_col_idx");
  file.write(m_mesh.d1_val.host_ptr(), d1_nnz, "d1_val");

  // Write transpose d1^T (edge → faces) in CSR
  int d1t_nnz = m_mesh.d1t_row_ptr[m_mesh.m_N_edges];
  file.write(m_mesh.d1t_row_ptr.host_ptr(), m_mesh.m_N_edges + 1, "d1t_row_ptr");
  file.write(m_mesh.d1t_col_idx.host_ptr(), d1t_nnz, "d1t_col_idx");
  file.write(m_mesh.d1t_val.host_ptr(), d1t_nnz, "d1t_val");

  // Write mesh parameters
  file.write(m_mesh.m_L, "L");
  file.write(m_mesh.m_N_r, "N_r");
  file.write(m_mesh.m_N_verts, "N_verts");
  file.write(m_mesh.m_N_edges, "N_edges");
  file.write(m_mesh.m_N_faces, "N_faces");
  file.write(m_mesh.m_N_tri, "N_tri");
  file.write(m_mesh.m_N_vert_s, "N_vert_s");
  file.write(m_mesh.m_N_edge_s, "N_edge_s");

  // Output downsampling: strides, kept-index arrays, and a small
  // standalone description of the downsampled mesh (the unique vertex
  // indices it references plus their positions). All written
  // unconditionally so the analysis / visualization scripts can detect
  // downsampling unambiguously: when both strides == 1 the index
  // arrays are absent.
  file.write(m_output_radial_stride,  "output_radial_stride");
  file.write(m_output_angular_stride, "output_angular_stride");
  if (m_output_radial_stride > 1 || m_output_angular_stride > 1) {
    file.write(m_out_edge_idx.data(),
               static_cast<size_t>(m_out_edge_idx.size()), "output_edge_idx");
    file.write(m_out_face_idx.data(),
               static_cast<size_t>(m_out_face_idx.size()), "output_face_idx");
    file.write(m_out_vert_idx.data(),
               static_cast<size_t>(m_out_vert_idx.size()), "output_vert_idx");
    // Dense vertex positions for the downsampled mesh — small (one
    // float per vertex per axis) and lets external tools render the
    // downsampled mesh without loading the full vert_x/y/z arrays.
    std::vector<Scalar> vx(m_out_vert_idx.size());
    std::vector<Scalar> vy(m_out_vert_idx.size());
    std::vector<Scalar> vz(m_out_vert_idx.size());
    for (size_t i = 0; i < m_out_vert_idx.size(); ++i) {
      int vi = m_out_vert_idx[i];
      vx[i] = vert_x_cart[vi];
      vy[i] = vert_y_cart[vi];
      vz[i] = vert_z_cart[vi];
    }
    file.write(vx.data(), vx.size(), "output_vert_x");
    file.write(vy.data(), vy.size(), "output_vert_y");
    file.write(vz.data(), vz.size(), "output_vert_z");
  }

  file.close();
  Logger::print_info("Mesh written to {}", filename);
}

// Self-describing snapshot metadata (7D F9): enough for post tools to
// rebuild the mesh (sphere stage) and interpret every dataset without
// mesh.h5 at hand.  "J_kind_dual2" flags the raw dual-2 J cochain
// (multiply by hodge1_inv — analytic — for the primal circulation).
void prismatic_data_exporter::write_meta(H5File& file) {
  file.write(m_mesh.m_L, "L");
  file.write(m_mesh.m_N_r, "N_r");
  file.write(m_mesh.m_N_tri, "N_tri");
  file.write(m_mesh.m_N_edge_s, "N_edge_s");
  file.write(m_mesh.m_N_vert_s, "N_vert_s");
  file.write(m_mesh.radii.host_ptr(), m_mesh.m_N_r + 1, "radii");
  if (m_J != nullptr) {
    file.write(m_J->edge_kind() == EdgeCochainKind::dual_2 ? 1 : 0,
               "J_kind_dual2");
  }
}

void prismatic_data_exporter::write_snapshot(uint32_t step, double time) {
  char fname[256];
  std::snprintf(fname, sizeof(fname), "%s/step_%06u.h5",
                m_output_dir.c_str(), step);

  // Sync fields to host (no-op for host-only buffers)
  m_E->data().copy_to_host();
  m_B->data().copy_to_host();

  if (m_aggregate) {
    write_aggregated(step, time);
    return;
  }

  if (m_distributed) {
    // Collective parallel write: each rank contributes its owned runs
    // of the global cochain datasets.  Bit-identical to the
    // single-rank file (owned values are exact, every global slot has
    // exactly one owner).
    auto file = hdf_create(std::string(fname), H5CreateMode::trunc_parallel);
    file.write_parallel_runs(m_E->host_ptr(), m_E->data().size(),
                             size_t(m_mesh.m_N_edges), m_E_runs.mem_off,
                             m_E_runs.file_off, m_E_runs.len, "E_e");
    file.write_parallel_runs(m_B->host_ptr(), m_B->data().size(),
                             size_t(m_mesh.m_N_faces), m_B_runs.mem_off,
                             m_B_runs.file_off, m_B_runs.len, "B_f");
    if (m_J != nullptr) {
      m_J->data().copy_to_host();
      file.write_parallel_runs(m_J->host_ptr(), m_J->data().size(),
                               size_t(m_mesh.m_N_edges), m_E_runs.mem_off,
                               m_E_runs.file_off, m_E_runs.len, "J_e");
    }
    auto write_vert = [&](nonown_ptr<prismatic_vertex_field>& f,
                          const char* name) {
      if (f == nullptr) return;
      f->data().copy_to_host();
      file.write_parallel_runs(f->host_ptr(), f->data().size(),
                               size_t(m_mesh.m_N_verts), m_V_runs.mem_off,
                               m_V_runs.file_off, m_V_runs.len, name);
    };
    write_vert(m_rho, "rho");
    write_vert(m_rho_abs, "rho_abs");
    write_vert(m_gamma_wsum, "gamma_wsum");
    file.write(static_cast<int>(step), "step");
    file.write(time, "time");
    write_meta(file);
    file.close();
    if (m_comm->radial_rank() == 0 && m_comm->angular_rank() == 0) {
      Logger::print_info("Snapshot written (parallel): step={}, time={:.4f}",
                         step, time);
    }
    return;
  }

  auto file = hdf_create(std::string(fname));

  if (m_output_radial_stride > 1 || m_output_angular_stride > 1) {
    // Gather subsampled values from the host buffers and write them.
    const Scalar* E_h = m_E->host_ptr();
    const Scalar* B_h = m_B->host_ptr();
    for (size_t i = 0; i < m_out_edge_idx.size(); ++i) {
      m_out_E_buf[i] = E_h[m_out_edge_idx[i]];
    }
    for (size_t i = 0; i < m_out_face_idx.size(); ++i) {
      m_out_B_buf[i] = B_h[m_out_face_idx[i]];
    }
    file.write(m_out_E_buf.data(),
               static_cast<size_t>(m_out_E_buf.size()), "E_e");
    file.write(m_out_B_buf.data(),
               static_cast<size_t>(m_out_B_buf.size()), "B_f");
  } else {
    // Full output (default).
    file.write(m_E->host_ptr(), m_mesh.m_N_edges, "E_e");
    file.write(m_B->host_ptr(), m_mesh.m_N_faces, "B_f");
    if (m_J != nullptr) {
      m_J->data().copy_to_host();
      file.write(m_J->host_ptr(), m_mesh.m_N_edges, "J_e");
    }
    auto write_vert = [&](nonown_ptr<prismatic_vertex_field>& f,
                          const char* name) {
      if (f == nullptr) return;
      f->data().copy_to_host();
      file.write(f->host_ptr(), m_mesh.m_N_verts, name);
    };
    write_vert(m_rho, "rho");
    write_vert(m_rho_abs, "rho_abs");
    write_vert(m_gamma_wsum, "gamma_wsum");
  }

  // Write metadata
  file.write(static_cast<int>(step), "step");
  file.write(time, "time");
  write_meta(file);

  file.close();
  Logger::print_info("Snapshot written: step={}, time={:.4f}", step, time);
}

// ===========================================================================
// Aggregated coarse-cochain snapshots (7D F9).  Each rank accumulates
// partial coarse sums over the fine elements it OWNS; distributed runs
// MPI_SUM-reduce the (small) coarse arrays to world rank 0, which
// writes serially.  No slab alignment constraint: partial sums add up
// correctly regardless of where rank boundaries fall.
// ===========================================================================
void prismatic_data_exporter::write_aggregated(uint32_t step, double time) {
  // Fine-value accessors: the cochain's OWN global index in, the field
  // value out — 0 for contributions this rank does not own (each fine
  // element has exactly one owner, so the MPI_SUM of partials is the
  // complete aggregation).
  auto make_val = [&](cochain_type t, const Scalar* data, int block_off) {
    const distributed_cochain_layout* L =
        m_distributed ? &m_mp->layout(t) : nullptr;
    return [L, data, block_off](int g) -> double {
      if (L == nullptr) return double(data[block_off + g]);
      const int l = L->to_local(g);
      if (l < 0 || l >= L->owned_size()) return 0.0;
      return double(data[block_off + l]);
    };
  };
  // NOTE on indexing: the per-cochain accessors take the cochain's OWN
  // global index (h edges from 0, v edges from 0, ...).  Single-rank
  // block offsets place the sub-blocks inside the combined buffers.
  const int e_split =
      m_distributed ? m_E->split() : (m_mesh.m_N_r + 1) * m_mesh.m_N_edge_s;
  const int b_split =
      m_distributed ? m_B->split() : (m_mesh.m_N_r + 1) * m_mesh.m_N_tri;

  auto fill = [&](std::vector<double>& out_E, const Scalar* E_data) {
    std::fill(out_E.begin(), out_E.end(), 0.0);
    m_agg.agg_h_edges(make_val(cochain_type::h_edge, E_data, 0),
                      out_E.data());
    m_agg.agg_v_edges(make_val(cochain_type::v_edge, E_data, e_split),
                      out_E.data() + m_agg.n_h_c());
  };
  auto fill_face = [&](std::vector<double>& out_B, const Scalar* B_data) {
    std::fill(out_B.begin(), out_B.end(), 0.0);
    m_agg.agg_tri_faces(make_val(cochain_type::tri_face, B_data, 0),
                        out_B.data());
    m_agg.agg_rect_faces(make_val(cochain_type::rect_face, B_data, b_split),
                         out_B.data() + m_agg.n_trif_c());
  };

  fill(m_agg_E, m_E->host_ptr());
  fill_face(m_agg_B, m_B->host_ptr());
  if (m_J != nullptr) {
    m_J->data().copy_to_host();
    m_agg_J.resize(m_agg_E.size());
    fill(m_agg_J, m_J->host_ptr());
  }
  auto fill_vert = [&](std::vector<double>& out,
                       nonown_ptr<prismatic_vertex_field>& f) {
    if (f == nullptr) return;
    f->data().copy_to_host();
    out.assign(m_agg.n_vertc_c(), 0.0);
    m_agg.agg_vertices(make_val(cochain_type::vertex, f->host_ptr(), 0),
                       out.data());
  };
  fill_vert(m_agg_rho, m_rho);
  fill_vert(m_agg_ra, m_rho_abs);
  fill_vert(m_agg_gw, m_gamma_wsum);

  int world_rank = 0;
  if (m_distributed) {
    world_rank = m_comm->world_rank();
    auto reduce = [&](std::vector<double>& v) {
      if (v.empty()) return;
      MPI_Reduce(world_rank == 0 ? MPI_IN_PLACE : v.data(), v.data(),
                 int(v.size()), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    };
    reduce(m_agg_E);
    reduce(m_agg_B);
    reduce(m_agg_J);
    reduce(m_agg_rho);
    reduce(m_agg_ra);
    reduce(m_agg_gw);
    if (world_rank != 0) return;
  }

  char fname[256];
  std::snprintf(fname, sizeof(fname), "%s/step_%06u.h5",
                m_output_dir.c_str(), step);
  auto file = hdf_create(std::string(fname));
  auto wr = [&](const std::vector<double>& v, const char* name) {
    if (v.empty()) return;
    std::vector<Scalar> tmp(v.begin(), v.end());
    file.write(tmp.data(), tmp.size(), name);
  };
  wr(m_agg_E, "E_e");
  wr(m_agg_B, "B_f");
  wr(m_agg_J, "J_e");
  wr(m_agg_rho, "rho");
  wr(m_agg_ra, "rho_abs");
  wr(m_agg_gw, "gamma_wsum");
  file.write(static_cast<int>(step), "step");
  file.write(time, "time");
  // Self-describing COARSE metadata: the dump is a bona fide
  // level-L_out DEC field on the strided radii.
  file.write(1, "aggregated");
  file.write(m_agg.L_out, "L");
  file.write(m_agg.N_r_c, "N_r");
  file.write(m_agg.n_tri_c, "N_tri");
  file.write(m_agg.n_edge_c, "N_edge_s");
  file.write(m_agg.n_vert_c, "N_vert_s");
  {
    std::vector<Scalar> cr(m_agg.N_r_c + 1);
    for (int K = 0; K <= m_agg.N_r_c; ++K) {
      cr[K] = m_mesh.radii[K * m_agg.R];
    }
    file.write(cr.data(), cr.size(), "radii");
  }
  if (m_J != nullptr) {
    // Aggregation of the raw dual-2 J is NOT a dual cochain on the
    // coarse mesh; flag it so post tools convert per-fine-edge instead.
    file.write(m_J->edge_kind() == EdgeCochainKind::dual_2 ? 1 : 0,
               "J_kind_dual2");
  }
  file.close();
  Logger::print_info("Aggregated snapshot: step={}, time={:.4f}", step, time);
}

}  // namespace Aperture
