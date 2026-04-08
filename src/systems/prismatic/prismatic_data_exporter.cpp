#include "systems/prismatic/prismatic_data_exporter.h"
#include "framework/environment.h"
#include "utils/hdf_wrapper.h"
#include "utils/logger.h"
#include <cstdio>
#include <filesystem>
#include <string>

namespace Aperture {

prismatic_data_exporter::prismatic_data_exporter(const prismatic_mesh& mesh)
    : m_mesh(mesh) {}

void prismatic_data_exporter::register_data_components() {
  m_E = sim_env().register_data<prismatic_edge_field>("E", m_mesh);
  m_B = sim_env().register_data<prismatic_face_field>("B", m_mesh);
}

void prismatic_data_exporter::init() {
  sim_env().params().get_value("fld_output_interval", m_output_interval);
  sim_env().params().get_value("fld_output_subsample", m_output_subsample);
  sim_env().params().get_value("output_dir", m_output_dir);

  if (m_output_subsample < 1) m_output_subsample = 1;

  // Create output directory
  std::filesystem::create_directories(m_output_dir);

  // Build kept-index arrays for downsampled output. Plain stride over
  // the global index — simple, deterministic, and reproducible by the
  // analysis script. The boundary elements are not filtered out;
  // analysis code can mask them via face_boundary / edge_boundary as
  // needed.
  if (m_output_subsample > 1) {
    int N_e = m_mesh.m_N_edges;
    int N_f = m_mesh.m_N_faces;
    m_out_edge_idx.reserve((N_e + m_output_subsample - 1) / m_output_subsample);
    m_out_face_idx.reserve((N_f + m_output_subsample - 1) / m_output_subsample);
    for (int i = 0; i < N_e; i += m_output_subsample) {
      m_out_edge_idx.push_back(i);
    }
    for (int i = 0; i < N_f; i += m_output_subsample) {
      m_out_face_idx.push_back(i);
    }
    m_out_E_buf.resize(m_out_edge_idx.size());
    m_out_B_buf.resize(m_out_face_idx.size());
    Logger::print_info(
        "Output subsample = {}: writing {}/{} edges, {}/{} faces per snapshot",
        m_output_subsample,
        static_cast<int>(m_out_edge_idx.size()), N_e,
        static_cast<int>(m_out_face_idx.size()), N_f);
  }

  // Write mesh file once
  write_mesh();

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

  // Write vertex positions
  file.write(m_mesh.vert_x.host_ptr(), m_mesh.m_N_verts, "vert_x");
  file.write(m_mesh.vert_y.host_ptr(), m_mesh.m_N_verts, "vert_y");
  file.write(m_mesh.vert_z.host_ptr(), m_mesh.m_N_verts, "vert_z");

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

  // Output downsampling: stride and the kept-index arrays. Always
  // written so the analysis script can detect downsampling
  // unambiguously (subsample == 1 → indices are absent).
  file.write(m_output_subsample, "output_subsample");
  if (m_output_subsample > 1) {
    file.write(m_out_edge_idx.data(),
               static_cast<size_t>(m_out_edge_idx.size()), "output_edge_idx");
    file.write(m_out_face_idx.data(),
               static_cast<size_t>(m_out_face_idx.size()), "output_face_idx");
  }

  file.close();
  Logger::print_info("Mesh written to {}", filename);
}

void prismatic_data_exporter::write_snapshot(uint32_t step, double time) {
  char fname[256];
  std::snprintf(fname, sizeof(fname), "%s/step_%06u.h5",
                m_output_dir.c_str(), step);
  auto file = hdf_create(std::string(fname));

  // Sync fields to host (no-op for host-only buffers)
  m_E->data().copy_to_host();
  m_B->data().copy_to_host();

  if (m_output_subsample > 1) {
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
  }

  // Write metadata
  file.write(static_cast<int>(step), "step");
  file.write(time, "time");

  file.close();
  Logger::print_info("Snapshot written: step={}, time={:.4f}", step, time);
}

}  // namespace Aperture
