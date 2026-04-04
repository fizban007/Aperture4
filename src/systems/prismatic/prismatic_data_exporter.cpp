#include "systems/prismatic/prismatic_data_exporter.h"
#include "framework/environment.h"
#include "utils/hdf_wrapper.h"
#include "utils/logger.h"
#include <cstdio>
#include <filesystem>
#include <string>

namespace Aperture {

prismatic_data_exporter::prismatic_data_exporter(const prismatic_mesh& mesh,
                                                 dec_field_solver& solver)
    : m_mesh(mesh), m_solver(solver) {}

void prismatic_data_exporter::init() {
  sim_env().params().get_value("fld_output_interval", m_output_interval);
  sim_env().params().get_value("output_dir", m_output_dir);

  // Create output directory
  std::filesystem::create_directories(m_output_dir);

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

  // Write mesh parameters
  file.write(m_mesh.m_L, "L");
  file.write(m_mesh.m_N_r, "N_r");
  file.write(m_mesh.m_N_verts, "N_verts");
  file.write(m_mesh.m_N_edges, "N_edges");
  file.write(m_mesh.m_N_faces, "N_faces");
  file.write(m_mesh.m_N_tri, "N_tri");
  file.write(m_mesh.m_N_vert_s, "N_vert_s");
  file.write(m_mesh.m_N_edge_s, "N_edge_s");

  file.close();
  Logger::print_info("Mesh written to {}", filename);
}

void prismatic_data_exporter::write_snapshot(uint32_t step, double time) {
  char fname[256];
  std::snprintf(fname, sizeof(fname), "%s/step_%06u.h5",
                m_output_dir.c_str(), step);
  auto file = hdf_create(std::string(fname));

  // Write field data
  file.write(m_solver.E_e().host_ptr(), m_mesh.m_N_edges, "E_e");
  file.write(m_solver.B_f().host_ptr(), m_mesh.m_N_faces, "B_f");

  // Write metadata
  file.write(static_cast<int>(step), "step");
  file.write(time, "time");

  file.close();
  Logger::print_info("Snapshot written: step={}, time={:.4f}", step, time);
}

}  // namespace Aperture
