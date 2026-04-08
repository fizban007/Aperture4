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
  sim_env().params().get_value("fld_output_radial_stride",
                               m_output_radial_stride);
  sim_env().params().get_value("fld_output_angular_stride",
                               m_output_angular_stride);
  sim_env().params().get_value("output_dir", m_output_dir);

  if (m_output_radial_stride  < 1) m_output_radial_stride  = 1;
  if (m_output_angular_stride < 1) m_output_angular_stride = 1;

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

  if (R > 1 || A > 1) {
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
      vx[i] = m_mesh.vert_x[vi];
      vy[i] = m_mesh.vert_y[vi];
      vz[i] = m_mesh.vert_z[vi];
    }
    file.write(vx.data(), vx.size(), "output_vert_x");
    file.write(vy.data(), vy.size(), "output_vert_y");
    file.write(vz.data(), vz.size(), "output_vert_z");
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
  }

  // Write metadata
  file.write(static_cast<int>(step), "step");
  file.write(time, "time");

  file.close();
  Logger::print_info("Snapshot written: step={}, time={:.4f}", step, time);
}

}  // namespace Aperture
