#pragma once

#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "utils/nonown_ptr.hpp"
#include <string>
#include <vector>

namespace Aperture {

class prismatic_data_exporter : public system_t {
 public:
  static std::string name() { return "prismatic_data_exporter"; }

  prismatic_data_exporter(const prismatic_mesh& mesh);
  ~prismatic_data_exporter() = default;

  void register_data_components() override;
  void init() override;
  void update(double dt, uint32_t step) override;

 private:
  void write_mesh();
  void write_snapshot(uint32_t step, double time);

  const prismatic_mesh& m_mesh;
  nonown_ptr<prismatic_edge_field> m_E;
  nonown_ptr<prismatic_face_field> m_B;

  int m_output_interval = 100;

  // Structured downsampling of raw cochain output. The radial stride
  // applies to the shell index k; the angular stride applies to the
  // sphere-element index (sphere triangle for triangular faces, sphere
  // edge for rectangular faces / horizontal edges, sphere vertex for
  // vertical edges). Both default to 1 (full output, identical to
  // pre-existing behavior).
  //
  // Setting (radial=2, angular=4) downsamples by a factor of 8 in a
  // way that mirrors exactly one level of refinement coarsening: every
  // other shell × the corner-child of every parent triangle (the
  // subdivide() routine pushes the (a, m_ab, m_ac) corner child first
  // for every parent, so stride 4 in t selects one specific child per
  // parent — see prismatic_mesh.cpp:117).
  //
  // The kept indices for each element type are stored in mesh.h5 as
  // output_face_idx / output_edge_idx, plus the strides themselves and
  // the unique vertices referenced by the kept set so external tools
  // can render the downsampled mesh standalone.
  int m_output_radial_stride  = 1;
  int m_output_angular_stride = 1;
  std::string m_output_dir = "Data";
  double m_time = 0.0;

  // Built once in init(); empty when both strides are 1.
  std::vector<int> m_out_edge_idx;
  std::vector<int> m_out_face_idx;
  // Unique vertex indices referenced by the kept faces and edges,
  // sorted ascending. Lets visualization tools build a self-contained
  // vertex list for the downsampled mesh.
  std::vector<int> m_out_vert_idx;
  // Per-snapshot scratch buffers (avoid reallocation each call).
  std::vector<Scalar> m_out_E_buf;
  std::vector<Scalar> m_out_B_buf;
};

}  // namespace Aperture
