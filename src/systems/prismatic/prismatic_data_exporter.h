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
  // Stride for downsampling raw cochain output. 1 = no downsampling
  // (writes every edge / face). N > 1 keeps every N-th element of the
  // global edge / face index, with the kept indices written to mesh.h5
  // as `output_edge_idx` and `output_face_idx` so the analysis script
  // knows what was retained. The unsampled snapshots become smaller by
  // a factor of N, which is the dominant cost at large L.
  int m_output_subsample = 1;
  std::string m_output_dir = "Data";
  double m_time = 0.0;

  // Built once in init(); empty when m_output_subsample == 1.
  std::vector<int> m_out_edge_idx;
  std::vector<int> m_out_face_idx;
  // Per-snapshot scratch buffers (avoid reallocation each call).
  std::vector<Scalar> m_out_E_buf;
  std::vector<Scalar> m_out_B_buf;
};

}  // namespace Aperture
