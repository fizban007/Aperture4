#pragma once

#include "framework/system.h"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "utils/nonown_ptr.hpp"
#include <string>

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
  std::string m_output_dir = "Data";
  double m_time = 0.0;
};

}  // namespace Aperture
