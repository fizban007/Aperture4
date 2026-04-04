#pragma once

#include "framework/system.h"
#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/prismatic_mesh.h"
#include <string>

namespace Aperture {

class prismatic_data_exporter : public system_t {
 public:
  static std::string name() { return "prismatic_data_exporter"; }

  prismatic_data_exporter(const prismatic_mesh& mesh,
                          dec_field_solver& solver);
  ~prismatic_data_exporter() = default;

  void init() override;
  void update(double dt, uint32_t step) override;

 private:
  void write_mesh();
  void write_snapshot(uint32_t step, double time);

  const prismatic_mesh& m_mesh;
  dec_field_solver& m_solver;

  int m_output_interval = 100;
  std::string m_output_dir = "Data";
  double m_time = 0.0;
};

}  // namespace Aperture
