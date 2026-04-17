#pragma once

#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "utils/nonown_ptr.hpp"
#include <string>
#include <vector>

namespace Aperture {

class prismatic_sph_output : public system_t {
 public:
  static std::string name() { return "prismatic_sph_output"; }

  prismatic_sph_output(prismatic_mesh& mesh);
  ~prismatic_sph_output() = default;

  void register_data_components() override;
  void init() override;
  void update(double dt, uint32_t step) override;

 private:
  void precompute_grid();
  void write_grid_info();
  void write_snapshot(uint32_t step, double time);

  prismatic_mesh& m_mesh;

  // Shared data (found from env)
  nonown_ptr<prismatic_edge_field> m_E;
  nonown_ptr<prismatic_face_field> m_B;
  nonown_ptr<prismatic_edge_field> m_J;
  nonown_ptr<prismatic_vertex_field> m_rho;

  int m_N_theta = 180;
  int m_N_phi = 360;
  int m_output_interval = 100;
  std::string m_output_dir = "Data";
  double m_time = 0.0;

  // Background metric identity, used to convert the flat-mesh Whitney
  // reconstruction to coordinate-basis KS (or flat-spherical) field
  // components at output time.  Read from the same config keys as the
  // GR solver / mesh-metric construction ("bh_spin", "use_flat_metric")
  // so output and evolution see the identical √γ.
  Scalar m_spin = 0;
  bool m_use_flat_metric = false;

  struct angular_point {
    int tri_idx;
    Scalar l[3];
    Scalar sx, sy, sz;
  };
  std::vector<angular_point> m_grid;

  std::vector<Scalar> m_Br, m_Bth, m_Bph;
  std::vector<Scalar> m_Er, m_Eth, m_Eph;
  std::vector<Scalar> m_Jr, m_Jth, m_Jph;
  std::vector<Scalar> m_rho_grid;
};

}  // namespace Aperture
