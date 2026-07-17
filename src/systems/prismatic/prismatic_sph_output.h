#pragma once

#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_vertex_recovery.h"
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
  nonown_ptr<prismatic_vertex_field> m_rho_abs;
  nonown_ptr<prismatic_vertex_field> m_gamma_wsum;

  int m_N_theta = 180;
  int m_N_phi = 360;
  int m_output_interval = 100;
  std::string m_output_dir = "Data";
  double m_time = 0.0;

  // Second-order C0 vertex-recovery gather for B on the output grid
  // (config "sph_use_recovery", default true).  The primal Whitney
  // gather is first order and measurably biases quadratic diagnostics
  // (e.g. Poynting luminosity reads 0.85 of analytic at L=5).  The
  // recovery fit assumes the flat mesh geometry — disable for GR runs.
  prismatic_vertex_recovery m_recovery;
  bool m_use_recovery = true;

  // Background metric identity, used to convert the flat-mesh Whitney
  // reconstruction to coordinate-basis KS (or flat-spherical) field
  // components at output time.  Read from the same config keys as the
  // GR solver / mesh-metric construction ("bh_spin", "use_flat_metric")
  // so output and evolution see the identical √γ.
  //
  // DEFAULT IS FLAT.  GR runs must set use_flat_metric = false (plus
  // bh_spin) explicitly.  The old default (false) silently applied the
  // a=0 Kerr-Schild √γ = sinθ√(Σ(Σ+2r)) — i.e. Schwarzschild with
  // M=1 — to flat-space problems whose configs never set the key,
  // scaling B^i down by √(1+2/r) (a 12% Poynting-flux deficit at r=7,
  // found by the A2.1 luminosity benchmark).
  Scalar m_spin = 0;
  bool m_use_flat_metric = true;

  struct angular_point {
    int tri_idx;
    Scalar l[3];
    Scalar sx, sy, sz;
    // Meridian frame of this grid column, stored from the φ loop
    // variable: at the poles sx = sy = 0 and the frame cannot be
    // recovered from the Cartesian direction, but every column still
    // has a well-defined θ̂/φ̂ along its own meridian.
    Scalar cos_phi, sin_phi;
  };
  std::vector<angular_point> m_grid;

  std::vector<Scalar> m_Br, m_Bth, m_Bph;
  std::vector<Scalar> m_Er, m_Eth, m_Eph;
  std::vector<Scalar> m_Jr, m_Jth, m_Jph;
  std::vector<Scalar> m_rho_grid, m_rho_abs_grid, m_gamma_grid;
};

}  // namespace Aperture
