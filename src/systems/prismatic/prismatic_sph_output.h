#pragma once

#include "core/buffer.hpp"
#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_mesh.h"
#include <string>
#include <vector>

namespace Aperture {

// Interpolates field data onto a regular spherical (theta, phi) grid at
// each radial shell at output time.  The angular grid is precomputed once
// at init: for each (theta_i, phi_j) we find the containing sphere
// triangle and barycentric coordinates, which are reused for every shell.
//
// Since grid points sit exactly on radial shells, the radial hat function
// is trivial: phi_0 = 1, phi_1 = 0.  Only the 3 horizontal edges and
// 3 vertical edges of the bottom level of the prism contribute.

class prismatic_sph_output : public system_t {
 public:
  static std::string name() { return "prismatic_sph_output"; }

  prismatic_sph_output(prismatic_mesh& mesh,
                       buffer<Scalar>& E_e, buffer<Scalar>& B_f);
  ~prismatic_sph_output() = default;

  void init() override;
  void update(double dt, uint32_t step) override;

 private:
  void precompute_grid();
  void write_grid_info();
  void write_snapshot(uint32_t step, double time);

  prismatic_mesh& m_mesh;
  buffer<Scalar>& m_E_e;
  buffer<Scalar>& m_B_f;

  int m_N_theta = 180;
  int m_N_phi = 360;
  int m_output_interval = 100;
  std::string m_output_dir = "Data";
  double m_time = 0.0;

  // Precomputed angular grid: for each (i_theta, i_phi), the containing
  // triangle and barycentric coordinates on the unit sphere.
  struct angular_point {
    int tri_idx;
    Scalar l[3];   // barycentric coordinates
    Scalar sx, sy, sz;  // unit sphere position (for coordinate transforms)
  };
  std::vector<angular_point> m_grid;  // [N_theta * N_phi]

  // Output buffers
  std::vector<Scalar> m_Br, m_Bth, m_Bph;
  std::vector<Scalar> m_Er, m_Eth, m_Eph;
};

}  // namespace Aperture
