#pragma once

#include "core/buffer.hpp"
#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_particles.h"

namespace Aperture {

class prismatic_ptc_updater : public system_t {
 public:
  static std::string name() { return "prismatic_ptc_updater"; }

  prismatic_ptc_updater(prismatic_mesh& mesh,
                        buffer<Scalar>& E_e, buffer<Scalar>& B_f,
                        buffer<Scalar>& J_e);
  ~prismatic_ptc_updater() = default;

  void init() override;
  void update(double dt, uint32_t step) override;

  prismatic_particles_t& particles() { return m_particles; }
  const prismatic_particles_t& particles() const { return m_particles; }

  // Add a particle at the given 3D Cartesian position with given momentum.
  // Returns the particle index, or -1 if the position is outside the mesh.
  int add_particle(Scalar x, Scalar y, Scalar z,
                   Scalar px, Scalar py, Scalar pz,
                   Scalar weight, uint32_t flag = 0);

 private:
  // Convert 3D Cartesian position to prism local coordinates.
  // Returns false if the point is outside the mesh.
  bool cartesian_to_local(Scalar x, Scalar y, Scalar z,
                          int& tri_idx, int& layer_idx,
                          Scalar& l1, Scalar& l2, Scalar& zeta,
                          int tri_hint = -1) const;

  // Convert prism local coordinates to 3D Cartesian position.
  void local_to_cartesian(int tri_idx, int layer_idx,
                          Scalar l1, Scalar l2, Scalar zeta,
                          Scalar& x, Scalar& y, Scalar& z) const;

  // Remove particles flagged for deletion (cell == empty_cell)
  void remove_dead_particles();

  prismatic_mesh& m_mesh;
  buffer<Scalar>& m_E_e;
  buffer<Scalar>& m_B_f;
  buffer<Scalar>& m_J_e;
  prismatic_particles_t m_particles;

  // Physics parameters
  Scalar m_charge_e = -1.0;  // electron charge
  Scalar m_mass_e = 1.0;     // electron mass
};

}  // namespace Aperture
