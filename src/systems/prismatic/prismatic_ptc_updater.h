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
  buffer<Scalar>& rho() { return m_rho; }
  const buffer<Scalar>& rho() const { return m_rho; }

  int add_particle(Scalar x, Scalar y, Scalar z,
                   Scalar px, Scalar py, Scalar pz,
                   Scalar weight, uint32_t flag = 0);

 private:
  void remove_dead_particles();

  prismatic_mesh& m_mesh;
  buffer<Scalar>& m_E_e;
  buffer<Scalar>& m_B_f;
  buffer<Scalar>& m_J_e;
  buffer<Scalar> m_rho;   // charge density on vertices [N_verts]
  prismatic_particles_t m_particles;

  Scalar m_charge_e = -1.0;
  Scalar m_mass_e = 1.0;
};

}  // namespace Aperture
