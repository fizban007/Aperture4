#pragma once

#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "utils/nonown_ptr.hpp"

namespace Aperture {

class prismatic_ptc_updater : public system_t {
 public:
  static std::string name() { return "prismatic_ptc_updater"; }

  prismatic_ptc_updater(prismatic_mesh& mesh);
  ~prismatic_ptc_updater() = default;

  void register_data_components() override;
  void init() override;
  void update(double dt, uint32_t step) override;

  int add_particle(Scalar x, Scalar y, Scalar z,
                   Scalar px, Scalar py, Scalar pz,
                   Scalar weight, uint32_t flag = 0);

 private:
  void remove_dead_particles();

  prismatic_mesh& m_mesh;

  // Shared data (owned by env)
  nonown_ptr<prismatic_edge_field> m_E;
  nonown_ptr<prismatic_face_field> m_B;
  nonown_ptr<prismatic_edge_field> m_J;
  nonown_ptr<prismatic_vertex_field> m_rho;
  nonown_ptr<prismatic_particle_data> m_ptc;

  Scalar m_charge_e = -1.0;
  Scalar m_mass_e = 1.0;
};

}  // namespace Aperture
