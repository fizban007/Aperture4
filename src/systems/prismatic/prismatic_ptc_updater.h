#pragma once

#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "utils/nonown_ptr.hpp"

namespace Aperture {

template <typename ExecPolicy>
class prismatic_ptc_updater : public system_t {
 public:
  static std::string name() { return "prismatic_ptc_updater"; }

  prismatic_ptc_updater(prismatic_mesh& mesh);
  ~prismatic_ptc_updater() = default;

  void register_data_components() override;
  void init() override;
  void update(double dt, uint32_t step) override;

  nonown_ptr<prismatic_particle_data> particles() { return m_ptc; }

  int add_particle(Scalar x, Scalar y, Scalar z,
                   Scalar px, Scalar py, Scalar pz,
                   Scalar weight, uint32_t flag = 0);

 private:
  prismatic_mesh& m_mesh;

  nonown_ptr<prismatic_edge_field> m_E;
  nonown_ptr<prismatic_face_field> m_B;
  nonown_ptr<prismatic_edge_field> m_J;
  nonown_ptr<prismatic_vertex_field> m_rho;
  nonown_ptr<prismatic_particle_data> m_ptc;

  Scalar m_charge_e = -1.0;
  Scalar m_mass_e = 1.0;
};

using prismatic_ptc_updater_t = prismatic_ptc_updater<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
