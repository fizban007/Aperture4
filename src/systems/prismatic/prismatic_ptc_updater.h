#pragma once

#include "core/typedefs_and_constants.h"
#include "data/rng_states.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_vertex_recovery.h"
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
  // Shared RNG pool (consumed by prismatic_ptc_injector, mirroring the
  // base ptc_updater's registration of "rng_states").
  nonown_ptr<rng_states_t<typename ExecPolicy::exec_tag>> m_rng_states;

  // C0 second-order B-gather (see prismatic_vertex_recovery.h); the
  // primal Whitney gather pitch-angle-scatters particles off face jumps.
  // Config "use_recovery_gather" (default true) selects it; E-gather and
  // deposition always stay primal Whitney.
  prismatic_vertex_recovery m_recovery;
  bool m_use_recovery_gather = true;

  // Absorb particles beyond this radius (config "ptc_absorb_radius").
  // <= 0 (default) disables the check; particles are then only absorbed
  // implicitly at the domain edges [r_min, r_max].  Magnetosphere runs
  // should set this to the damping-layer entrance.
  Scalar m_absorb_radius = Scalar(0);

  Scalar m_charge_e = -1.0;
  Scalar m_mass_e = 1.0;
  int m_sort_interval = 100;
  bool m_use_gca = false;
  bool m_include_curvature = false;
};

using prismatic_ptc_updater_t = prismatic_ptc_updater<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
