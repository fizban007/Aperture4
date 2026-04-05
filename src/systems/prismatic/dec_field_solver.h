#pragma once

#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "utils/nonown_ptr.hpp"

namespace Aperture {

template <typename ExecPolicy>
class dec_field_solver : public system_t {
 public:
  static std::string name() { return "dec_field_solver"; }

  dec_field_solver(prismatic_mesh& mesh);
  ~dec_field_solver() = default;

  void register_data_components() override;
  void init() override;
  void update(double dt, uint32_t step) override;

 private:
  void update_explicit(double dt);
  void update_semi_implicit(double dt);

  void compute_rhs(buffer<Scalar>& E_in, buffer<Scalar>& B_in,
                   buffer<Scalar>& dE_dt, buffer<Scalar>& dB_dt);

  void apply_damping(buffer<Scalar>& E, buffer<Scalar>& B, double dt);
  void apply_inner_bc(buffer<Scalar>& E, buffer<Scalar>& B, double time);
  void set_initial_dipole();

  Scalar project_B_on_face(int face_idx, Scalar Bx, Scalar By, Scalar Bz) const;
  Scalar project_E_on_edge(int edge_idx, Scalar Ex, Scalar Ey, Scalar Ez) const;

  static void dipole_B(Scalar x, Scalar y, Scalar z, Scalar mx, Scalar my,
                       Scalar mz, Scalar& Bx, Scalar& By, Scalar& Bz);

  prismatic_mesh& m_mesh;

  // Shared field data (owned by env, found in register_data_components)
  nonown_ptr<prismatic_edge_field> m_E;
  nonown_ptr<prismatic_face_field> m_B;
  nonown_ptr<prismatic_edge_field> m_J;

  // Temporary buffers for semi-implicit iteration (owned by solver)
  buffer<Scalar> m_tmp_E, m_tmp_B;
  buffer<Scalar> m_dE_dt, m_dB_dt;
  buffer<Scalar> m_dE_dt_new, m_dB_dt_new;

  // Physics parameters
  Scalar m_Bp = 1.0;
  Scalar m_Omega = 1.0;
  Scalar m_obliquity = 0.0;

  // Damping layer
  int m_damping_length = 10;
  Scalar m_damping_coef = 0.05;

  // Update toggles
  bool m_update_e = true;
  bool m_update_b = true;

  // Semi-implicit parameters
  bool m_use_implicit = false;
  Scalar m_beta = 0.55;
  int m_implicit_iters = 4;

  double m_time = 0.0;
};

using dec_field_solver_t = dec_field_solver<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
