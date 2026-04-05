#pragma once

#include "core/buffer.hpp"
#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_mesh.h"

namespace Aperture {

template <typename ExecPolicy>
class dec_field_solver : public system_t {
 public:
  static std::string name() { return "dec_field_solver"; }

  dec_field_solver(prismatic_mesh& mesh);
  ~dec_field_solver() = default;

  void init() override;
  void update(double dt, uint32_t step) override;

  // Access to field data
  buffer<Scalar>& E_e() { return m_E_e; }
  buffer<Scalar>& B_f() { return m_B_f; }
  buffer<Scalar>& J_e() { return m_J_e; }
  const buffer<Scalar>& E_e() const { return m_E_e; }
  const buffer<Scalar>& B_f() const { return m_B_f; }
  const buffer<Scalar>& J_e() const { return m_J_e; }

 private:
  void update_explicit(double dt);
  void update_semi_implicit(double dt);

  // Compute RHS: dB/dt = -d1*E, dE/dt = h1inv*(d1t*h2*B - J)
  void compute_rhs(buffer<Scalar>& E_in, buffer<Scalar>& B_in,
                   buffer<Scalar>& dE_dt, buffer<Scalar>& dB_dt);

  void apply_damping(buffer<Scalar>& E, buffer<Scalar>& B,
                     buffer<Scalar>& J, double dt);
  void apply_inner_bc(buffer<Scalar>& E, buffer<Scalar>& B, double time);
  void set_initial_dipole();

  Scalar project_B_on_face(int face_idx, Scalar Bx, Scalar By, Scalar Bz) const;
  Scalar project_E_on_edge(int edge_idx, Scalar Ex, Scalar Ey, Scalar Ez) const;

  static void dipole_B(Scalar x, Scalar y, Scalar z, Scalar mx, Scalar my,
                       Scalar mz, Scalar& Bx, Scalar& By, Scalar& Bz);

  prismatic_mesh& m_mesh;

  // Primary field storage
  buffer<Scalar> m_E_e;   // electric field line integrals on edges
  buffer<Scalar> m_B_f;   // magnetic flux on faces
  buffer<Scalar> m_J_e;   // current 1-cochain on edges (accumulated by deposit)

  // Temporary buffers for semi-implicit iteration
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

  // Semi-implicit parameters
  bool m_use_implicit = false;
  Scalar m_beta = 0.55;       // implicitness (0.5 = trapezoidal, >0.5 = dissipative)
  int m_implicit_iters = 4;   // number of predictor-corrector iterations

  double m_time = 0.0;
};

// Convenience alias: uses host policy on CPU builds, GPU policy on GPU builds
using dec_field_solver_t = dec_field_solver<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
