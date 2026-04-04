#pragma once

#include "core/buffer.hpp"
#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_mesh.h"

namespace Aperture {

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
  const buffer<Scalar>& E_e() const { return m_E_e; }
  const buffer<Scalar>& B_f() const { return m_B_f; }

 private:
  void apply_damping(double dt);
  void apply_inner_bc(double time);
  void set_initial_dipole();

  // CG solver: solves M₁ × x = rhs, returns iteration count
  int cg_solve(const buffer<Scalar>& rhs, buffer<Scalar>& x,
               int max_iter, Scalar tol);

  // Project analytic fields onto mesh elements
  Scalar project_B_on_face(int face_idx, Scalar Bx, Scalar By, Scalar Bz) const;
  Scalar project_E_on_edge(int edge_idx, Scalar Ex, Scalar Ey, Scalar Ez) const;

  static void dipole_B(Scalar x, Scalar y, Scalar z, Scalar mx, Scalar my,
                       Scalar mz, Scalar& Bx, Scalar& By, Scalar& Bz);

  prismatic_mesh& m_mesh;

  // Primary field storage
  buffer<Scalar> m_E_e;   // electric field line integrals on edges
  buffer<Scalar> m_B_f;   // magnetic flux on faces

  // Scratch buffers for CG and update
  buffer<Scalar> m_rhs;       // RHS of Ampere: d₁ᵀ M₂ B
  buffer<Scalar> m_M2B;       // M₂ × B (scratch)
  buffer<Scalar> m_dE_prev;   // Previous CG solution for warm-start
  buffer<Scalar> m_cg_r;      // CG residual
  buffer<Scalar> m_cg_z;      // CG preconditioned residual
  buffer<Scalar> m_cg_p;      // CG search direction
  buffer<Scalar> m_cg_Ap;     // CG matrix-vector product

  // Physics parameters
  Scalar m_Bp = 1.0;
  Scalar m_Omega = 1.0;
  Scalar m_obliquity = 0.0;

  // Damping layer
  int m_damping_length = 10;
  Scalar m_damping_coef = 0.05;

  // CG parameters
  int m_cg_max_iter = 10;
  Scalar m_cg_tol = 1e-5;

  double m_time = 0.0;
};

}  // namespace Aperture
