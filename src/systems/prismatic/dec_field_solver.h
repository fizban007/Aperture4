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

  // Access to field data (for initial conditions and output)
  buffer<Scalar>& D_e() { return m_D_e; }
  buffer<Scalar>& B_f() { return m_B_f; }
  buffer<Scalar>& E_tilde() { return m_E_tilde; }
  buffer<Scalar>& H_tilde() { return m_H_tilde; }
  const buffer<Scalar>& D_e() const { return m_D_e; }
  const buffer<Scalar>& B_f() const { return m_B_f; }
  const buffer<Scalar>& E_tilde() const { return m_E_tilde; }
  const buffer<Scalar>& H_tilde() const { return m_H_tilde; }

 private:
  void apply_damping(double dt);
  void apply_inner_bc(double time);
  void set_initial_dipole();

  // Project a vector field (Bx,By,Bz) onto a face to get B_f
  Scalar project_B_on_face(int face_idx, Scalar Bx, Scalar By, Scalar Bz) const;
  // Project a vector field (Ex,Ey,Ez) onto an edge to get D_e
  Scalar project_E_on_edge(int edge_idx, Scalar Ex, Scalar Ey, Scalar Ez) const;

  // Compute dipole field at position (x,y,z) given dipole moment (mx,my,mz)
  void dipole_B(Scalar x, Scalar y, Scalar z, Scalar mx, Scalar my, Scalar mz,
                Scalar& Bx, Scalar& By, Scalar& Bz) const;

  prismatic_mesh& m_mesh;

  // Field storage
  buffer<Scalar> m_D_e;       // electric flux on edges
  buffer<Scalar> m_B_f;       // magnetic flux on faces
  buffer<Scalar> m_E_tilde;   // auxiliary E = hodge1_inv * D
  buffer<Scalar> m_H_tilde;   // auxiliary H = hodge2_inv * B

  // Physics parameters
  Scalar m_Bp = 1.0;           // dipole field strength
  Scalar m_Omega = 1.0;        // rotation angular frequency
  Scalar m_obliquity = 0.0;    // angle between rotation and magnetic axes

  // Damping layer
  int m_damping_length = 10;
  Scalar m_damping_coef = 0.05;

  double m_time = 0.0;
};

}  // namespace Aperture
