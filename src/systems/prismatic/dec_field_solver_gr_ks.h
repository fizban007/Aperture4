#pragma once

#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_mesh_gr_ks.h"
#include "utils/nonown_ptr.hpp"

namespace Aperture {

template <typename ExecPolicy>
class dec_field_solver_gr_ks : public system_t {
 public:
  static std::string name() { return "dec_field_solver_gr_ks"; }

  dec_field_solver_gr_ks(prismatic_mesh_gr_ks& mesh);
  ~dec_field_solver_gr_ks() = default;

  void register_data_components() override;
  void init() override;
  void update(double dt, uint32_t step) override;

  // Initial condition: Wald solution (uniform B_z on Kerr background)
  void set_initial_wald(Scalar B0 = 1.0);

  // Public for GPU lambda access
  void update_explicit(double dt);
  void update_semi_implicit(double dt);

  // Compute GR 3+1 right-hand side:
  //   dB[f]/dt = -sum_e d1[f,e] * E_aux[e]
  //   dD[e]/dt = sum_f d1t[e,f] * H_aux[f] * hodge2[f]  * hodge1_inv[e] - J[e]
  //
  // E_aux and H_aux encode the constitutive relations with lapse, metric,
  // and shift cross-coupling.
  void compute_rhs(buffer<Scalar>& D_in, buffer<Scalar>& B_in,
                   buffer<Scalar>& dD_dt, buffer<Scalar>& dB_dt);

  void apply_damping(buffer<Scalar>& D, buffer<Scalar>& B, double dt);
  void apply_horizon_bc(buffer<Scalar>& D, buffer<Scalar>& B);

 private:
  prismatic_mesh_gr_ks& m_mesh;

  // Shared field data (owned by env)
  nonown_ptr<prismatic_edge_field> m_D;  // electric displacement 1-cochain
  nonown_ptr<prismatic_face_field> m_B;  // magnetic flux 2-cochain
  nonown_ptr<prismatic_edge_field> m_J;  // current density

  // Auxiliary fields for constitutive relations (owned by solver)
  buffer<Scalar> m_E_aux;   // [N_edges] line integrals of E_aux
  buffer<Scalar> m_H_aux;   // [N_faces] surface integrals of H_aux

  // Temporary buffers for semi-implicit iteration
  buffer<Scalar> m_tmp_D, m_tmp_B;
  buffer<Scalar> m_dD_dt, m_dB_dt;
  buffer<Scalar> m_dD_dt_new, m_dB_dt_new;

  // Physics parameters
  Scalar m_a = 0.0;  // BH spin (read from mesh)

  // Damping layer
  int m_damping_length = 10;
  Scalar m_damping_coef = 0.05;

  // Horizon damping: damp fields for edges/faces with r < r_horizon_damp
  Scalar m_r_horizon_damp = 0.0;

  // Update toggles
  bool m_update_d = true;
  bool m_update_b = true;

  // Semi-implicit parameters
  bool m_use_implicit = false;
  Scalar m_beta = 0.55;
  int m_implicit_iters = 4;

  double m_time = 0.0;
};

using dec_field_solver_gr_ks_t =
    dec_field_solver_gr_ks<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
