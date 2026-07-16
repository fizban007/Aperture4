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

  // Initial condition helpers — call from main after env.init().
  void set_initial_dipole();
  void set_initial_deutsch();

  // Spherical-cavity TE/TM eigenmode initial condition.
  //   l, m         : angular quantum numbers (l >= 1, |m| <= l)
  //   n_root       : radial root index (1 = lowest)
  //   polarization : 'E' for TE (B_r = max angular structure),
  //                  'M' for TM (E_r = max angular structure)
  //   start_with_e : if true,  E(t=0) at maximum, B(t=0) = 0;
  //                  if false, B(t=0) at maximum, E(t=0) = 0  (default)
  // The eigenvalue ω = c k is computed from r_min and r_max at runtime.
  // Reads m_resonator_amp as the overall amplitude (default 1.0).
  void set_initial_resonator_mode(int l, int m, int n_root,
                                  char polarization, bool start_with_e);

  // Public because HIP/CUDA compilers require __device__ lambdas in public methods
  void update_explicit(double dt);
  void update_semi_implicit(double dt);

  void compute_rhs(buffer<Scalar>& E_in, buffer<Scalar>& B_in,
                   buffer<Scalar>& dE_dt, buffer<Scalar>& dB_dt);

  void apply_damping(buffer<Scalar>& E, buffer<Scalar>& B, double dt);
  // Overwrites inner-boundary E and B with the analytic solution.  E and
  // B take separate evaluation times because the leapfrog stores B at
  // half-steps (t + dt/2 after the Faraday update, i.e. dt/2 BEHIND the
  // end-of-step time): the explicit path passes time_B = time_E - dt/2,
  // while the (co-located) semi-implicit path passes time_B = time_E.
  // Evaluating both at the integer time injects an O(dt) boundary error
  // (found by the A2.0 Deutsch dt-halving study).
  void apply_inner_bc(buffer<Scalar>& E, buffer<Scalar>& B, double time_E,
                      double time_B);

  // PEC (perfect conductor) boundary on inner and outer shells:
  // zero tangential E (horizontal edges) and normal B (triangular faces)
  // on shells k = 0 and k = N_r.
  void apply_pec_bc(buffer<Scalar>& E, buffer<Scalar>& B);

 private:

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

  // Use full Deutsch retarded fields for inner BC (for convergence testing)
  bool m_use_deutsch_bc = false;

  // Use PEC (perfect conductor) boundary instead of dipole/Deutsch BC.
  // When true, apply_pec_bc() is called each step instead of apply_inner_bc().
  bool m_use_pec_bc = false;

  // Resonator mode amplitude (used by set_initial_resonator_mode)
  Scalar m_resonator_amp = 1.0;

  double m_time = 0.0;
};

using dec_field_solver_t = dec_field_solver<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
