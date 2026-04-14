#pragma once

#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_mesh_metric.h"
#include "utils/nonown_ptr.hpp"

namespace Aperture {

// DEC field solver for any spherical-metric spacetime with a purely
// radial shift (β^θ = β^φ = 0).  Examples: flat space (trivially),
// Schwarzschild, Kerr-Schild.
//
// The solver uses the metric mesh's precomputed Hodge stars (already
// metric-weighted) and per-element lapse α and shift √γ β^r.  The shift
// cross-coupling terms vanish on vertical edges and triangular faces
// because their tangents / normals are aligned with the radial shift
// direction; they are computed only on horizontal edges and rectangular
// faces via a scalar triple product that doesn't require reconstructing
// the full magnetic or electric 3-vector.
template <typename ExecPolicy>
class dec_field_solver_gr_ks : public system_t {
 public:
  static std::string name() { return "dec_field_solver_gr_ks"; }

  dec_field_solver_gr_ks(prismatic_mesh_metric& mesh);
  ~dec_field_solver_gr_ks() = default;

  void register_data_components() override;
  void init() override;
  void update(double dt, uint32_t step) override;

  // Initial condition: uniform B_z (Wald background on a BH spacetime)
  void set_initial_wald(Scalar B0 = 1.0);

  // Public for GPU lambda access
  void update_explicit(double dt);
  void update_semi_implicit(double dt);

  // Compute 3+1 right-hand side:
  //   dB[f]/dt = -sum_e d1[f,e] * E_aux[e]
  //   dD[e]/dt = hodge1_inv[e] * (sum_f d1t[e,f] * hodge2[f] * H_aux[f] - J[e])
  //
  // E_aux and H_aux encode the full constitutive relations (lapse scaling
  // + radial shift cross-coupling).  For flat/no-gravity metrics these
  // reduce to D and B, recovering the standard flat-space update.
  void compute_rhs(buffer<Scalar>& D_in, buffer<Scalar>& B_in,
                   buffer<Scalar>& dD_dt, buffer<Scalar>& dB_dt);

  void apply_damping(buffer<Scalar>& D, buffer<Scalar>& B, double dt);
  void apply_horizon_bc(buffer<Scalar>& D, buffer<Scalar>& B);

 private:
  prismatic_mesh_metric& m_mesh;

  // Shared field data (owned by env)
  nonown_ptr<prismatic_edge_field> m_D;
  nonown_ptr<prismatic_face_field> m_B;
  nonown_ptr<prismatic_edge_field> m_J;

  // Auxiliary fields for constitutive relations (owned by solver)
  buffer<Scalar> m_E_aux;
  buffer<Scalar> m_H_aux;

  // Temporary buffers for semi-implicit iteration
  buffer<Scalar> m_tmp_D, m_tmp_B;
  buffer<Scalar> m_dD_dt, m_dB_dt;
  buffer<Scalar> m_dD_dt_new, m_dB_dt_new;

  // Damping layer (outer boundary absorption)
  int m_damping_length = 10;
  Scalar m_damping_coef = 0.05;

  // Horizon damping: smoothly ramp fields to zero between r_horizon_damp
  // and r_horizon_inner (where the inner value is typically the outer
  // horizon radius).  Set r_horizon_damp <= 0 to disable.
  Scalar m_r_horizon_damp = 0.0;
  Scalar m_r_horizon_inner = 0.0;

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
