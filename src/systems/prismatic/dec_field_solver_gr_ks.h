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
//
// Inner boundary condition: for black hole spacetimes, place r_min
// inside the outer horizon r_+.  The DEC stencil is self-closing at the
// inner boundary (no explicit BC needed) and GR causality guarantees
// that anything at r < r_+ cannot influence r > r_+.  An optional
// in-horizon safety damping (apply_horizon_damping) is available for
// suppressing numerical leakage from discretization-level superluminal
// modes, but is disabled by default.
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

  // Faraday half-step: compute E_aux from (D, B), then dB = -d1·E_aux.
  // Leaves the result in dB_out; does not modify B.
  void compute_dB_dt(buffer<Scalar>& D_in, buffer<Scalar>& B_in,
                     buffer<Scalar>& dB_out);

  // Ampère half-step: compute H_aux from (D, B), then dD = h1inv·(d1t·h2·H_aux - J).
  // Leaves the result in dD_out; does not modify D.
  void compute_dD_dt(buffer<Scalar>& D_in, buffer<Scalar>& B_in,
                     buffer<Scalar>& dD_out);

  void apply_damping(buffer<Scalar>& D, buffer<Scalar>& B, double dt);

  // Optional safety damping inside the horizon.  The primary "inner BC"
  // for this solver is causal disconnection: with r_min placed inside the
  // outer horizon, the DEC stencil is self-closing (no explicit BC is
  // needed) and anything unstable at r < r_+ cannot physically propagate
  // outward to the domain of interest.  This method provides a light
  // numerical insurance layer against any superluminal-mode leakage at
  // the discretization level.  Disabled by default (r_horizon_damp = 0).
  void apply_horizon_damping(buffer<Scalar>& D, buffer<Scalar>& B);

  // Inner boundary condition (mirrors the 2D GR-KS solver's treatment):
  // overwrites the innermost-shell field values (shell 0 for horizontal
  // edges/tri faces, slab 0 for vertical edges/rect faces) using a
  // "ghost = interior + (D0_interior - D0_ghost)" extrapolation from
  // shell 1.  This is a Neumann-like BC on the perturbation δ = field -
  // background that preserves the background gradient across the
  // innermost layer, preventing spurious gradients from driving
  // in-horizon instabilities.
  void apply_inner_boundary(buffer<Scalar>& D, buffer<Scalar>& B);

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

  // Background fields: the outer damping layer relaxes (D, B) toward
  // these target values instead of zero.  Populated by set_initial_wald
  // (and any future setter).  If m_has_background = false, damping falls
  // back to damping toward zero (flat-space convention).
  buffer<Scalar> m_D_bg, m_B_bg;
  bool m_has_background = false;

  // Damping layer (outer boundary absorption)
  int m_damping_length = 10;
  Scalar m_damping_coef = 0.05;

  // Optional horizon safety damping (disabled by default).
  //
  // When r_horizon_damp > 0, fields are smoothly ramped to zero for
  // elements with r_horizon_inner ≤ r < r_horizon_damp, using a quadratic
  // profile that is 1 at the outer edge and 0 at r_horizon_inner.  This
  // is *not* the physical inner boundary condition — causal disconnection
  // handles that.  Its only purpose is to suppress numerical artifacts
  // that might otherwise accumulate inside the horizon due to
  // discretization-level superluminal modes.
  //
  // Leave at 0 to rely purely on causal protection (recommended default).
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
