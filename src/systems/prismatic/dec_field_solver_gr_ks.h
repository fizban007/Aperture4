#pragma once

#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_mesh_metric.h"
#include "utils/nonown_ptr.hpp"

namespace Aperture {

// STATUS (2026-07): SHELVED.  The prismatic effort is focused on flat-space
// NS magnetospheres (see ROADMAP_NS_MAGNETOSPHERE.md); 3D GR science is
// covered by the traditional-grid solver on the develop branch.  This
// solver is kept compiling but is not maintained or validated — do not
// build new work on it without revisiting the roadmap.
//
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
// that anything at r < r_+ cannot influence r > r_+.  An exponential
// inner damping layer (apply_inner_damping) over the innermost shells
// absorbs any numerical leakage from discretization-level superluminal
// modes inside the horizon.
template <typename ExecPolicy>
class dec_field_solver_gr_ks : public system_t {
 public:
  static std::string name() { return "dec_field_solver_gr_ks"; }

  dec_field_solver_gr_ks(prismatic_mesh_metric& mesh);
  ~dec_field_solver_gr_ks() = default;

  void register_data_components() override;
  void init() override;
  void update(double dt, uint32_t step) override;

  // Analytic Kerr-Schild Wald IC.
  //
  //   a_field — spin parameter of the analytic Maxwell field (A_μ, D^i
  //             formulas).  Sets the Wald field pattern: a_field = 0
  //             gives Schwarzschild Wald (uniform B_z), a_field > 0
  //             gives Kerr Wald with frame-dragging in A_t.
  //
  // The background-metric spin used for index lowering/raising and for
  // the KS normal-observer projection into D^i is taken from m_spin
  // (populated in init() from the "bh_spin" config key — must match the
  // spin passed to compute_metric() on the underlying mesh).
  //
  // When a_field == m_spin this sets up the stationary Kerr-Wald
  // equilibrium.  When a_field != m_spin the IC is off-shell (e.g.,
  // Schwarzschild Wald Maxwell field on a rotating Kerr background),
  // useful for relaxation tests.  Stored as the background for the
  // inner/outer damping layers.
  //
  // Both integrations use 10-point Gauss quadrature along the primal
  // edge in Cartesian space, so curved-edge effects on the icosphere
  // are handled consistently with the Hodge construction.
  void set_initial_kerr_wald(Scalar a_field, Scalar Bp = 1.0);

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

  // Inner damping layer — exponential absorption of the perturbation
  // δ = field − background over the innermost m_inner_damping_length
  // radial shells.  Mirrors the outer damping in structure but ramps
  // from strongest at k=0 down to weakest at the outer edge of the
  // layer.  Intended to sit fully inside the horizon: place enough
  // mesh shells below r_+ (via a correspondingly low r_min) that the
  // damping region is causally disconnected from the physics domain.
  // Applied after each time step in addition to apply_inner_boundary.
  void apply_inner_damping(buffer<Scalar>& D, buffer<Scalar>& B, double dt);

  // Inner boundary condition (mirrors the 2D GR-KS solver's treatment):
  // overwrites the innermost-shell field values (shell 0 for horizontal
  // edges/tri faces, slab 0 for vertical edges/rect faces) using a
  // "ghost = interior + (D0_interior - D0_ghost)" extrapolation from
  // shell 1.  This is a Neumann-like BC on the perturbation δ = field -
  // background that preserves the background gradient across the
  // innermost layer, preventing spurious gradients from driving
  // in-horizon instabilities.
  void apply_inner_boundary(buffer<Scalar>& D, buffer<Scalar>& B);

  // Outer boundary condition: pins D on the outermost horizontal
  // (triangular-ring) edges and B on the outermost triangular faces to
  // their stored background values, freezing them at the IC.  These
  // outermost elements have their shift-term reconstruction biased by
  // one-sided averaging (no rect face above them), which seeds polar
  // artifacts.  Pinning them sidesteps the biased reconstruction and
  // serves as a hard Dirichlet BC for the asymptotic Wald background.
  // Requires m_has_background = true; no-op otherwise.
  void apply_outer_boundary(buffer<Scalar>& D, buffer<Scalar>& B);

  // Diagnostic: populate m_E_aux, m_H_aux from the current D, B state
  // (running one Faraday and one Ampère constitutive-relation build),
  // then write them along with D, B to an HDF5 file so the raw values
  // can be compared against the analytic expectations off-line.
  void dump_aux_fields(const std::string& path);

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

  // Background-metric spin (Kerr-Schild).  Read from the "bh_spin" config
  // key in init(); must match the spin passed to compute_metric() on the
  // underlying mesh.  Used by set_initial_kerr_wald for consistent index
  // lowering and KS normal-observer projection.  Defaults to 0
  // (Schwarzschild / flat) if the key is absent.
  Scalar m_spin = 0;

  // Damping layer (outer boundary absorption)
  int m_damping_length = 10;
  Scalar m_damping_coef = 0.05;

  // Inner damping layer (inside-horizon absorption)
  int m_inner_damping_length = 0;
  Scalar m_inner_damping_coef = 0.5;

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
