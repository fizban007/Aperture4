#pragma once

#include "core/buffer.hpp"
#include "core/exec_tags.h"
#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_mesh_metric.h"
#include <string>

namespace Aperture {

// =========================================================================
// Whitney-form Galerkin Hodge stars on the prismatic mesh (GRPIC_PLAN B1b).
//
// C++ port of python/hodge_lab_whitney.py, the validated reference: both
// diagonal (mass-lumped) Hodge stars are O(1)-defective for a != 0 because
// γ_rφ != 0 breaks the primal/dual orthogonality they assume (B1).  The
// fix is the Galerkin mass matrix of the lowest-order Whitney forms on the
// triangular prism (wedge) element, triangle treated as AFFINE in (θ, φ)
// and linear in ζ = (r − r_k)/Δr.  The affine-chart treatment is adopted
// deliberately (it is what the lab validated); it differs from the mesh's
// curved-cell DOF quadrature at O(h²), consistent at lowest order.
//
//   1-forms, 9 DOF (covariant (r, θ, φ)), w_ij = λ_i∇λ_j − λ_j∇λ_i:
//     horizontal bottom/top:  w_ij·(1−ζ) ,  w_ij·ζ      (signed by
//                             tri_edge_signs into the global edge DOF)
//     vertical at vertex i:   W_r = λ_i/Δr
//   2-forms, 5 DOF (DENSITIZED proxy Bd^i = √γ B^i):
//     bottom/top tri:         Bd^r = (1−ζ)/A_c ,  ζ/A_c
//     rect on edge ij:        Bd^(θ,φ) = −rot90(w_ij)/Δr   (triangle RT0;
//                             the −1 is the mesh's rect-face orientation
//                             relative to the tri_edge_signs RT0 basis —
//                             getting it wrong reads as a convergent
//                             scheme with a +38.8% flat energy offset)
//
//   M1 [a,b] = ∫ γ^{ij} W_a,i W_b,j √γ  dr dθ dφ          (no lapse)
//   M1α[a,b] = same integrand × α                          (lapse INSIDE
//   M2α[a,b] = ∫ γ_ij Bd_a^i Bd_b^j α/√γ  dr dθ dφ         the quadrature)
//   C1 [e,f] = ∫ γ^{il} (β×B_f)_l W_e,i √γ  dr dθ dφ
//            = ∫ √γ β^r [ (γ^{rφ}W_r + γ^{φφ}W_φ) Bd^θ − γ^{θθ}W_θ Bd^φ ]
//
// The first three are Gram matrices of real inner products ⇒ symmetric
// positive definite by construction, which is the stability condition:
// with
//   E_aux = M1⁻¹ (M1α D_p + C1 B)     and    H_aux = M2α B + C1ᵀ D_p
// the semi-discrete generator's CROSS terms cancel exactly in the energy
// norm ½ D_pᵀ M1α D_p + ½ Bᵀ M2α B (they are transposes of one another
// because d1t = d1ᵀ and every factor is symmetric).  The lapse must NOT
// be applied per-element outside the matrices: diag(α)·M is asymmetric
// the moment M has off-diagonal entries, and a per-face α applied after
// the fact measured +50% and destroyed convergence.
//
// C1 is the Whitney treatment of the SHIFT cross terms (E = αD + β×B,
// H = αB − β×D).  The consistent Ampère pairing C2[f,e] = ⟨β×D_e, B_f⟩₂
// equals −C1ᵀ analytically (both integrands are the scalar triple
// product ε(β,·,·) up to one transposition), hence H_aux = M2α B + C1ᵀ D_p
// — the shift pair is mutually adjoint BY CONSTRUCTION.  This matters:
// splicing the legacy √γ-weighted local averaging shift onto the Whitney
// base maps fails in both directions (measured 2026-08-01): feeding it
// M1⁻¹D̃ produces a slow equatorial instability straddling the horizon
// (e-fold ~16 M at L4, rate growing with L — the pair loses adjointness
// where β^r and γ_rφ peak), while feeding it hodge1_inv·D̃ is stable but
// O(1)-inconsistent (hodge1_inv·M1 ≠ I by 13–25% near the horizon) and
// parks the relaxed state +16% off with a persistent ±1% wobble.  The
// legacy averaging remains only on the diagonal (use_whitney_hodge =
// false) path.
//
// Quadrature: 3-point edge-midpoint triangle rule (degree 2) × 2-point
// Gauss in ζ.  Quadrature points never touch the axis (edge midpoints of
// a triangle with a polar vertex all have θ ≥ O(h)), so the genuine
// 1/sin²θ in γ^φφ is finite; all other sin θ cancellations are done
// analytically inside the Metric's gu_* accessors.
//
// Assembly runs on the host in the mesh's native Scalar precision and is
// uploaded to the device afterwards.  Storage is CSR; M1 and M1α share
// one sparsity pattern.
//
// Validated reference numbers (lab, Kerr a = 0.998, Wald):
//   Gauss flux through r = const (truth exactly 0): order +1.90/+1.98/+2.00
//   near-horizon magnetic energy: +0.63% / +0.15% / +0.039% at L3/4/5
// versus diagonal stars: flux stuck at −16.6, energy at a ~20% floor.
// =========================================================================

struct prismatic_whitney_hodge_ptrs {
  // M1 / M1α: edge × edge, shared CSR pattern.
  const int* m1_row_ptr;      // [N_edges + 1]
  const int* m1_col_idx;
  const Scalar* m1_val;       // no lapse
  const Scalar* m1a_val;      // lapse inside the quadrature
  const Scalar* m1_jacobi;    // 1 / diag(M1)  (Jacobi preconditioner)

  // M2α: face × face.
  const int* m2_row_ptr;      // [N_faces + 1]
  const int* m2_col_idx;
  const Scalar* m2a_val;      // lapse inside the quadrature

  // C1 (edge × face) and its exact transpose C1ᵀ (face × edge): the
  // Galerkin shift coupling ⟨β×B_f, W_e⟩₁ — see the class comment.
  const int* c1_row_ptr;      // [N_edges + 1]
  const int* c1_col_idx;
  const Scalar* c1_val;
  const int* c1t_row_ptr;     // [N_faces + 1]
  const int* c1t_col_idx;
  const Scalar* c1t_val;

  int N_edges;
  int N_faces;
};

class prismatic_whitney_hodge {
 public:
  prismatic_whitney_hodge() = default;
  ~prismatic_whitney_hodge() = default;

  // Host assembly.  Requires a fully built mesh (topology + sphere tables
  // on the host; compute_metric need not have run — the metric is
  // evaluated directly at the quadrature points).  Templated on the
  // concrete metric type; explicit instantiations for
  // flat_spherical_metric and ks_spherical_metric live in the .cpp.
  template <typename Metric>
  void build(const prismatic_mesh_metric& mesh, const Metric& met);

  bool ready() const { return m_ready; }

  // Debug/validation dump of all CSR arrays to HDF5.
  void dump(const std::string& path) const;

  prismatic_whitney_hodge_ptrs host_ptrs() const;
  prismatic_whitney_hodge_ptrs get_ptrs(exec_tags::host) const {
    return host_ptrs();
  }
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  prismatic_whitney_hodge_ptrs dev_ptrs() const;
  prismatic_whitney_hodge_ptrs get_ptrs(exec_tags::device) const {
    return dev_ptrs();
  }
  void copy_to_device();
#endif

  buffer<int> m1_row_ptr, m1_col_idx;
  buffer<Scalar> m1_val, m1a_val, m1_jacobi;
  buffer<int> m2_row_ptr, m2_col_idx;
  buffer<Scalar> m2a_val;
  buffer<int> c1_row_ptr, c1_col_idx;
  buffer<Scalar> c1_val;
  buffer<int> c1t_row_ptr, c1t_col_idx;
  buffer<Scalar> c1t_val;

 private:
  bool m_ready = false;
  int m_N_edges = 0;
  int m_N_faces = 0;
};

}  // namespace Aperture
