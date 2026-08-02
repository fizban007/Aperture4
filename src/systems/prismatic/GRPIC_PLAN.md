# Plan: GR PIC on the prismatic mesh — plasma-filled Wald

**Status: PLANNING (opened 2026-08-01).** Prerequisite done: the GR field
solver is un-shelved and validated in vacuum (commit `f5f00a9f1`; Meissner
flux expulsion, +1.99 convergence over L3–L6).

**Goal.** Reproduce §3.3 "Plasma-filled Wald Solution" of *Introducing
APERTURE* (arXiv:2503.04558, Chen, Luepker & Yuan 2025) on the prismatic
icosahedral mesh: start from the vacuum rotating Wald solution, inject e±
pairs volumetrically where the plasma is magnetized and unscreened, and
watch field lines threading the ergosphere get dragged in, forming an
equatorial current sheet.

**This is a port, not an invention.** The reference implementation is
Aperture's own traditional-grid GRPIC path, and
`problems/gr_3d_kerr_schild/src/main_wald.cpp` *is* the paper's §3.3 setup.
Every piece of GR physics below already exists and is validated against
`grid_ks_t`; the work is re-expressing it against the prismatic mesh. When in
doubt, read the traditional-grid version first and match its physics
exactly — divergence from it is a bug, not a design choice.

---

## Reference map — where each piece already lives

| Need | Traditional-grid reference |
|---|---|
| GR pusher (geodesic + EM, covariant `u_i`, FIDO) | `systems/policies/coord_policy_gr_ks_sph.hpp` |
| Geodesic RHS | `systems/physics/geodesic_ks.hpp` |
| KS metric functions | `systems/physics/metric_kerr_schild.hpp` (`Metric_KS`) |
| J/ρ metric normalization | `coord_policy_gr_ks_sph.hpp::process_J_Rho` |
| Paper's injection criterion | `problems/gr_3d_kerr_schild/src/injector.cpp` |
| Injected-momentum tetrad transform | `problems/gr_3d_kerr_schild/src/density_floor_injector.cpp` |
| GR moments | `systems/compute_moments_gr_ks*` |
| Whole §3.3 driver | `problems/gr_3d_kerr_schild/src/main_wald.cpp` |
| Test-particle validation | `tests/test_gr_ks_sph.cpp` (Carter constant) |

Prismatic assets that carry over unchanged: the distributed PIC stack
(migration, halo/reduce, checkpointing, MPI), the injector framework
(`prismatic_ptc_injector.hpp`, `prismatic_surface_injector.h`), the Whitney
deposit, and the vertex-recovery B-gather.

---

## Design decisions (settle these before writing code)

### D1 — Work in Kerr–Schild SPHERICAL (r, θ, φ). Not Cartesian.

The prismatic mesh embeds vertices with `sph_to_cart`, `x = r sinθ cosφ`
(`prismatic_mesh.cpp:17`), and applies `ks_spherical_metric` =
`Metric_KS::g_11/g_22/g_33/g_13` in (r, θ, φ). The Cartesian coordinates are
a **topological embedding** used for cell lookup and Whitney interpolation —
they are not a physical chart.

**TRAP.** `systems/physics/metric_ks_cartesian.hpp` exists, is unit-tested
(`tests/test_metric_ks_cartesian.cpp`), and exposes exactly the 3+1
quantities a pusher wants (α, β^i, γ_ij, γ^ij, √γ) from Cartesian input. It
looks like a drop-in. **It is not.** It implements the true Kerr–Schild
Cartesian chart (`l_x = (r x + a y)/(r²+a²)`), which differs from the mesh's
naive embedding by a spin-dependent azimuthal twist and oblateness. Using it
with mesh coordinates is wrong for a ≠ 0 and would degrade smoothly to
"looks plausible, converges to the wrong answer". It is currently referenced
by nothing but its own test — leave it that way unless someone deliberately
re-charts the mesh (which would invalidate the field solver's metric).

Consequence: the pusher operates on (r, θ, φ) obtained from the particle's
position, matching the chart the field solver discretizes.

### D2 — Momentum variable: covariant `u_i`, as on the traditional grid.

`coord_policy_gr_ks_sph` stores covariant `u_i` and converts to FIDO only
where needed. The prismatic kernel currently stores Cartesian `(px,py,pz)`
for Boris and `(u_par, mu)` for GCA
(`prismatic_ptc_update_kernel.hpp:368-369`). GR adds a third representation.
Keep the existing flag-based dispatch pattern; do not try to unify.

### D3 — Deposit convention: `J̃` dual-2, unchanged. **VERIFY FIRST (P0).**

The GR Ampère update is *purely topological on the dual mesh*:

```
dD̃[e] = Σ_f d1t[e,f] · H_aux[f] − J̃[e]      (impl:292-305, no Hodge star)
```

versus the flat solver's `E_e += dt · h1inv · (curl_H − J_e)`
(`dec_solver_dist.h:496`). The updater already tags its output
`m_J->set_edge_kind(EdgeCochainKind::dual_2)`
(`prismatic_ptc_updater_impl.hpp:49`) — the same kind as `D̃`.

**Hypothesis (to be confirmed in P0):** discrete charge conservation is a
combinatorial property of the Whitney trajectory split and is therefore
metric-independent, so it carries over to GR *unchanged*; the only open
question is the physical normalization (which α/√γ factors relate the
deposited `J̃` to the GR Ampère source). If that hypothesis holds, the
deposit needs **no code change at all**, and GR enters solely through the
pusher's coordinate velocity `dx^i/dt` (which contains −β^i).

This single question sizes the project. Settle it on paper + a numerical
check before writing any deposit code.

---

## Gaps

Ordered by size. G1 dominates.

- **G1 — Pusher is flat-space only.** `prismatic_ptc_update_kernel.hpp` does
  a flat Boris push on Cartesian momentum. Needs geodesic + EM force on
  covariant `u_i`. `geodesic_ks.hpp` has the math but `#include`s
  `systems/grid_ks.h`, so the formulas must first be lifted into a
  grid-free header. GCA-in-GR (`coord_policy_gr_ks_sph_gca.hpp`) is
  **deferred** — full-orbit first.
- **G2 — Chart trap.** See D1. Costs nothing if respected, costs everything
  if not.
- **G3 — Field gather returns the wrong object.** The GR solver registers
  `"E"` but stores **D̃** (dual-2). `interpolate_fields` never consults
  `edge_kind` or `hodge1_inv` — it assumes a primal 1-cochain. Silent
  failure mode: fields wrong by a Hodge factor with no error. The
  `edge_kind` mechanism already exists (sph output honors it); wire it into
  the gather. Also the vertex-recovery B-gather rescales weights by
  `(r_ref/r_k)²` assuming flat geometric shells — check that against the
  metric mesh.
- **G4 — Deposit normalization.** See D3. Possibly zero work; possibly a
  factor sweep. P0 decides.
- **G5 — Injection is flat.** The criterion machinery exists
  (`inj_eb_threshold`, `inj_min_sigma`) and is structurally the paper's
  `sigma > sigma_thr && |D·B|/B² > ε_{D·B}`, but needs: **D**·B not E·B,
  with the metric dot product; the Crinquand tetrad transform for injected
  momenta; √γ in the weight; a GR σ.
- **G6 — No inner particle absorber.** Only `ptc_absorb_radius` (outer)
  exists. Particles crossing the horizon must be removed. Small.
- **G7 — Moments are flat.** `rho_abs` / `gamma_wsum` assume flat; γ is
  frame-dependent in GR (needs `u_0` / FIDO). Coupled to G5, because the
  injector's own σ criterion reads these.
- **G8 — No GR PIC driver.** `problems/prismatic_wald/src/main.cpp`
  registers only solver + exporters. Needs an `ns_rotator`-style stack and a
  mesh built with both the metric and the particle bundle.

---

## Phasing

Each phase ends with a cheap, decisive check. Do not carry an unvalidated
phase forward.

### P0 — Derivation + deposit audit — **DONE 2026-08-01. Result: D3 confirmed, but a BLOCKER was found.**

**D3 confirmed — the deposit needs no normalization change.** Take the dual
divergence of the GR Ampère update `dD̃ = d1t·H_aux − J̃`: the curl term
annihilates, leaving `∂_t(div D̃) = −div J̃`. With `div D̃ = ρ̃` (Gauss) that
is the **coordinate-time** continuity equation, so `J̃[e]` must be charge per
unit coordinate time through the metric dual face — **no α, no √γ**. The
Whitney deposit already produces exactly that, and its conservation property
is combinatorial (barycentric weights only), hence metric-independent. GR
enters solely through the pusher's coordinate velocity `dx^i/dt` (which
contains −β^i).

#### BLOCKER B1 — BOTH Hodge stars are O(1)-defective for a != 0

Auditing `div D~ = rho~` on the metric mesh turned up a resolution-independent
error, and following it through showed the same defect in the OTHER Hodge too.

**Mechanism.** In Kerr-Schild *spherical* coordinates `g_rphi != 0`, so a
coordinate-radial primal edge is not metric-orthogonal to its r=const dual
face, and a tri face's normal (`dr`) is not metric-parallel to its dual edge
(`d_r`).  The diagonal (mass-lumped) stars assume exactly that orthogonality.
Each star is wrong in TWO ways: an omitted off-diagonal term, and a scalar
prefactor `1/sqrt(1-k^2)` (hodge1) or `sqrt(1-k^2)` (hodge2) with
`k^2 = g_rphi^2/(g_rr g_phph)`.  Near the horizon `k^2 ~ 0.70`.

**hodge1 — measured.** Discrete flux through r=const (continuum truth is
EXACTLY 0, verified by quadrature; vacuum):

| L | r=1.106 | r=3.09 | r=8.016 |
|---|---|---|---|
| 3 | -1.4887e+1 | -1.6356e+1 | -1.6726e+1 |
| 6 | -1.4681e+1 | -1.6194e+1 | -1.6615e+1 |

Converges to a nonzero constant, ~8% of the cap-flux scale.  It matches the
functional the diagonal star actually forms,
`\oint (D_r/sqrt(g_rr)) sqrt(g_thth g_phph)`, to 4 digits (-14.65/-16.19/-16.61)
-- the scheme converges faithfully to the wrong operator.

**hodge2 — measured.** Pointwise tri-face defect reaches 249% at r=1.106
*including a sign flip* (H_full=-1.24e-1 vs H_diag=+1.84e-1); 37.7% of tri
faces exceed 10% error at L3.  Integrated: near-horizon magnetic energy over
r in [1.1,1.5], ground truth `E_true = 5.44547e+00` by quadrature:

| L | E(diagonal) | error | E(Galerkin) | error |
|---|---|---|---|---|
| 3 | 7.179e+0 | +31.8% | 5.792e+0 | +6.37% |
| 4 | 6.850e+0 | +25.8% | 5.565e+0 | +2.19% |
| 5 | 6.705e+0 | +23.1% | 5.466e+0 | **+0.38%** |

The diagonal star converges to a **~20% floor** (fit `F + A h` gives F ~ 20%).

**RESOLUTION: Galerkin.** Built as `M = sum_p V_p L^T Gamma L` with L a local
reconstruction from the element's cochain DOFs -- symmetric PSD by
construction (measured symmetry defect 1.09e-16).  Both fixed:

- M1, Gauss flux: 7.05e-2 / 2.00e-2 / 5.32e-3 at L3/4/5, order **+1.91,
  +1.98, +2.00** (vs a diagonal that does not converge at all).
- M2, energy: converges to `E_true`, roughly 2nd order (table above).

Cost is not a discriminator: Jacobi-preconditioned `cond = 1.15`, PCG
converges in **7 iterations** to 1e-10 (3 ms at L3, 63k unknowns), against 5
Picard iterations already in the semi-implicit step.  `lambda_max` is
unchanged to 6 digits, so **no CFL penalty**; conditioning +0.08%.

**Stability requirement (surfaced by the probe).** Fold the lapse in
SYMMETRICALLY, `sqrt(alpha) H sqrt(alpha)`, never `diag(alpha) @ H`.  The
latter is symmetric only while H is diagonal; with any off-diagonal term it
breaks symmetry, and with it energy conservation.  Costs nothing on the
diagonal part.  With symmetry the generator is skew-adjoint in the energy
norm (energy-rate residual 2.8e-17); an asymmetric control is caught at
1.0e-5.  This also re-reads the roadmap's earlier "reconstruction Hodge is
unconditionally unstable" verdict: its recorded diagnosis was
"single-anchor row ASYMMETRY -> strongly non-normal", i.e. a fixable cause,
not a property of wide stencils.

#### B1b — REFERENCE IMPLEMENTATION: proper Whitney forms (validated)

`python/hodge_lab_whitney.py` is the reference a C++ port should reproduce.
Lowest-order triangular prism (wedge) element, triangle affine in
(theta, phi), linear in `zeta = (r-r_k)/dr`:

- **1-forms, 9 DOF** (covariant components in (r,th,ph)), with
  `w_ij = lam_i grad lam_j - lam_j grad lam_i`:
  - horizontal bottom / top: `w_ij * (1-zeta)` , `w_ij * zeta`
  - vertical at vertex i:    `W_r = lam_i / dr`
- **2-forms, 5 DOF**, using the DENSITIZED proxy `Bd^i = sqrt(g) B^i` -- that
  is what has polynomial components and unit face flux:
  - bottom / top tri: `Bd^r = (1-zeta)/A_c` , `zeta/A_c`
  - rect on edge ij:  `Bd^(th,ph) = rot90(w_ij)/dr`   (triangle RT0)

```
M1[a,b] = \int g^{ij} W_a,i W_b,j sqrt(g)      dr dth dph
M2[a,b] = \int g_ij  Bd_a^i Bd_b^j alpha/sqrt(g) dr dth dph
```

Both are Gram matrices of a real inner product => SYMMETRIC POSITIVE
DEFINITE by construction, which is exactly the stability condition of B1.
Quadrature used: 3-point edge-midpoint triangle rule (degree 2) x 2-point
Gauss in zeta.

**Validated, both stars, both non-blind projections:**

| L | flux r=1.106 | r=3.093 | r=8.016 | energy err |
|---|---|---|---|---|
| 3 | 4.579e-2 | -6.443e-2 | -6.961e-2 | +0.631% |
| 4 | 1.335e-2 | -1.495e-2 | -1.733e-2 | +0.153% |
| 5 | 3.583e-3 | -3.779e-3 | -4.330e-3 | **+0.039%** |
| **order** | **+1.90** | **+1.98** | **+2.00** | **~ +2** |

versus diagonal stars: flux stuck at -16.6 (non-convergent), energy stuck at
a ~20% floor.  Runtime 39 s at L5 in Python -- no port needed for the lab.

**THREE THINGS THE PORT WILL GET WRONG SILENTLY IF NOT TOLD:**

1. **Rect-face orientation is -1** relative to an RT0 basis built from
   `tri_edge_signs`.  Getting it wrong gives +38.8% energy error that is
   FLAT IN L -- it reads as a convergent scheme with a bad constant, not as
   a sign error.  (Independently confirmed: stored `B[f]` / predicted flux
   has median -0.99977 on rect faces, +0.994 on tri faces.)
2. **alpha must be INSIDE the quadrature.**  Applying it per-face afterwards
   gave +50% and destroyed convergence entirely.  It must also enter
   symmetrically, `sqrt(alpha) M sqrt(alpha)` -- `diag(alpha) @ M` is
   symmetric only while M is diagonal, and symmetry is the stability
   condition.
3. **Cancel sin(theta) analytically at the axis** in `g^phph` and `sqrt(g)`,
   exactly as `prismatic_sph_output.cpp` already does.  Otherwise the polar
   prisms are 0/0.  Regular forms: `g^rr = P/(g_rr rho2)`, `g^rphi = a/rho2`,
   `g^phph = g_rr/(rho2 sin^2)`, with `P = r^2+a^2+Z a^2 sin^2`.

**Two caveats on the reference itself:**

- The triangle is treated as affine in (theta, phi), whereas the mesh's
  actual cells are geodesic on the sphere and its stored DOFs come from
  metric quadrature over the curved cells.  That is an O(h^2) mismatch in
  the DOF definitions -- consistent at lowest order, but NOT identical to
  the code's conventions.  A C++ implementation should either integrate over
  the curved cells or adopt this as the definition deliberately.
- Score against the ground truth over the ACTUAL radial extent the included
  prisms cover ([1.1057, 1.4997] here), not the nominal band.  Using the
  nominal [1.1, 1.5] charges the scheme ~1% for a band-selection difference
  it did not make, and makes the error appear to GROW with L.

**Inversion.**  The solver needs `M^-1` (state is `D~`, `E_aux` needs
`D_primal`).  Jacobi-preconditioned `cond = 1.15`; PCG converges in 7
iterations to 1e-10.  Cheap against the 5 Picard iterations already present.

#### B1a — the diagnostics have null spaces (why none of this showed up)

This is the important methodological finding, and it invalidates the way
some already-committed numbers read.

- **The on-shell fixed-point residual is blind.** Scored on the same states,
  the diagonal star and Galerkin M2 give C = 7.964e-2/4.059e-2/2.060e-2 and
  7.917e-2/4.060e-2/2.063e-2 -- **identical to 3 digits**, order +0.98 both.
  A correct and a 20%-wrong Hodge are indistinguishable to it.  Root cause:
  the residual is a *curl*, so it annihilates curl-free error.  Rescaling
  `H_aux` by 2 everywhere changes C by **exactly 0.0%**; a smooth `1+1/r`
  (2x near the horizon) by 2.8%.
- **Cap-circulation is also blind.**  Summing the residual over a polar cap
  telescopes to `\oint H.dl` = enclosed current = 0 -- reference-free with
  exact ground truth, but the rim integrand `H_phi` is identically zero for a
  stationary axisymmetric state, so a multiplicative error times zero is
  still zero.  Measured: clean +1.00 order at r=3 and r=8, i.e. no signal.
- **Static background subtraction pins the endpoint.**  With
  `use_static_background`, vacuum Maxwell being linear makes the evolution
  exactly `rhs(dD, dB)`, so `delta = 0` is an exact fixed point REGARDLESS of
  how wrong the operator is.  The relaxation's final state is therefore
  pinned to the analytic background by construction, not by accuracy.

**Two projections that DO work, and why.**  Both have integrands that are
nonzero pointwise but integrate to a known value, so a systematic error
breaks the cancellation instead of being annihilated:
  1. `hodge1` -> Gauss flux through r=const (truth exactly 0).
  2. `hodge2` -> near-horizon magnetic energy (positive-definite quadratic
     form, so errors accumulate; truth from a 2-D quadrature).
Use them as a PAIR.  Either alone misleads: the residual says "no
difference", the energy says "20% wrong and not converging".

#### Corrections to commit f5f00a9f1

- The operator orders (Faraday 1.01, Ampere 0.98, bulk) measure the
  STENCIL's cancellation, not the Hodge.  They are unchanged by a correct
  Hodge (proven above) and are not evidence about the constitutive relation.
- The "+1.99 over L3-L6" promotion evidence is the convergence of the
  OBSERVABLE's discretization (analytic-on-mesh vs continuum, no evolution
  involved).  It is not evidence that the evolution operator is accurate.
- The residual `relaxed vs analytic-on-mesh` (+1.6e-4/+2.0e-4/+2.7e-4,
  settled but mildly growing) is the only part of that campaign that tested
  the operator, and its non-convergence now has a mechanism: B1.
- The vacuum promotion conclusion still stands -- Meissner expulsion is real
  and the solver is usable -- but it rests on a narrower base than the
  commit message implies.

#### Option (c) is CLOSED (was: a chart with g_rphi = 0)

Kerr admits no flat, nor even conformally flat, spatial slicing.  Doran
coordinates are unit-lapse and horizon-penetrating but their 3-metric is
NOT diagonal (Baines, Berry, Simpson & Visser, arXiv:2009.01397, eq. A.15:
"Note h_ij is not diagonal"); they carry the same `g_rphi`.  That paper also
proves a no-go: unit lapse is incompatible with diagonalizing the 3-metric
while keeping axial symmetry.  Independently, their eq. (3.10) requires
`g_rphi/g_phph` to be theta-independent for an azimuth-only diagonalization;
for Kerr-Schild it varies by 19.6% at r=1.11.  Boyer-Lindquist is the one
diagonal chart and its azimuthal shift `a/Delta` diverges at the horizon.
**No coordinate choice avoids this; the Hodge must carry the term.**

**Gate (revised): B1 must be resolved before P2.** P1 is independent of it
and can proceed in parallel.

**STATUS 2026-08-01: B1 is diagnosed, and the fix is validated in the lab
(B1b).  What remains is the C++ implementation** -- port the Whitney M1/M2
into `prismatic_mesh_metric`, replace the diagonal `hodge1_inv`/`hodge2` in
`dec_field_solver_gr_ks` only (the FLAT solver is unaffected: flat space has
no off-diagonal spatial metric, and the NS paper runs on it), add a
Jacobi-PCG for `M1^-1`, and re-run the vacuum campaign.  Scope the change to
the GR solver.

### P1 — Grid-free geodesic + GR push kernel (the bulk of the work)

Lift `geodesic_ks.hpp`'s RHS into a header with no `grid_ks.h` dependency
(pure `Metric_KS` formulas), then write the GR push against it. Follow
`coord_policy_gr_ks_sph::update_ptc` / `move_ptc` exactly.

**Gate:** test-particle orbits in the frozen analytic Wald field conserve
the Carter constant to the tolerance `tests/test_gr_ks_sph.cpp` uses. Use
the frozen-field trajectory harness from commit `24b701a39` as the pattern —
that test already exists for the flat dipole and is the right shape.

### P2 — Gather + deposit wiring

Fix G3 (edge_kind in the gather), apply the P0 normalization, add G6.

**Gate:** a static charge distribution reproduces Gauss's law on the metric
mesh; a single charge on a circular orbit deposits a current whose discrete
divergence matches −∂ρ/∂t to round-off.

### P3 — GR moments + injection

G5 and G7 together.

**Gate:** injected pairs at rest in the FIDO frame stay at rest (no spurious
drift); σ and D·B maps match the traditional-grid run on a comparable
configuration.

### P4 — Driver + first plasma run

G8. Config mirrors `main_wald.cpp` + the paper: a = 0.999, ε_{D·B} = 1e-3,
σ threshold, B0 such that the magnetization matches.

**Gate:** field lines threading the ergosphere are dragged in; an equatorial
current sheet forms inside the ergosphere.

### P5 — Validation against the traditional grid

**This is the real scientific gate, and it needs stating up front:** paper
§3.3 is a **2D axisymmetric** run (Nr × Nθ = 1024 × 1024, a = 0.999). Going
straight to 3D on the prismatic mesh means the current sheet's kink
instability is a *new result*, not a reproduction — so it cannot validate
the code. Keep an axisymmetric-equivalent comparison as the actual
validation target: match `gr_2d_kerr_schild` on a configuration where both
codes should agree (H_φ structure, σ map, current-sheet location, horizon
flux), and only then treat the 3D dynamics as physics output.

---

## Gotchas (from the vacuum work — these already bit us)

- **`H_aux` already carries `hodge2`.** It is a dual 1-cochain; the Ampère
  update is a bare `d1t_val * H_aux`. The flat solver is the opposite
  (`hodge2 * B`). Applying it twice in analysis fabricates a
  sub-first-order convergence order with a spurious O(1) floor.
- **Mask ghost layers at BOTH radial ends** in any residual/convergence
  diagnostic. Masking only the outer one understates the bulk order.
- **`use_flat_metric = true` is incoherent with the Wald IC** — the mesh
  goes flat but `set_initial_kerr_wald` still lowers with `Metric_KS`.
  `prismatic_sph_output` defaults to flat, so GR runs must set it false.
- **Registration order matters.** `lmesh` is built at registration time
  because the injector's `init()` runs before the updater's and needs
  `ptc_mesh()`. Building it in `init()` segfaults.
- **Stale binaries.** Problem binaries need explicit `--target` rebuilds
  after library edits, silently running old code otherwise.
- **Never launch long GPU runs as background Bash tasks** — they get reaped.
  Use `setsid nohup ... < /dev/null &`.

---

## Open questions

- Does the recovery B-gather's `(r_ref/r_k)²` weight rescaling remain valid
  on a metric mesh? (G3)
- Does the ~2e-4 vacuum accuracy floor (relaxed vs the mesh's own analytic
  state, settled but mildly growing with refinement) matter for plasma runs,
  or is it far below the PIC noise floor? Almost certainly the latter —
  shot noise is percent-level — but it should be stated once, measured.
- The vertical-edge Ampère spin effect (2nd order in Schwarzschild, 1st in
  Kerr — the rect-face shift cross-term) is a constant, not an order, for
  the bulk. Irrelevant for plasma runs unless it interacts with the current
  sheet; note and move on.
