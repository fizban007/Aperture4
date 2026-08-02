# Roadmap: NS Magnetosphere Prototype + Methods Paper

**Status as of 2026-07-16.** Supersedes the GR-oriented sequencing in
`PHASE_4_1B_PLAN.md` (the phase content survives, retargeted — see Track B).

## Strategic decisions

1. **GR work is shelved.** The near-term paper is a numerical-methods paper
   on the prismatic icosahedral mesh applied to NS (flat-space)
   magnetospheres. The traditional-grid 3D GR effort on `develop` covers the
   BH science. `dec_field_solver_gr_ks`, `prismatic_wald`, and the Wald/drift
   diagnostics are frozen — not deleted, not maintained. Do not resume them
   without revisiting this document.

   **AMENDED 2026-08-01 — the stated reason for shelving was wrong, and the
   principal cause is now identified and fixed.** The GR solver was shelved
   because vacuum Kerr-Wald at L5 relaxed to a state with no Meissner flux
   expulsion and a badly wrong toroidal field. That was attributed to the
   mesh/discretization. It was not. Two defects, both now repaired:

   - **Sign error in `apply_inner_boundary`** (`dec_field_solver_gr_ks_impl.hpp`,
     4 sites). The intended BC is zero radial gradient on the perturbation,
     `delta_0 = delta_1`, i.e. `X_0 = X_1 + bg_0 - bg_1`. The code had
     `+ bg_1 - bg_0`, which maps an exact background `bg_0 -> 2*bg_1 - bg_0`
     and so injected an O(h) error at shell 0 every step. Introduced in
     `84bdf628`, present for the entire life of the solver.
   - **No static background subtraction.** Expelled flux at a = 0.998 is
     ~0.2% of the far-field flux, while the Ampere residual near the horizon
     is percent-level — the truncation error is an order of magnitude larger
     than the observable, so the relaxed state is dominated by it and
     REFINING MAKES THE RELATIVE CORRUPTION WORSE (expulsion sharpens with
     resolution faster than the error shrinks). This is why L5 looked worse
     than lower resolutions.

   Measured effect on the actual test (a_field = 0 relaxing to a = 0.998,
   L3, damped, t = 60 M), reported as `Phi_cap(horizon)/Phi_cap(r=8)` with
   analytic target **0.00180**:

   | configuration | relaxed value | error |
   |---|---|---|
   | as shipped | 0.0168 | +833% (no expulsion) |
   | + BC fix + background subtraction | 0.0020 | +9% |

   With subtraction the exact on-shell state is now an **exact discrete fixed
   point** (flux ratio constant to 5 decimals over 10 M; it previously swung
   by ~800% of the expulsion signal). Toroidal/poloidal near the horizon
   improves 0.126 -> 0.259 against an exact value of 0.345.

   Note the O(1) objection does not apply: subtraction helps here not because
   `delta` is small (vacuum -> rotating Wald is an O(1) change) but because
   `delta = 0` becomes an exact fixed point, giving the relaxation the correct
   attractor. Kerr-Wald is an exact stationary vacuum Maxwell solution on
   Kerr, so holding it fixed deletes no physics — cleaner than the flat-space
   case where only the aligned dipole component can be subtracted.

   **Convergence, measured properly for the first time** (normalization-free
   cancellation measure; `analyze_drift.py`'s `dD/dt` over `|D|` misreads the
   order by one, since `D~` is a dual 2-cochain scaling as h^2):

   | operator | order |
   |---|---|
   | Faraday `d1.E_aux` | +1.3 |
   | Ampere `d1t.H_aux`, global | +0.65 |
   | Ampere, excluding the outermost shell | **+0.90 to +0.95** |

   The entire global deficit comes from ONE shell — the outermost, which is
   a ghost layer (`n_ghost_outer = 1`) that `analyze_drift.py` nonetheless
   includes, and whose one-sided shift stencil `834d15d2` flagged as needing
   "a deeper fix". Spin-independent (Schwarzschild and Kerr identical), so
   not frame-dragging. The bulk is a clean first order, matching the flat
   solver's known diagonal-Hodge tier — the mesh is not the problem.

   Also fixed/added while establishing the above:
   - `field_spin` config (default 0, preserving the shipped off-shell IC) —
     without it the on-shell residual could not be measured at all, so the
     "discrete-equilibrium residual" `analyze_drift.py` was built for had
     only ever been evaluated on data that is not an equilibrium.
   - `background_spin` config (default = `field_spin`) so the subtracted
     background can be on-shell while the IC is off-shell.
   - `use_static_background` for the GR solver, mirroring
     `dec_field_solver.h:202-209`.
   - Latent inconsistency, not fixed: `use_flat_metric = true` makes the mesh
     flat but `set_initial_kerr_wald` still lowers indices with `Metric_KS`
     at `bh_spin`, so that combination is incoherent. Harmless at the default
     (false); do not use it.

   **What this does NOT establish.** The remaining ~9% flux error and the
   ~25% toroidal error are consistent with the bulk first-order accuracy and
   have not been driven down. Recovery interpolation is NOT the lever for
   them: the Ampere residual is dominated 6:1 by the lapse term
   `alpha * hodge2 * B`, not the shift cross-term, and second-order Ampere is
   the reconstruction-Hodge route already measured unconditionally unstable
   (see the mitigation matrix below). Whether the GR effort should resume is
   still a scheduling question — but "the prismatic mesh cannot do GR" is no
   longer a supported reason.
2. **MPI field work continues to completion.** Phases 1–4.1a are done and
   tested; Phase 4.1b proceeds, but **retargeted at the flat
   `dec_field_solver`** instead of the GR solver. The flat solver has no
   shift cross-terms, so the halo-dependency structure is simpler than the
   4.1b plan assumed: exchange E (h+v edges) before Faraday, exchange B
   (tri+rect faces) before Ampère, refresh inside each semi-implicit Picard
   iteration. Everything else in `PHASE_4_1B_PLAN.md` (buffer splits, commit
   boundaries, layout/ptrs design, gotchas) carries over.
3. **The prototype magnetosphere runs single-GPU** (L=5–6, N_r per
   `python/prismatic_memory_budget.py`). Distributed PIC requires particle
   migration + additive J-reduction (Phase 6) — the current halo backends
   implement ghost-fill only. Phase 6 is post-prototype; a multi-GPU hero
   run is a stretch goal, not a paper dependency.
4. **Paper scorecard** (each item is also a bring-up milestone):
   - Field convergence: cavity eigenmodes (done), vacuum dipole (done).
     Report both L2 and pointwise norms; map error vs distance to the 12
     valence-5 vertices.
   - Particle-mesh: charge conservation (unit-tested), gyration / E×B drift
     convergence, plasma oscillation, primal-Whitney vs vertex-recovery
     interpolation scattering comparison.
   - Rotator: vacuum rotating dipole vs analytic Deutsch fields.
   - Flagship: aligned rotator with plasma — spin-down vs force-free
     μ²Ω⁴/c³, Y-point + equatorial current sheet, interior corotation.
   - Method selling point: quasi-uniform cells → no polar CFL penalty
     (quantify dt advantage vs an equivalent-resolution (θ,φ) Yee grid)
     and no polar filtering; multi-rank field-solver scaling from Track B.

## Phase 0 — Housekeeping (week 0)

- 0.1 Review + apply `stash@{0}` and commit. It contains definitions the
  committed tree needs even with GR shelved (`EdgeCochainKind` is used by
  `prismatic_sph_output`; `face_area`/`edge_length` in `prismatic_mesh_ptrs`).
  Review the ~70-line deletion in `dec_field_solver_impl.hpp` and the
  `wald_solution.hpp` changes separately; keep only what the flat path needs.
- 0.2 Shelve GR in the build: keep `dec_field_solver_gr_ks*` compiling if
  trivial after 0.1, otherwise gate it and `problems/prismatic_wald` behind
  a CMake option (default OFF). Add a status header to both files' docs.
- 0.3 Add a fast compile check (script or CI) covering the prismatic
  targets + tests, so the tree never regresses to non-building again.

## Track A — Physics (critical path for the paper)

### A1 — Second-order B-gather (vertex recovery) + PIC validation battery (weeks 1–3)

The single highest-risk/highest-value item.

**Decision (2026-07-16): implement vertex least-squares recovery for the
B-gather instead of dual Whitney interpolation.** Rationale: the dual
scheme has two unresolved design gaps (fan-decomposition spoke-edge
circulations; radial boundary half-shells) and tops out at first order
with partial continuity, while ZZ-style vertex recovery is fully C⁰,
second-order, needs no dual mesh or dual point location, and its linear
fit `B(x) = B₀ + G·x` yields the ∇B needed to later un-stub the GCA
curvature terms. Dual interpolation
(`docs/icosahedral_prismatic_pic/dual_interpolation_plan.md`) is kept as
a documented alternative / future work, not implemented.

Structure-preservation constraints (do not violate):
- Deposition stays primal Whitney (charge-conserving, unit-tested).
- E-gather stays primal Whitney 1-forms — it is the adjoint partner of
  the J-deposit; changing it risks secular numerical heating. Only the
  B-gather changes (the magnetic force is workless, so B-gather is
  structurally unconstrained).

Scheme: per-vertex linear-reproducing LSQ fit of B from the adjacent
face fluxes (~18 fluxes interior, ~15 at valence-5 vertices, ~12 one-
sided at radial boundary vertices — consider one extra layer of patch
depth there). Weights precomputed at mesh build; with log-spaced shells
the layers are self-similar, so store weights per sphere-vertex and
rescale analytically (fluxes ∝ r², lengths ∝ r). Per step: one
O(N_verts) kernel producing vertex B vectors; particles hat-interpolate
the components (existing Whitney 0-form machinery). Runtime switch keeps
the primal gather available as the paper's baseline.

**Prototype findings (2026-07-16, `python/prismatic_recovery.py`):**
1. The fit is rank-deficient unless **div B = 0 (trace-free G) is imposed**:
   the unconstrained trace mode is exactly null at the 12 valence-5
   vertices (cond ~1e9) and weak (σ≈0.01) at valence-6.  With the
   constraint, cond ≈ 16 everywhere including valence-5 and boundaries.
2. Patches need tri-face fans at shells k-1,k,k+1 (a single shell never
   samples dB_r/dr).
3. Measured on the dipole at L=2→4: primal converges at ratio ~2.0
   (1st order), recovery at ~4.7–6.3 (2nd order); at L=4 recovery is
   ~28× more accurate (rms).  Face-crossing jumps: primal 10–26% of |B|
   at L=3; recovery continuous to round-off.
4. **Production bug found**: the perpendicular-projection barycentric in
   `prismatic_mesh_ptrs.h compute_barycentric` does not tile the sphere —
   ~0.3% of positions fall in O(1e-4)-λ slivers claimed by no triangle;
   `find_triangle` then walks to its N_tri iteration cap and returns an
   arbitrary hint-dependent cell.  Fix in the CUDA port: central
   (gnomonic) barycentric (solve p̂ ∝ Σλᵢvᵢ, normalize by |Σλ_raw| — the
   abs guards against antipodal-triangle false positives), which tiles
   exactly.

**Particle-level results (2026-07-16, `python/recovery_study.py`, L=3,
Boris ensembles, gyro-radius ≈ 1/4 cell):**
- Uniform B, 3000 gyro-steps: primal gather ejected **73%** of purely
  gyrating particles from the shell (guiding centers random-walk ~0.45 r_*
  off the jumps) with rms Δμ/μ 4.3e-2; recovery matched the exact-field
  pusher to round-off (0% lost, Δμ/μ ~1e-11, wander identical to exact).
- Dipole drift orbits, 6000 steps: primal lost **75%** of trapped
  particles, survivors' phase-averaged Δμ/μ = 0.35; exact and recovery
  both lost 0% with Δμ/μ = 2.3e-2 and 4.3e-2.
- Conditioning table (div-free fit): median ~28, max ~38, statistically
  identical for valence-5 vs valence-6 and interior vs boundary.
- Caveat for the paper figure: L=3 is coarse; repeat at L=4-5 where the
  O(h) primal jumps are smaller, and scan gyro-radius/cell-size ratio.

Prototype verdict: recovery goes to production.

**CUDA port LANDED (2026-07-16, `prismatic_vertex_recovery.h/.cpp` +
`test_prismatic_recovery.cpp`):** div-free per-sphere-vertex weights with
(r_ref/r_k)² interior rescaling (O(N_vert_s) storage), per-step vertex
kernel + C0 hat gather through the exec-policy lambdas (same code host
and GPU), gnomonic compute_barycentric in production (fixes the sliver
bug), config `use_recovery_gather` (default true).  Full suite + GPU
smoke run pass.  Still open in A1:
- GCA path: gca_push re-interpolates B internally with the primal
  gather; thread Bv through when GCA becomes relevant (A3 uses Boris).
- Paper figures (loss-rate scan at L=4-5 etc.): deferred to drafting
  time — collected in PAPER_TODO.md.

Validation battery as tests + small drivers (prototype all of it first in
Python via the `prismatic_interp` pybind module before CUDA work):

- single-particle gyration: energy + gyroradius convergence vs dt and L;
- E×B and grad-B drift against analytic rates;
- plasma oscillation frequency (uses the deposit→J→Ampère loop end-to-end);
- primal-vs-recovery pitch-angle scattering / heating comparison (paper
  figure);
- recovery fit quality vs vertex valence and radius (valence-5 vertices
  and boundary shells are where a referee will poke).

### A2 — Vacuum rotating dipole vs Deutsch (weeks 3–4)

Fields only, single rank. Use the existing rotating-dipole inner BC +
Deutsch analytic IC in `dec_field_solver_impl.hpp`. Spin up, compare
against the analytic Deutsch solution in the wave zone, convergence in L.
Reuse the cavity analysis tooling for error maps.

**A2.0 + A2.1 results (2026-07-16).** Benchmark configs
`problems/prismatic_dipole/config_deutsch_L{4,5,6}.toml` (causally clean
protocol: r_max=45 > 9 + P, no damping, one period, recurrence error
||F(P)−F(0)|| in r<9 is pure solver error) + `config_deutsch_damped_L5`
(6-period absorber variant); analysis `deutsch_stationarity.py`,
`deutsch_luminosity.py`.  Note the code's "Deutsch" is the retarded
rotating point dipole (exact vacuum solution; the finite-star E
quadrupole is absent) — fine as a benchmark, worth a footnote in the
paper.  L_analytic = (8π/3)Bp²Ω⁴sin²α in code (rationalized) units.

Measured (Ω=0.2, α=60°): errB(r<9) = 4.17e-3 / 1.38e-3 / 6.85e-4 at
L=4/5/6 (ratios 3.0, 2.0); errE = 2.25e-2 / 1.17e-2 / 6.52e-3 (~1.9×).
The luminosity in the clean domain holds L/L_analytic = 0.85 steady
through the period at L=5.

**Open items found (do these next in A2):**
1. **Half-step staggering of IC + inner BC — fixed, with a corrected
   diagnosis.**  `apply_inner_bc(E, B, time_E, time_B)` now takes
   per-field times (explicit passes time_B = time − dt/2; the
   co-located semi-implicit passes equal times), and
   `set_initial_deutsch` initializes B at −dt/2 for the leapfrog, so
   every dump holds the stagger-consistent pair (B(t−dt/2), E(t)).
   The ladder rerun shows the recurrence numbers are UNCHANGED — as
   they must be in hindsight: a uniform dt/2 phase offset shifts the
   whole periodic solution and cancels exactly in F(t+P)−F(t).  The
   dt-halving signal originally read as O(dt) is equally consistent
   with ordinary O(dt²) leapfrog dispersion.  The remaining ~1.9×
   errE ratios are a genuine first-order SPATIAL component, most
   plausibly from driving the discrete interior with the analytic
   inner BC (O(h²) mismatch injected per step × O(1/h) steps per
   crossing → O(h)).  Quantifying/improving that (e.g. one-cell
   transition region, or comparing against a discrete reference
   solution instead) is a paper-polish item, not a blocker: absolute
   errors at L=6 are 7e-4 (B) / 7e-3 (E) per period.

   **Order-isolation study (2026-07-16, hdt_matrix.py + annulus test):**
   - (h,dt) matrix at fixed mesh: errB is ~pure spatial (dt-refinement
     changes it <5%); errE has a genuine ~O(dt^1) component that
     SURVIVES the half-step fix, plus a weakly-converging spatial floor
     (8.0e-3 / 6.7e-3 at L4/L5).
   - Causally isolated annulus (33<r<40, r_max=75, one period — free
     propagation, no boundary influence): L5→L6 ratios **5.0 (B) and
     4.6 (E)** — the BULK scheme is cleanly 2nd order for traveling
     waves, corroborating the cavity study.  (L4→L5 ratios 1.5/2.3 are
     pre-asymptotic: ~13 cells/wavelength at L4.)
   - Over-determination hypothesis REJECTED: driving tangential E only
     (new config `inner_bc_overwrite_b = false`, rotating-conductor
     style) reproduces the same errors and ratios.
   - **Mechanism NAILED (2026-07-16, spurious-curl test).**  The
     discrete curl h1inv·d1t·(h2·B) of the EXACT static dipole (a
     curl-free field; GPU IC dump + scipy matvec, seconds per level) is
     pure Hodge truncation.  Measured: relative spurious curl is
     UNIFORM across annuli (scale-invariant, 2.9e-3 at L4) and
     converges at exactly 2.0×/level — the diagonal circumcentric
     Hodge is intrinsically FIRST order for quasi-static fields on
     this mesh.  Wave dynamics enjoy supraconvergent cancellation
     (hence the 2nd-order annulus/cavity results); the quasi-static
     near-zone response inherits the full O(h).  Per-annulus SOLUTION
     error vs analytic confirms: [1,5) ratios ≈ 2.0 for both E and B;
     [5,9) (wave-dominated) ≈ 3-6.  This is the same phenomenon known
     from icosahedral C-grid dynamical cores (TRiSK 1st-order
     operators, Peixoto 2016).
   - Boundary-shell Hodge: the "×2 ghost" was replaced by the
     consistent truncated dual (hygiene); note the spurious-curl
     identity legitimately fails on the half-open boundary dual loops,
     and those rows are masked by BCs in every current use.
   - **SCVT (spherical Lloyd) mesh relaxation implemented**
     (`prismatic_mesh::sphere_optimize_iters`, config
     `mesh_optimize_iters`, default 0): improves the interior
     truncation constant ~1.5× but does NOT change the order —
     consistent with the dynamical-core literature.
   - **DECISION (2026-07-16): accept the two-tier order structure.**
     The scheme is a 2nd-order wave solver with 1st-order quasi-statics
     (both halves proven by cheap diagnostics; mechanism identified;
     TRiSK/C-grid literature anchor).  Galerkin and FV-reconstruction
     Hodge routes were explored in earlier sessions and hit their own
     first-order-limiting issues; not pursued.  Rationale: PIC shot
     noise (percent-level at realistic ppc) exceeds the measured
     quasi-static truncation (~4e-4 relative at L=7) by orders of
     magnitude, and the mesh's win is resolution economics (uniform
     CFL, no polar filtering).  Paper framing: report both tiers
     honestly with the mechanism; note spherical Yee's filtered polar
     caps are themselves effectively low-order special regions,
     whereas this mesh has uniform, characterized error everywhere.
     If the Hodge question is ever revisited, the spurious-curl probe
     (seconds, mesh-only) tests 2nd-order consistency of any candidate
     operator BEFORE solver integration.
   - Follow-ups adopted instead: (i) static-background subtraction in
     the flat solver — **DONE (2026-07-17)**: main-code E0/B0+delta
     pattern mirrored in dec_field_solver (config
     `use_static_background`, default off; B0 = aligned static dipole
     Bp·cos(obliquity), E0 ≡ 0; "E"/"B" are totals = background +
     delta, consumed by particles/sph/dumps; solver evolves
     "Edelta"/"Bdelta"; inner BC writes analytic−B0; damping acts on
     delta only).  Aligned static test: B drift over 2 periods drops
     4.6e-2 → 3.8e-8 (float round-off) — the quasi-static O(h) tier is
     eliminated for the background field.  Oblique damped L5 benchmark
     unchanged (L/L_dip 0.9567) with recurrence improved (errB
     1.15e-3→1.02e-3).  Time-dependent backgrounds deliberately
     unsupported (a rigidly rotating dipole is not a Maxwell solution;
     subtracting it would delete retardation physics) — for oblique
     runs only the aligned component is subtracted.  (ii) quantify the
     resolution-economics argument for the paper (PAPER_TODO).

   **REOPENED (2026-07-16, user request) — Hodge lab results
   (`python/hodge_lab.py`, offline probes, no solver changes):**
   - Probe 1 (B→H map, spurious curl of exact static dipole): the
     **reconstruction Hodge** — div-free linear LSQ fit from ~30
     nearby fluxes (the validated vertex-recovery machinery) +
     EXACT integration along the dual segment — is 2nd-order-plus
     consistent: 2.9e-3→5.1e-4 (L4), 1.5e-3→6.8e-5 (L5), order ratio
     **7.55** vs diagonal's 1.97.  Orientation fixed geometrically
     from stored face-vertex order (never from field values).
   - Probe 2 (isolated h1inv pairing, linear B with constant curl,
     analytically exact dual circulations): pairing error 6.9e-4 →
     2.4e-4, ratio **2.83** (~order 1.5) — after fix #1 this is the
     DOMINANT residual.  Fix = same pattern: reconstruct the vector
     field from dual-face fluxes (trapezoid/polygon dual geometry),
     integrate exactly along the primal edge.  Keeping the d1t
     loop-sum structure preserves boundary-of-boundary exactness, so
     Gauss-law/charge-conservation with deposited J stays exact.
   - **FULL-CHAIN CONSISTENCY ACHIEVED offline (2026-07-16,
     `python/hodge_lab_chain.py`):** the complete corrected Ampère
     operator W1·d1t·W2 on the exact static dipole:
     (diag,diag) 3.0e-3/1.5e-3 ratio 1.98; (W2,W1) 4.2e-4/6.5e-5
     ratio **6.51** — 2nd-order-plus, 23× more accurate at L5.  The
     pairing W1 (per-vertex div-free LSQ on dual-face fluxes — h-edge
     ruled patches, v-edge circumcenter polygons — + exact primal-edge
     integration) is exact on linear fields to 1.8e-13.
   - C++ integration LANDED (`prismatic_recon_hodge.*`, flag
     `use_reconstruction_hodge`, default OFF, kept as experimental
     reference).  **STABILITY: CONCLUSIVELY NEGATIVE (2026-07-16,
     `python/hodge_lab_spectral.py` + solver gauntlet).**  Complete
     mitigation matrix, all measured:
       * single-anchor: unstable, growth 15-20 c/r★ (solver, dt-indep)
       * anchor-averaged (incl. radial anchor pairs): 1.0/time (L2) →
         4.0/time (L3), growth ∝ 1/h
       * mixed chains (W2,diag) and (diag,W1): both unstable
       * D-weighted symmetrizations: worse (adjoint not consistent)
       * (curl-curl)² filtered leapfrog: cannot reach the unstable
         modes — they are MID-BAND (0.14-0.28 ω_max), and explicit
         filters have their own CFL at ν₄ω⁴=2
       * β-damped 4-iter Picard semi-implicit: monotonically WORSE in β
         (truncated Picard amplifies non-normal transients)
     Mechanism: the wide-stencil LSQ correction makes the operator
     non-normal with mid-band eigenvalues off the imaginary axis; W1
     additionally breaks the Gauss-law telescoping (D1⁻¹W1 ≠ I — my
     earlier "topologically exact" claim was WRONG for the pairing;
     it holds only for the diagonal E-update).  The only
     theoretically-sound remaining route is the full SPD Galerkin mass
     matrix with real solves (explored in earlier sessions, known
     walls, 5-15× cost) — nothing new learned that changes its
     assessment.
   - **CONCLUSION: the accepted two-tier framing STANDS**, now with a
     rigorous justification: explicit local reconstruction corrections
     are 2nd-order consistent but unconditionally unstable (measured
     spectra, identified mechanism).  This negative result + the probe
     methodology is paper material (see PAPER_TODO).
2. **Absorbing-layer artifact — RESOLVED (2026-07-16 absorber study).**
   The old configuration (r_max=20, linear σ ramp, coef 0.1) settled at
   L ≈ 0.59 L_analytic post-diagnostic-fix (the historical 0.52 included
   the √γ bug).  Mechanism NAILED, three stacked effects, all measured
   at L=4 (entrance-kr scan + standing-wave fits from sph dumps):
   (a) coef 0.1 is under-damped — round-trip optical depth ~1, so the
   outer wall reflects ~35% amplitude (coef scan: 0.60→0.88 recovering
   monotonically to coef 3);
   (b) the layer-entrance reflection interferes with the HARD analytic
   inner BC at FIRST order in the reflected amplitude ρ:
   δL ≈ 2ρ·cos(2k·d_in), resonantly pumped at 2k·d_in = 2πn (entrance
   at 15.4 → +12%, at 30.7 → +8.6%; standing-wave ratio 4.5% on
   resonance vs 1.0% off) and zeroed at quarter-points
   d_in = (2n+1)λ/8;
   (c) ρ itself is the near-field impedance mismatch, ρ ≈ 1.7/(k r_in)²
   — resolution-INDEPENDENT (L4→L5 changes it ~20%), only weakly
   affected by taper smoothing, so no sponge beats ~1-2% with a
   reflective inner BC.  An entrance inside the induction zone
   (k r_in ≲ 3) additionally corrupts the wave-zone L(r) flatness.
   Production absorber (config_deutsch_damped_L5): r_max=45, entrance
   19.8 = 5λ/8 (quarter-point, k r_in = 4), cubic taper
   (damping_exponent = 3, new config knob), coef 1.0.  Measured at L=5:
   L/L_analytic = 0.957 vs clean 0.981 (systematic −2.4%, was −40%),
   settles by period 3, period-1 mean 0.982 reproduces the clean
   benchmark, recurrence err(r<9) 1.15e-3 (B) — BELOW the old ~1.5e-3
   floor and the clean-run 1.37e-3.  Premium option: entrance 51 =
   13λ/8 (k r_in = 10, needs r_max ≈ 110, +10 shells under log
   spacing) → −1.2%.  Sub-1% would need a genuinely reflectionless
   outer treatment (Silver-Müller BC or PML) — future work, noted in
   PAPER_TODO.  Bug fixed en route: update_semi_implicit damped the
   base state once per Picard iteration (and double-damped the final
   state) instead of damping the iterate.
3. **Diagnostic accuracy — RESOLVED (2026-07-16), two parts.**
   (a) The dominant deficit was a bug: prismatic_sph_output defaulted
   `use_flat_metric = false`, silently applying the a=0 Kerr-Schild
   √γ = sinθ√(Σ(Σ+2r)) — Schwarzschild with M=1 — to flat-space runs,
   scaling B^i by 1/√(1+2/r) (≈ 0.881 at the r=7 probe, matching the
   measured continuum limit exactly).  Default is now FLAT; GR revival
   must set use_flat_metric = false (noted in prismatic_wald/README).
   (b) sph_output's B gather now uses the C0 second-order vertex
   recovery (config `sph_use_recovery`, default true; auto-disabled on
   KS backgrounds).  Post-fix: L/L_analytic at the IC = 0.962 / 0.981 /
   0.990 at L=4/5/6, converging to 1 at first order (the E gather is
   still primal Whitney — an E-circulation recovery is a possible
   future upgrade but needs its own edge-moment fit).
4. Transient settling: with an absorber, the recurrence metric needs
   ~4 periods of settling (measure the last period pair).

### A3 — Aligned rotator prototype (weeks 5–8)

- Injection: start from the existing `fill_volume` machinery; add a simple
  surface/ubiquitous injection scheme (small addition to
  `prismatic_ptc_updater`). Scaled-down B (standard practice), plain Boris —
  the GCA curvature no-op stays a no-op for now.
- Deduplicate the injection code currently copy-pasted between
  `problems/prismatic_dipole/src/main.cpp` and `streaming_test.cpp` into a
  shared helper while touching it.
- Runs at L=5 (bring-up) → L=6 (production single-GPU). Scorecard: spin-down
  luminosity, current sheet / Y-point morphology, corotation.
- Stretch: Michel split-monopole benchmark (cheap, analytic).

## Track B — MPI (finish what's built; parallel to Track A)

### B0 — De-risk before solver conversion (weeks 1–2)

- Convert `prismatic_mesh_local` / `prismatic_mesh_metric_local` /
  `prismatic_d1_local` storage from `std::vector` to `buffer<Scalar>`/
  `buffer<int>` with `copy_to_device()`, and add the
  `prismatic_mesh_local_ptrs` struct (4.1b.0) able to hand out host or
  device pointers. Mechanical now, painful after kernels convert.
- Add the missing operator-level test: local d1 blocks × haloed field ==
  global d1 × field, on simulated 20-rank angular, K-slab radial, and
  combined partitions (in-process backend). This catches the bug class
  4.1b will produce.
- Extend `test_prismatic_mpi_backend_multirank` to radial (K ranks) and
  combined (20·K) configurations; exercise `comm_radial` over real MPI.

### B1 — Phase 4.1b retargeted: convert `dec_field_solver` (weeks 2–5)

Follow `PHASE_4_1B_PLAN.md`'s commit sequence, applied to the flat solver:

1. scratch/field buffer split (h_edge/v_edge, tri/rect) sized to layouts;
2. Faraday side (`B -= dt·d1·E`) on local buffers, E-halo before;
3. Ampère side (`E += dt·h1inv·(d1t·h2·B − J)`), B-halo before;
   semi-implicit: halo refresh inside every Picard iteration;
4. ICs (dipole / Deutsch / cavity) iterate owned elements, one exchange
   after; BC/damping routines act only on ranks owning the boundary shells;
5. `prismatic_edge_field`/`prismatic_face_field` split (plan option (a):
   two internal buffers, `.h()`/`.v()` accessors). **Coordinate with the
   particle updater**, which registers the same E/B/J slots and indexes
   them through global ptrs — for now the particle path is only supported
   on a single-rank partition; assert that explicitly rather than breaking
   silently.
6. Per-rank `dump_aux_fields`; parallel HDF5 deferred (Phase 5).

Note: `mpi_halo_backend` stays host-staged through B1 (correctness first).
Host↔device copies at exchange points are acceptable at validation scale.

### B2 — Multirank validation + scaling figure (weeks 5–6)

- Cavity-resonator eigenmode on 1 vs 20 (and 20×K if feasible) ranks:
  max relative difference < 1e-4 (float, per the plan's acceptance
  criterion), and identical convergence slope. This doubles as the analytic
  reference the GR plan lacked.
- Weak/strong scaling numbers for the field solver → paper figure.
- Device-direct (CUDA-aware) exchange in `mpi_halo_backend` if the
  host-staging shows up in the scaling data; it is one contained class.

### B3 — Status update (2026-07-18): Phases 5–6 LANDED

- **Halo exchange is packed device-direct** (config `halo_device_direct`,
  default on; `false` falls back to full-buffer host staging for
  debugging).  GPU-aware MPI detected at runtime; wire format identical
  in both modes.  Bit-exact at 20/40 ranks in both modes.
- **Phase 5 done**: exporter writes single global snapshot files via
  collective parallel-HDF5 owned-run hyperslabs
  (`H5File::write_parallel_runs`); sph output gathers owned cochains to
  world rank 0 and runs the serial path there.  All outputs bit-identical
  to 1-rank at 20/40 ranks.  Downsampling strides remain single-rank.
- **Phase 6 done, ARCHITECTURE CHANGED from the original plan**: instead
  of local-indexed particle kernels + halo reduce() (the recovery
  B-gather stencils need far deeper halos than the solver plans), the
  particle path runs on GLOBAL field replicas ("E_ptc"/"B_ptc"
  refreshed each step by `prismatic_field_replicator`, registered
  first), particles sharded by cell ownership, deposits allreduced
  (`prismatic_field_sync`) into the local "J"/"rho"/... the solver and
  sph consume, and leavers migrated by device-packed MPI_Alltoallv in
  `prismatic_ptc_updater::migrate()`.  Rationale: every rank holds the
  full global mesh anyway (until 4.1a.4), and fields are small next to
  particle buffers, which are what sharding must scale.  Acceptance:
  `test_prismatic_pic_multirank` — counts identical, zero misowned,
  outputs within 3.6e-5 of serial at 20/40 ranks; ns_rotator runs the
  full distributed stack.
- Optimization backlog (do when profiles demand): device-side
  scatter/pull in `prismatic_field_sync` (host loops today), owner-sum
  neighbor reduction replacing the J/rho allreduce, allgather
  replication replaced by halo-depth-extended local fields.

### Still deferred

- Partition-aware `prismatic_mesh::build()` + `release_global_buffers()`
  (4.1a.4): only needed beyond L≈7 where the global-build footprint
  bites.  Note Phase 6's replicated-field architecture also assumes the
  global mesh per rank; 4.1a.4 requires revisiting it together.
- Weak/strong scaling measurements + paper figure (deliberately pushed
  until the full PIC stack existed — 2026-07-18 decision).

## Merge point (weeks 7–9)

If Track B lands B2 early and Phase 6 looks tractable, attempt a
multi-GPU rotator. Otherwise the paper ships with: single-GPU PIC
magnetosphere (A3) + multi-rank field-solver validation/scaling (B2) +
distributed PIC as future work. **Do not let the paper wait on Phase 6.**

## Risk register

- A1 vertex-recovery gather is the only genuinely new numerics; if it
  slips, everything in A2/A3 still proceeds with primal interpolation and
  the paper's particle claims weaken. Start it first (Python prototype via
  `prismatic_interp` before CUDA); timebox to 3 weeks before deciding
  whether the paper leads with fields + charge conservation.
- Recovery-specific risks: patch conditioning at the 12 valence-5
  vertices and one-sided boundary patches (~12 fluxes for 12 unknowns —
  may need deeper patches), and weight storage at high L (mitigated by
  per-sphere-vertex weights + analytic radial rescaling under log-spaced
  shells; verify the shells are actually log-spaced in the configs used).
- B1 field-container split (step 5) is the one place Tracks A and B touch
  the same code (`prismatic_field_data.h`, updater registration). Land it
  as a single coordinated commit; run the full prismatic test suite + one
  A-track driver before and after.
- Semi-implicit Picard halo refresh: silent multi-rank-only drift if
  missed (plan gotcha). The B2 bit-close test is the guard — run it with
  the semi-implicit path, not just explicit.
