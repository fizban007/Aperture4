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
1. **O(dt) staggering error in IC + inner BC** — dt-halving at fixed
   mesh leaves errB unchanged but drops errE from 2.25e-2 to 1.39e-2
   (an O(dt) part ≈1.7e-2 + spatial floor 5.3e-3 at L4): the analytic
   IC and the per-step BC overwrite evaluate B at integer times while
   leapfrog B lives at t+dt/2.  Fix: evaluate the B overwrite at
   time+dt/2 (and stagger the B IC by dt/2); then rerun the ladder —
   errE should become 2nd order and errB's L5→L6 ratio should recover
   toward 4.  (The cavity analysis already applies this half-step shift
   *in post-processing*; the BC needs it *in the solver*.)
2. **Absorbing-layer artifact**: the damped steady state settles at
   L ≈ 0.52 L_analytic at BOTH L=4 and L=5 (resolution-independent),
   vs 0.85 in the clean domain — absorber reflection/interference, not
   solver decay.  It also floors the damped recurrence error at ~1.5e-3.
   Needs an absorber study (taper profile, length, r_max) before A3,
   which will run with damping.
3. **Diagnostic accuracy**: L(IC, exact fields) measures 0.82/0.85 of
   analytic at L=4/5 because prismatic_sph_output interpolates with
   first-order primal Whitney forms.  Switching sph_output's B (and E?)
   interpolation to the recovery gather would make all grid diagnostics
   second order.  Cheap and high-value.
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

### B3 — Deferred (post-paper unless the paper needs a hero run)

- Partition-aware `prismatic_mesh::build()` + `release_global_buffers()`
  (4.1a.4): only needed beyond L≈7 where the global-build footprint bites.
- Phase 5 parallel HDF5.
- Phase 6 distributed PIC: particle migration between ranks + additive
  J/rho halo **reduction** (owner sums ghost contributions — the reverse
  of the existing ghost-fill exchange; needs a new `reduce()` path in the
  halo backends alongside `exchange()`).

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
