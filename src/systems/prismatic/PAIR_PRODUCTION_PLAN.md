# Plan: pair production for the prismatic magnetosphere (staged)

Written 2026-07-31.  Foundation session: Stage 1 implemented; Stages 2-3
designed here so later sessions inherit the decisions.

## 1. Why

The volumetric E·B-triggered injection is a FIELD-conditioned source
with a cadence (`inj_interval`), so it beats against the very structures
that regulate E·B — observed as strong intermittency at the separatrix
base and at the Y-point.  Real pair production is conditioned on the
PARTICLES (their Lorentz factors and radiated photons), closing the loop
locally: acceleration → gamma growth → pairs → screening.  The staged
plan replaces the injector as the interior source while keeping it
available for seeding/floor duty.

## 2. The three stages

### Stage 1 — gamma-threshold instant pairs (IMPLEMENTED)

Any electron/positron reaching `pair_gamma_thr` immediately spawns an
e+/e- pair at its own location:

- children each carry `pair_gamma_secondary` (= gamma_s), momentum along
  the parent's direction (relativistic beaming), parent weight;
- the parent loses 2*gamma_s (momentum rescaled, direction kept) —
  energy is exactly conserved (no radiation loss in this stage: the
  intermediate photon is instantaneous and virtual);
- purely LOCAL: children are born in the parent's cell on the parent's
  rank — no migration, no comm, no new checkpoint state;
- GCA parents (flag gca_state): the deduction treats Gamma ≈
  sqrt(1 + u_par²) (locked limit, mu ≈ 0, drift factor kappa ≈ 1 —
  documented approximation; exact in the region that matters).  Children
  of GCA parents are born in the GCA representation (u_par =
  sign(parent) * sqrt(gamma_s² - 1), mu = 0), mirroring inj_gca.
- children are flagged PtcFlag::secondary (diagnostics can separate
  cascade generations from injected plasma).

System `prismatic_pair_producer` (header
`prismatic_pair_producer.hpp`), registered BETWEEN the injector and the
updater — newborns are pushed and deposit in the same step, matching
injector semantics.  Kernel is a single pass over particles with an
atomic slot cursor; buffer-overflow-safe (threads whose reserved slots
exceed capacity skip production entirely and are counted; the parent is
left untouched so production retries next step).

Config:
```
use_pair_production   = true
pair_gamma_thr        = <trigger Lorentz factor>
pair_gamma_secondary  = <child Lorentz factor, default 5>
pair_prod_r_max       = <radial gate; <= 0 (default) = everywhere>
```
Init aborts unless pair_gamma_thr >= 2*pair_gamma_secondary + 2 (parent
must stay super-luminal-free, i.e. gamma_new >= ~2).

Deliberately NOT in stage 1 (documented):
- per-cell multiplicity cap (PS18-style, <= 10 within r < 2R*): the
  rho_abs deposit is available for it; add when a run shows runaway.
- rate limiting / probabilistic throttle: threshold + energy loss is
  self-limiting to one production per ~step per particle.
- radiated energy: none is lost (see above).  Stage 2 changes this.

### Stage 2 — photon tracing with finite free path (CB14 scheme)

Chen & Beloborodov 2014: the particle at threshold emits a PHOTON
(energy 2*gamma_s carried off, parent loses it), the photon propagates
BALLISTICALLY, and converts to the e+/e- pair after a randomized free
path ell * (0.9 + 0.2u).  The main codebase has exactly this scheme as
grid-bound policy classes — `threshold_emission` (trigger) +
`fixed_photon_path` (emit/convert) under `radiative_transfer` — reuse
the LOGIC, not the code (it is welded to the Cartesian ptc_ptrs/ph_ptrs
and Config<Dim> grid).

Prismatic design decisions (made now, implement later):
- Photon storage: own `prism_photon` DEF_PARTICLE_STRUCT (x1,x2,x3
  barycentric+zeta, p1,p2,p3 Cartesian photon momentum, E, weight,
  path_left, cell, id, flag) — 9 Scalar components; do NOT overload the
  particle buffer.
- Push: straight lines in Cartesian (flat space), reusing
  local_to_cartesian_impl / cartesian_to_local_impl for the cell walk —
  the Boris branch's position update minus the momentum update.
  path_left -= c*dt each step; on path_left < 0 convert (deposit an
  e+/e- pair exactly as stage 1 does, energy E_ph/2 each), guarded by
  the same r-gate; absorb at r boundaries like particles.
- Photons deposit NOTHING (no current, no rho) — cheapest species.
- Migration: photons cross ranks; extend the migration machinery with a
  photon wire (the particle wire carries 8 scalars; photons need 9 —
  a separate Alltoallv with its own component count, reusing
  exchange_wire's shape).  This and the CHECKPOINT SCHEMA (photon
  datasets in the snapshot) are the two real costs of stage 2.
- The stage-1 instant-pair path stays as the `photon_path -> 0` limit
  and as the cheap fallback (config-switched).

### Stage 3 — physical conversion (LATER, exploratory)

Replace the fixed free path with field-dependent conversion (one-photon
magnetic conversion / gamma-B, possibly gamma-gamma on a thermal
background), and threshold emission with curvature-spectrum sampling
(`sync_curv_emission` machinery).  Tricky (resonance physics, spectrum
sampling cost); explicitly deferred until stages 1-2 have
production-validated the plumbing.  Design constraint to preserve: the
emit/convert decision points in the stage-2 code must stay two small
policy hooks so stage 3 swaps them without touching the transport.

## 3. Interplay with the existing injector

Stage 1 runs IN ADDITION to the volumetric injector.  The experiment
that motivated this plan is: reduce the injector to a seed (raise
inj_eb_threshold / lower inj_weight, or restrict to first shells) and
let threshold pairs carry the interior supply — then compare
intermittency at the separatrix base and Y-point against the pure
injector baseline.  Do this as an A/B at L5-L6 before any L7 spend.

Diagnostic to add when analyzing: secondary fraction map (rho_abs of
PtcFlag::secondary vs total) — shows where the cascade actually runs.

## 4. Validation

Stage 1 (tests/test_prismatic_pair_prod.cpp, "[pairprod]"):
- threshold partition: below-threshold particles untouched bitwise;
- energy conservation: sum(gamma*w) identical before/after production;
- child properties: direction collinear with parent, weight/cell/coords
  inherited, secondary flag set, GCA children in GCA representation;
- buffer overflow: capacity-clamped, no partial pairs, parents
  untouched when skipped, live count consistent;
- ion immunity (ions never produce).

Stage 2 adds: photon ballistic propagation across cells/ranks (walk
correctness vs analytic straight line), conversion bookkeeping, photon
census in checkpoints, migration wire round-trip.
