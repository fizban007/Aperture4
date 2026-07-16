# Code-Paper TODO — figures & production runs

Deferred items to execute when drafting the methods paper.  The
infrastructure for all of these exists; each entry names its driver.
Working results already in hand are noted so the paper draft can start
from them.

## Particle-gather figures (A1 machinery, done at prototype scale)

- [ ] **Loss-rate / scattering comparison at L=4-5** with a
      gyro-radius/cell-size scan.  Prototype numbers (L=3, committed in
      ROADMAP A1 notes): primal ejects ~73% of gyrating particles in
      3000 steps in uniform B, 75% of trapped dipole particles in 6000
      steps; recovery loses none and matches exact-field pushing.
      Drive via the C++ updater (`use_recovery_gather` on/off) now that
      it is fast; the Python driver `python/recovery_study.py` defines
      the metrics (loss fraction, phase-averaged Δμ/μ, gc wander).
- [ ] **Gather convergence figure**: primal 1st vs recovery 2nd order
      (prototype data L=2-4 in `python/recovery_study.py` /
      `prismatic_recovery.py`; regenerate with more L values).
- [ ] **Continuity/jump histogram**: primal 10-26% |B| face jumps vs
      recovery round-off (V7 in the prototype).
- [ ] **Conditioning table**: div-free fit condition numbers vs vertex
      valence / radius — shows valence-5 vertices are not special.
      One-liner: `conditioning_table()` in `python/recovery_study.py`.

## Field-solver figures

- [ ] Cavity eigenmode convergence (exists: `cavity_convergence*.png`,
      `run_cavity_convergence.sh`) — regenerate final version, report
      both L2 and pointwise norms, error map vs distance to the 12
      valence-5 vertices.
- [ ] Vacuum dipole div-B / flux-error validation
      (`validate_vacuum_dipole.py`) — final version.
- [ ] A2 Deutsch benchmark figures (see roadmap A2): stationarity drift
      convergence, spin-down luminosity vs sin²α, wave-zone error maps.

## Method-argument figures

- [ ] **Two-tier convergence figure** (the paper's honest centerpiece):
      2nd-order wave dynamics (cavity + causally-isolated-annulus
      ratios 5.0/4.6) beside 1st-order quasi-statics (uniform
      spurious-curl ratios 2.0) with the circumcenter/midpoint-offset
      mechanism and the TRiSK/C-grid literature anchor (Peixoto 2016).
      Include the noise-dominance argument: measured static truncation
      ~4e-4 relative at L=7 vs percent-level PIC shot noise at
      realistic ppc.
- [ ] **Timestep advantage vs spherical Yee**: quantify polar-cell CFL
      penalty of an equivalent-resolution (θ,φ) grid vs the quasi-uniform
      icosahedral cells (analytic + measured dt).  Frame honestly:
      production Yee codes filter the poles instead of paying the dt —
      but filtered caps are effectively low-order anisotropic special
      regions; this mesh has uniform characterized error everywhere.
- [ ] Resolution-economics table: cells x steps per effective
      resolution at equal wall-clock, ico vs (theta,phi) grid.
- [ ] Optional: near-pole pitch-angle diffusion comparison vs the
      traditional-grid code on `develop` (novel figure; needs the Yee
      PIC run on Frontier or local).
- [ ] Multi-rank field-solver validation + weak/strong scaling (gated on
      Track B2).

## Prototype-scale results usable directly in the draft

- Recovery scheme definition + degeneracy analysis (div-free constraint
  cures the valence-5 null space) — ROADMAP A1 notes + commit messages
  `b3bc0e241`, `3fc6986c2`.
- Gnomonic locator fix (exact tiling; sliver pathology of perpendicular
  projection) — for the implementation section.
