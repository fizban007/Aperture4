# prismatic_wald — Kerr-Schild Wald on the prismatic icosahedral mesh

Kerr-Schild Wald problem on the prismatic icosahedral mesh
(`dec_field_solver_gr_ks`). The prismatic effort's near-term target is
flat-space NS magnetospheres
(`src/systems/prismatic/ROADMAP_NS_MAGNETOSPHERE.md`); 3D GR science
currently runs on the traditional-grid solver on the `develop` branch,
well-tested on Frontier.

**Status (2026-08-01). ACTIVE — un-shelved.** Vacuum Kerr-Wald relaxes to
the rotating Wald solution with Meissner flux expulsion, and the cap-flux
observable converges at **second order over L3–L6**. The earlier "no
expulsion at L5" result came from two fixable defects (inner-BC sign,
missing background subtraction) plus a diagnostic protocol that could not
have measured the effect anyway — see "Convergence results" below and
`ROADMAP_NS_MAGNETOSPHERE.md` "Strategic decisions" §1.

## The relaxation test

Start from the Schwarzschild (`a_field = 0`) Wald field on a spinning Kerr
background and evolve; the system should radiate the mismatch away and
relax to the rotating Wald solution, which for near-extremal spin expels
most magnetic flux from the horizon.

Run it as:

```toml
bh_spin           = 0.998
field_spin        = 0.0      # off-shell IC — the thing being relaxed
background_spin   = 0.998    # on-shell background held fixed
use_static_background = true
damping_length    = 24       # outer absorber; sponge over r >= 9.33 at L5
damping_coef      = 0.2      # NOT 0.08 — see "Protocol" below
use_flat_metric   = false    # REQUIRED: prismatic_sph_output defaults to flat
```

Both `use_static_background` and the outer absorber matter. Without
subtraction the relaxed state has ~9x too much flux through the horizon
(no expulsion at all); without damping, reflections off `r_max` dominate
and nothing settles.

`background_spin` must be **on-shell** (equal to `bh_spin`). Subtracting an
off-shell background is meaningless: its continuum time derivative is not
zero, so its discrete RHS is not pure truncation residual.

### Protocol — the three things that made the measurement possible

Each of these was wrong in the original attempt, and each on its own is
enough to hide the effect entirely.

1. **Run to t = 220 M, score the mean over 150–220 M.** The relaxation
   rings with an ~18.5 M period (a horizon ↔ sponge cavity mode) decaying
   on ~33 M. At t = 60 M — the original duration — the ring is still 5–8%
   of the signal, two orders of magnitude above the quantity being
   measured, so a single-time score at 60 M measures the phase of a
   transient, not a discretization error.
2. **`damping_coef = 0.2`, not 0.08.** Scanned at L3: 0.08 leaves a 4.8%
   ring still going at t = 110 M; 0.2 kills it to 0.09% by 150 M; above
   ~0.5 the sponge ramp itself starts reflecting and the ring grows again.
   Put the sponge *outside* the reference shell (r >= 9.33 for an r = 8
   normalization) so the denominator is not pinned to the background.
3. **Use a nested radial family.** Fix `r_min`/`r_max` and double `N_r`
   per level, so the log-spaced shell radii are nested and every level has
   a shell at exactly the same radius. Otherwise the "horizon shell" slides
   with L and the measured ratio changes for purely geometric reasons.
   `make_conv_configs.py` emits the whole family.

## Convergence results (2026-08-01)

Cap-flux ratio `Phi_cap(r=1.10569)/Phi_cap(r=8.016)`, continuum Wald value
6.414483e-3. The no-expulsion (uniform-field) value at these radii is
1.9026e-2, so the relaxed state expels ~66% of the flux a non-rotating hole
would thread.

| L | analytic on mesh | vs continuum | relaxed (150–220 M) | vs continuum |
|---|---|---|---|---|
| 3 | 6.407144e-3 | −0.1144% | 6.408183e-3 | −0.098% |
| 4 | 6.412630e-3 | −0.0289% | 6.413868e-3 | −0.010% |
| 5 | 6.414018e-3 | −0.0072% | 6.415703e-3 | +0.019% |
| 6 | 6.414366e-3 | −0.0018% | — | — |

The observable converges at **+1.99, +1.99, +1.99** across four levels —
the direct reversal of the old "refining makes it worse" symptom.

**Accuracy floor.** Relaxed vs the mesh's own analytic state is +1.6e-4,
+2.0e-4, +2.7e-4 at L3/L4/L5 — settled (ring < 2e-5 by 180 M) and mildly
*growing*, so it does not converge. It falls only weakly with domain size
(+1.62e-4 → +1.04e-4 for a 4.6x larger domain). End-to-end accuracy
therefore stops improving around 2e-4, which is why an L6 relaxation is not
worth its ~3 h: its discretization error (~1.8e-5) is well under the floor.

**Operator orders.** Measured with the normalization-free cancellation
ratio `||A x|| / || |A| |x| ||`:

| rows | Faraday `d1.E_aux` | Ampere `d1t.H_aux` |
|---|---|---|
| all | 1.27, 1.15, 1.08 | 0.66, 0.60, 0.56 |
| **bulk** (ghost shell masked at **both** radial ends) | **1.03, 1.01, 1.01** | **0.97, 0.98, 0.98** |

Both operators are clean, stable first order in the bulk; Ampere fits
`C = 0.0805 h` with 2.1% residual and no meaningful floor. Everything that
looked sub-first-order and degrading was boundary rows: each radial end
carries a ghost layer whose half-open dual loops make the residual identity
fail by O(1), and at a ~1/N_r fraction they drag the global figure down and
make it look like it degrades with refinement. **Mask both ends** — masking
only the outer one (which is what produced the earlier "+0.90 to +0.95"
figure) is not enough.

One genuine spin effect survives: Ampere on **vertical** edges (stencil =
rect faces only) is second order in Schwarzschild (1.99, 2.00, 2.00) but
first order in Kerr (1.00, 0.95, 0.90) — the rect-face shift cross-term
degrading an otherwise superconvergent stencil. It is worth ~25x on those
rows but they are a minority of the bulk, so it moves the constant, not the
order. SCVT mesh relaxation (`mesh_optimize_iters`) does **not** help:
bulk Ampere 0.0742 h vs 0.0805 h, ~8%, matching the flat-solver result.

## Configs and tools

Convergence family (generated — edit `make_conv_configs.py`, not these):

- `config_conv_L{3..6}_ana.toml` — on-shell IC (`field_spin = bh_spin`),
  1 step. The analytic state sampled on that level's mesh: both the scoring
  target and the discrete-equilibrium residual in `ic_aux.h5`.
- `config_conv_L{3..6}_relax.toml` — the relaxation test, t = 220 M.
- `config_movie_L5.toml` — as L5_relax but with meridional-resolution
  spherical output (`sph_N_theta = 181`, `sph_N_phi = 4`) for the movie.

Tools:

- `flux_cap.py` — cap-flux ratio from the tri-face flux cochain;
  `--series` for the time series. Evaluates at a **fixed** radius shared by
  every level (see `R_HORIZON_DEFAULT`).
- `score_convergence.py` — assembles the table above from the `_ana` /
  `_relax` families.
- `residual_order.py` — the operator orders, with both-ends masking.
- `make_wald_movie.py` — meridional movie: colour `H_phi`, contours
  poloidal field lines. `--verify` checks the `H_phi` reconstruction
  against the on-shell state, where it must vanish.

Legacy (pre-2026-08, kept): `config_L4_ic.toml`, `config_L5_ic.toml`,
`config_L5_check.toml`, `config_L5_a0.toml`, `config_L5.toml`.

## Diagnostics — read this before trusting a number

- `analyze_drift.py` loops over **all** shells `0..N_r`, including both
  ghost layers, which drags the measured Ampere order from 0.98 to 0.56.
  Exclude the ghost layer at **both** radial ends — excluding only the
  outer one gets you to ~0.90-0.95 and still understates the bulk. Vertical
  edges and rect faces at layer `k` span `k..k+1`, so they need
  `k+1 <= N_r-1` too. `residual_order.py` does this correctly.
- `analyze_drift.py` normalizes `dD/dt` by `|D|`. `D~` is a dual 2-cochain
  (~h^2) while `dD` is a signed sum of ~h^1 terms, so `dD/D ~ h^(p-1)`: a
  *constant* ratio already means first order, not zero order. For an
  order measurement prefer a normalization-free form, e.g.
  `||d1t H_aux|| / || |d1t| |H_aux| ||`.
- **`H_aux` already carries `hodge2`.** The GR solver builds it as a dual
  1-cochain, `H_aux[f] = face_alpha[f]*hodge2[f]*B[f] + shift`
  (`dec_field_solver_gr_ks_impl.hpp:246`), and the Ampere update is a bare
  `d1t_val * H_aux`. Post-processing must **not** apply `hodge2` a second
  time; doing so leaves the boundary rows O(1) and fakes a sub-first-order
  order with a spurious floor. The FLAT solver is the other way round —
  `dec_solver_dist.h:489` does `d1t_val * hodge2[f] * B_f[f]` — so there
  the analogue of `x` really is `hodge2 * B`.
- `check_ic.py` compares `E_aux` per edge against `A_0(v0) - A_0(v1)`. That
  reference is a pure discrete gradient and `d1 d0 = 0`, so the comparison
  is **gauge-dependent**: a large per-edge discrepancy on horizontal edges
  can be entirely curl-free and physically invisible. Judge Faraday by
  `d1 E_aux`, not by `E_aux` itself.
- Physical acceptance test: `Phi_cap(horizon)/Phi_cap(r=8)` from the
  tri-face flux cochain, via `flux_cap.py`. **Evaluate it strictly outside
  the horizon, at a radius that exists at every level.** The default is
  r = 1.10569, the coarsest common shell outside `r_+ = 1.06321`.
  - The retired "analytic target 0.00180, fixes give 0.0020" figure was
    measured on a shell at r = 1.0285 — *inside* the horizon. Two checks
    agree: the as-shipped 0.0168 is exactly the unexpelled uniform-field
    value at r = 1.037, and no grid with a shell outside `r_+` yields
    0.00180. That radius sits where `Phi_cap` crosses zero
    (`dlnPhi/dlnr = 34.6`), so +-1% in shell placement is +-35% in the
    score, and it lies in the causally disconnected interior where the
    inner BC and damping act. Do not use it as an anchor.
  - "First shell outside `r_+`" is also wrong as a rule: on a nested family
    each refinement adds shells, so it slides inward with L (1.1057 at
    L3/L4 but 1.0644 at L5) and the ratio then moves for geometric reasons.
- `H_phi -> 0` is the sharpest stationarity check, and it is what the movie
  shows. Stationary vacuum gives `curl H = 0`, so `H = grad(chi)`, whose
  phi-component vanishes under axisymmetry — while `B_phi` stays finite
  (toroidal/poloidal ~ 0.35 near the horizon). At L5 it falls 23x, from
  8.67e-2 to 3.75e-3, onto an analytic floor of 2.94e-3.

Note `analyze_drift.py` hardcodes `a = 0.998` for its `r_+` annotation.
