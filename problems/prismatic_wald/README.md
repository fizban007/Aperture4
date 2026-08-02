# prismatic_wald — Kerr-Schild Wald on the prismatic icosahedral mesh

Kerr-Schild Wald problem on the prismatic icosahedral mesh
(`dec_field_solver_gr_ks`). The prismatic effort's near-term target is
flat-space NS magnetospheres
(`src/systems/prismatic/ROADMAP_NS_MAGNETOSPHERE.md`); 3D GR science
currently runs on the traditional-grid solver on the `develop` branch,
well-tested on Frontier.

**Status (2026-08-01).** Previously marked SHELVED because vacuum Wald at
L5 relaxed to a state with no Meissner flux expulsion. That was traced to
two fixable defects, not to the mesh — see the amendment in
`ROADMAP_NS_MAGNETOSPHERE.md` "Strategic decisions" §1 for the full record
and measurements. Still not a maintained production path, but the
"prismatic mesh can't do GR" conclusion no longer holds.

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
damping_length    = 7        # outer absorber; without it waves reflect
damping_coef      = 0.08
use_flat_metric   = false    # REQUIRED: prismatic_sph_output defaults to flat
```

Both `use_static_background` and the outer absorber matter. Without
subtraction the relaxed state has ~9x too much flux through the horizon
(no expulsion at all); without damping, reflections off `r_max` dominate
and nothing settles.

`background_spin` must be **on-shell** (equal to `bh_spin`). Subtracting an
off-shell background is meaningless: its continuum time derivative is not
zero, so its discrete RHS is not pure truncation residual.

## Configs

- `config_L4_ic.toml`, `config_L5_ic.toml` — IC-only (`max_steps = 1`),
  dump `ic_aux.h5` for residual/convergence studies via `analyze_drift.py`.
  Set `field_spin = bh_spin` to measure the discrete-equilibrium residual at
  the analytic stationary state; leaving it at the default 0 measures the
  drift of an off-shell state, which is not a discretization error.
- `config_L5_check.toml` — IC-only for `check_ic.py`.
- `config_L5_a0.toml` — a = 0 stationarity test.
- `config_L5.toml` — full evolution.

## Diagnostics — read this before trusting a number

- `analyze_drift.py` loops over **all** shells `0..N_r`, including both
  ghost layers. The outermost (ghost) shell alone drags the measured Ampere
  convergence order from ~0.93 to ~0.65; exclude it for a bulk number.
- `analyze_drift.py` normalizes `dD/dt` by `|D|`. `D~` is a dual 2-cochain
  (~h^2) while `dD` is a signed sum of ~h^1 terms, so `dD/D ~ h^(p-1)`: a
  *constant* ratio already means first order, not zero order. For an
  order measurement prefer a normalization-free form, e.g.
  `||d1t H_aux|| / || |d1t| |H_aux| ||`.
- `check_ic.py` compares `E_aux` per edge against `A_0(v0) - A_0(v1)`. That
  reference is a pure discrete gradient and `d1 d0 = 0`, so the comparison
  is **gauge-dependent**: a large per-edge discrepancy on horizontal edges
  can be entirely curl-free and physically invisible. Judge Faraday by
  `d1 E_aux`, not by `E_aux` itself.
- Physical acceptance test: `Phi_cap(horizon)/Phi_cap(r=8)` from the
  tri-face flux cochain. Analytic target at a = 0.998, L3 is **0.00180**;
  as-shipped code relaxes to 0.0168, with the fixes to 0.0020.

Configs, drift diagnostics, and movies otherwise reflect the state of the
GR investigation as of mid-2026.
