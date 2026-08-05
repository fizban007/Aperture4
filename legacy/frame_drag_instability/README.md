# Frame-drag W-term instability diagnostics (2026-08-04)

The persistent near-surface tangential-E layer in the fake-GR runs is a
**numerical grid-scale instability of the Faraday-side frame-drag coupling
W in the field solver alone** — it grows in vacuum with no plasma, its
growth rate scales ~1/h with resolution, and its spectrum is exactly
±symmetric in the W sign (which is why the broken-sign v1 run and the fixed
shift-only run saturate to identical layers). See the session memory note
`prismatic-frame-drag-w-instability.md` for the full evidence chain.

## Tools

- `vacuum_fd.cpp` — standalone single-rank host evolution of the DEC core
  (faraday with W, ampere, corotating inner BC, damping; no particles).
  Build (login node, no HIP):

      g++-13 -std=c++17 -O2 -march=native -Isrc \
        -Ideps/{fmt,mortonlib,visit_struct,cpptoml,cxxopts,gsl}/include \
        -Ideps/vectorclass1 -DFMT_HEADER_ONLY \
        legacy/frame_drag_instability/vacuum_fd.cpp \
        src/systems/prismatic/{prismatic_mesh,icosphere_topology,\
        prismatic_partition,prismatic_mesh_partition,prismatic_mesh_local,\
        prismatic_mesh_local_ptrs,prismatic_d1_local,\
        prismatic_cochain_layout,prismatic_halo_plan}.cpp \
        src/utils/logger.cpp src/core/buffer_impl.cpp -o vacuum_fd

  Usage: `vacuum_fd L N_r r_max n_steps dt noise_mode`.
  noise_mode=0: start from the analytic MT state (dipole B0, corot E) and
  watch per-shell mean |E − E_MT| on h-edges — blows up by step ~15k at L4.
  noise_mode=1: homogeneous system (B0=0, zeroed BC) seeded with 1e-6 noise
  — pure linear-stability test. Measured: γ_L4 ≈ 0.038, γ_L5 ≈ 0.096
  (ratio 2.5/level), extrapolating to γ_L6 ≈ 0.19 vs the production layer's
  measured 0.19–0.48.

- `dump_fd_ops.cpp` — dumps d1, W, hodge diagonals, boundary flags
  (same build line). `spectral_fd.py DUMP` computes the energy-norm growth
  bound: eigsh of −sym(h2·d1·W) in the h2 metric. Results: ±1.40 (L3),
  ±1.21 (L4), localized in shells k=1–3. `spectral_fd_filter.py` tests a
  face-graph filter inside W: bound 1.40 → 0.63/0.33/0.23 for 1/2/3 passes.

- `etheta_layer.py RUN_DIR STEP` — per-shell measured-vs-analytic (MT and
  flat corotation) comparison of h-edge circulations by colatitude band.
- `layer_history.py RUN_DIR` — time series of mean |E_e|, colat 10–20°.
- `layer_structure.py SPH_FILE`, `early_growth.py`, `rotation_profile.py`,
  `rho_profile.py` — sph-resample-based structure analysis (E∥/E⊥ split,
  colat/m-spectrum, ω(r) E×B profiles, charge densities). NOTE: the m~200+
  mode aliases to m~30–60 at the default 128-point φ resample; use
  `sph_from_dump.py --ntheta 128 --nphi 512` to resolve it.

- `vac_L4.txt`, `vac_L4_noise.txt`, `vac_L5_noise.txt` — run outputs behind
  the measured rates.

## The fix (validated in vacuum, 2026-08-05)

`vacuum_fd_adj.cpp` implements and tests the adjoint-paired scheme: the
dropped Ampère-side β×E term is restored as the exact transpose of the
existing W under the diagonal Hodge (H_aux = h2·B + Wᵀh1inv⁻¹E), which makes
U_full = ½Eᵀh1inv⁻¹E + ½Bᵀh2B + Eᵀh1inv⁻¹W·B an exact semi-discrete
invariant for any W — no Whitney mass matrices, no CG solve. Under leapfrog
both couplings must be midpoint-centred (2 Picard sweeps each; the `adj`
argument selects 0 = one-sided original, 1 = explicit adjoint, 2 = symmetric).

Measured (L4 noise test, 20000 steps): one-sided γ≈0.038; explicit adjoint
γ≈0.2 (worse); Ampère-centred only γ≈0.015; fully centred: monotone energy
decay, no growth (`adj2_L4_noise.txt`). MT-state run frozen at truncation
residual through step 20000 where the one-sided scheme exploded ~70×
(`adj2_L4_mt.txt` vs `vac_L4.txt`).
