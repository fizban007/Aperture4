# Phase 4.1b Plan — Solver Buffer Conversion

> **STATUS (2026-07): RETARGETED.**  GR work is shelved
> (`ROADMAP_NS_MAGNETOSPHERE.md`); this conversion now applies to the flat
> `dec_field_solver`, not `dec_field_solver_gr_ks`.  The buffer splits,
> commit boundaries, and gotchas below carry over, but the shift
> cross-terms — and their halo dependencies (B before compute_dB_dt,
> D̃-tangent before compute_dD_dt) — do not exist in the flat solver:
> exchange E (h+v) before Faraday and B (tri+rect) before Ampère, plus a
> refresh inside each semi-implicit Picard iteration.  Wald-IC and
> horizon-damping sections apply only if the GR solver is ever revived.

## Context

Phase 4.1a finished: every per-cochain mesh array (mesh + metric + d1/d1^T)
has a tested local form sized to the rank's owned + halo regions, plus
a `prismatic_mesh_partition` bundle holding layouts and pre-localized
halo plans for all 5 cochain types.  The infrastructure is ready;
4.1b ports the solver to consume it.

## Goal

Replace `dec_field_solver_gr_ks`'s flat global-indexed buffers and
mesh accesses with per-cochain local-indexed buffers, so that running
on N MPI ranks each consumes ≈ 1/N of the per-cochain field memory
plus its small halo.

End state: `dec_field_solver_gr_ks::update()` runs entirely on local
data, calling `mp_halo_backend.exchange()` at the right sync points
to keep ghost slots fresh.

## Files to be touched

- `src/systems/prismatic/prismatic_field_data.h` — `prismatic_edge_field`
  and `prismatic_face_field` size to local layouts.
- `src/systems/prismatic/dec_field_solver_gr_ks.h` — buffer sizes and
  mesh-partition reference.
- `src/systems/prismatic/dec_field_solver_gr_ks_impl.hpp` — every
  kernel in compute_dB_dt / compute_dD_dt / update_* / set_initial_*
  / apply_*_boundary / apply_damping / dump_aux_fields.
- `src/systems/prismatic/prismatic_mesh_metric_ptrs.h` — local-ptrs
  struct that solver kernels pass to GPU lambdas.
- `src/systems/prismatic/prismatic_mesh.h/.cpp` — eventually a
  `release_global_buffers()` (4.1a.4 — depends on this phase).

## Order of operations

### 4.1b.0 — Local mesh ptrs

Add a `prismatic_mesh_local_ptrs` struct mirroring the existing
`prismatic_mesh_ptrs` / `prismatic_mesh_metric_ptrs` but pointing into
the local arrays in `prismatic_mesh_local`, `prismatic_mesh_metric_local`,
and `prismatic_d1_local`.  Pass-by-value into GPU lambdas as today.

Fields:

```
struct prismatic_mesh_local_ptrs {
  // Per-cochain layout sizes.
  int n_owned_tri,  n_local_tri;
  int n_owned_rect, n_local_rect;
  int n_owned_he,   n_local_he;
  int n_owned_ve,   n_local_ve;
  int n_owned_v,    n_local_v;

  // Per-cochain mesh data (face_area, edge_length, hodge1_inv, hodge2,
  // etc.) split by cochain type (tri_face_area, h_edge_length, ...).

  // Per-cochain metric data (alpha, sq_gamma_beta_r, sqrt_gamma, ...).

  // d1 / d1^T sparse blocks (row_ptr / col_idx / val pointers for each
  // of the 6 blocks).
};
```

Document carefully which arrays have global-vertex indices in their
VALUES (e.g. h_edge_v0/v1, tri_face_v0/v1/v2): under partitioning we
keep these as global indices for now and translate to local via the
vertex layout's `to_local` only at the call site that needs it.

### 4.1b.1 — Solver scratch buffers

The solver's scratch and auxiliary buffers (`m_E_aux`, `m_H_aux`,
`m_dD_dt`, `m_dB_dt`, `m_tmp_D`, `m_tmp_B`, `m_dD_dt_new`,
`m_dB_dt_new`, `m_D_bg`, `m_B_bg`) are sized to global edge / face
counts.  Convert each to TWO buffers (h_edge + v_edge for edge-side,
tri_face + rect_face for face-side) sized to layout.local_size().

Keep the same naming pattern: split `m_E_aux` → `m_E_aux_h`, `m_E_aux_v`;
`m_H_aux` → `m_H_aux_tri`, `m_H_aux_rect`; etc.

This step is mechanical and self-contained; doesn't change kernels yet
(they still use the unsplit buffers — which now don't exist, so the
kernels won't compile until 4.1b.2 lands).  Better to do this as part
of 4.1b.2.

### 4.1b.2 — compute_dB_dt rewrite

The Faraday half-step computes:

```
  E_aux[e] = α[e] · hodge1_inv[e] · D̃[e] + shift_cross_term(e)
  dB[f] = -Σ_e d1[f,e] · E_aux[e]
```

Split into 4 kernel passes:

1. **E_aux on h_edges**: lapse + shift cross term using d1t_h_rect to
   walk adjacent rect faces.  Output: `m_E_aux_h[0 .. local_he)`.
2. **E_aux on v_edges**: pure lapse term.  Output: `m_E_aux_v`.
3. **dB on tri_faces**: `dB[f] = -Σ d1_tri_h[f,e] · E_aux_h[e]`.  Output:
   `m_dB_dt_tri`.
4. **dB on rect_faces**: `dB[f] = -Σ_h d1_rect_h[f,e] E_aux_h[e]
   - Σ_v d1_rect_v[f,e] E_aux_v[e]`.  Output: `m_dB_dt_rect`.

**Halo exchanges needed before this**:
- D̃[e] for h_edges and v_edges (for E_aux computation reading owned + halo D̃).
- B[f] for rect_faces (for the shift cross term in E_aux on h_edges,
  which reads B at adjacent rect faces — some owned, some halo'd).

The shift-cross-term loop currently iterates `mp.d1t_row_ptr[e]..d1t_row_ptr[e+1]`
filtering for `is_tri_face`.  Replaced by walking `d1t_h_rect` directly.

### 4.1b.3 — compute_dD_dt rewrite

Ampère half-step:

```
  H_aux[f] = α[f] · hodge2[f] · B[f] + shift_cross_term(f)
  dD̃[e] = Σ_f d1t[e,f] · H_aux[f] - J̃[e]
```

Same 4-pass split:

1. **H_aux on tri_faces**: pure lapse term.  Output: `m_H_aux_tri`.
2. **H_aux on rect_faces**: lapse + shift cross term using d1_rect_h
   (or equivalent) to walk adjacent h_edges.  Output: `m_H_aux_rect`.
3. **dD̃ on h_edges**: `Σ d1t_h_tri · H_aux_tri + Σ d1t_h_rect · H_aux_rect - J̃`.
4. **dD̃ on v_edges**: `Σ d1t_v_rect · H_aux_rect - J̃`.

**Halo exchanges needed before this**:
- B[f] for tri_face and rect_face (covered if updated together with B
  in compute_dB_dt's pre-halo).
- D̃[e] for h_edges (for the shift cross term in H_aux on rect_faces,
  which reads D_tangent on adjacent h_edges).

### 4.1b.4 — set_initial_kerr_wald

The IC integrates analytic A_μ along edges and analytic D^i flux
through dual faces.  Iterate owned edges only, write into the local
D̃ / B buffers.  Then halo-exchange D̃ and B once so non-owners get
the right ghost values.

The vertex-coord lookups (`mp.sphere_vx[s]` etc.) stay as global
sphere-side indices since sphere-side data is replicated.

### 4.1b.5 — Boundary / damping routines

- `apply_inner_boundary`: modifies field values on the innermost shell.
  Innermost shell is owned by exactly one rank (radial slab 0); other
  ranks no-op.
- `apply_outer_boundary`: pins outermost shell to background.  Same
  pattern; only the last radial slab acts.
- `apply_damping`: exponential absorption over the outer N shells.
  Each rank handles the shells in its owned range that fall inside
  the damping zone.  Compares against `m_D_bg` / `m_B_bg` (which are
  also local now).
- `apply_inner_damping`: same logic for the inner damping layer.

All four operate purely on radial-axis logic; no angular halo needed.
A radial halo MAY be needed for the comparisons against background
fields, but typically the background is set at IC time and has correct
values everywhere, so probably not.

### 4.1b.6 — Field data registration

`prismatic_edge_field` and `prismatic_face_field` (in
`prismatic_field_data.h`) currently allocate buffers sized to
`m_mesh.m_N_edges` / `m_mesh.m_N_faces`.

Two plausible designs:

(a) Each `prismatic_edge_field` holds TWO local buffers (h + v).  API
    accessors `.h()` and `.v()` give pointers.  Solver kernels iterate
    one or the other.

(b) Replace with separate `prismatic_h_edge_field`, `prismatic_v_edge_field`,
    `prismatic_tri_face_field`, `prismatic_rect_face_field` types.
    Cleaner type-safety; more code churn at registration sites.

Lean toward (a) for less call-site disruption.  The framework's data
registry doesn't care about the internal split as long as the
container still supports `.copy_to_device()` / `.host_ptr()`.

Hodge1_inv conversion (D̃ ↔ D[e]_primal in sph_output) becomes per-axis.

### 4.1b.7 — dump_aux_fields

Currently writes raw global-indexed cochains to a single HDF5 dataset
per rank.  Under partitioning each rank only has its local slice.
Two options:

(i) Each rank dumps its local slice into a per-rank file (simplest).
(ii) Use parallel HDF5 — defer to Phase 5.

Pick (i) for now to keep 4.1b self-contained.  Diagnostic scripts
already in the repo (`analyze_drift.py`, `make_movie.py`) will need
updating, or a stitching script can recombine per-rank files.

## Halo exchange wiring (Phase 4.2 in plan terms — natural to do
together with 4.1b since otherwise the converted solver wouldn't have
correct ghost values)

In `update_explicit`:
```
  exchange D̃ on h_edge + v_edge axes (both angular and radial)
  exchange B on tri_face + rect_face axes
  compute_dB_dt(D̃, B, dB)
  B += dt · dB
  exchange B on tri + rect    (B updated)
  compute_dD_dt(D̃, B, dD̃)
  D̃ += dt · dD̃
```

Each `exchange` is a call to the partition's `mpi_halo_backend` (or
`in_process_halo_backend`) using the pre-localized angular and radial
plans from `prismatic_mesh_partition`.  6 exchanges per step total.

The `update_semi_implicit` path needs an exchange inside each Picard
iteration after the field is updated.

## Validation strategy

The existing `test_prismatic_mpi_backend_multirank` validates halo
exchange against an in-process oracle.  After 4.1b, add a similar
**solver multirank test**:

1. Set up the same Kerr-Wald IC on (a) a single-rank global solver
   and (b) a 20-rank distributed solver.
2. Run N steps.
3. Gather the 20-rank state into a single global view (via the
   layout's `local_to_global`) and compare cell-by-cell against the
   single-rank state.

Acceptance: max relative difference < 1e-4 for float-precision (semi-
implicit iterations and CSR sparse mat-vec aren't strictly bit-exact
across rank counts due to floating-point ordering).

For the Kerr-Wald stationary test specifically, the diagnostic
`analyze_drift.py` should produce the same drift-rate magnitude under
both single-rank and 20-rank runs at the same L / N_r.

## Risks & gotchas

- **Sphere-vertex indices in mesh_local fields** (e.g. `h_edge_v0[l]`
  holds global 3D vertex idx, not local).  Add a `vertex_layout.to_local`
  call site at any kernel that uses these — or pre-translate at build
  time.  Decide upfront and document.

- **valence-5 corner v_edge fan**: depends on the multi-incidence
  rect_face halo plan extension already in place (Phase 3.3, generalized
  in 4.1a.2).  The d1t_v_rect rows need 5 or 6 rect entries; ensure
  the layout includes all of them as ghosts.

- **Semi-implicit iteration**: each Picard step modifies B and D, so
  halos must refresh inside the loop, not just once before the loop.
  Easy to miss; results would silently drift on multi-rank runs only.

- **Order of operations in compute_dB_dt's shift-term**: currently the
  shift term reads B[f] at adjacent rect faces, which means B must be
  haloed BEFORE compute_dB_dt is called.  Currently the global solver
  doesn't think about this because B is global.  After conversion, the
  exchange call sites matter — see the update_explicit sketch above.

- **Damping toward background**: `m_D_bg` / `m_B_bg` were captured at
  IC time.  They're stored locally now too.  The damping kernel
  compares owned cells against owned bg cells — no halo needed, but
  must confirm the bg buffers cover the full owned range.

- **Boundary detection**: `apply_inner_boundary` checks `edge_radial_layer
  == 0`.  Under partitioning, only the rank owning shell 0 has those
  edges.  All other ranks must skip cleanly (no-op).

- **set_initial_kerr_wald uses the J̃ buffer**: J̃ is a third edge
  cochain (current density).  Same conversion as D̃.

## Suggested commit boundaries

1. `4.1b.0`: prismatic_mesh_local_ptrs + small lookup helpers.  No
   solver changes; just the ptrs API + tests verifying it points to
   the right local arrays.
2. `4.1b.1+2`: split scratch buffers + rewrite compute_dB_dt.  This
   is the biggest single commit; everything that was global about the
   Faraday side becomes local.  Add unit tests comparing local
   output against a reference single-rank run.
3. `4.1b.3`: rewrite compute_dD_dt analogously.
4. `4.1b.4`: set_initial_kerr_wald conversion.  At this point a full
   single-rank distributed solver should run end-to-end and produce
   results equivalent to the previous global path.
5. `4.1b.5`: boundary / damping routines.
6. `4.1b.6`: prismatic_edge_field / prismatic_face_field split.
7. `4.1b.7`: per-rank dump_aux_fields.
8. `multirank solver validation test`: bit-close comparison against
   single-rank reference at the end.

## Roughly out-of-scope for 4.1b (handle in Phase 5+)

- Parallel HDF5 output (Phase 5).
- Particle depositor / particle updater (Phase 6).
- Build-time partition awareness in `prismatic_mesh::build()`.  4.1b
  keeps the existing global-build-then-extract-local pattern.
  4.1a.4 (release_global_buffers) becomes useful only after this
  phase finishes — drop it in as a final cleanup commit.
