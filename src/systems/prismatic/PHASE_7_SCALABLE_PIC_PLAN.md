# Phase 7 — Generalized angular decomposition + fully-local PIC

> **Goal.** Retire the two scaling blockers at once: (1) the angular
> axis is hardwired to exactly 20 ranks (one ico-face each), capping
> total ranks at 20·K and making a single 8-GCD Frontier node
> unusable; (2) the particle path runs on replicated global fields and
> every rank holds the full global 3D mesh, capping the whole design
> at roughly L6.  End state: A·K ranks with A spanning ~1 to ~10^3+
> (target: 8000 GCDs ≈ 1000 Frontier nodes), all field AND particle
> state O(local) per rank, no global 3D arrays, no replicated fields.
>
> Supersedes the Phase-6 replicated-field architecture (commit
> `7f2ce6078`), which stays in-tree as the reference implementation to
> diff against until 7C lands, then is deleted.

## 0. Design foundations (verified against the tree)

**F1 — Patch structure of triangle indices.** `subdivide()` pushes the
4 children of each parent consecutively, so after L levels a triangle
index reads as base-4 digits `(ico_face, c1, ..., cL)`.  Consequence:
every *level-m patch* (m ∈ [0, L]) is a contiguous block of 4^(L−m)
triangles, and `unit(t) = t >> 2(L−m)` is pure arithmetic.  This is
the backbone of the generalized angular axis: no lookup tables in hot
paths, injector/migration ownership tests stay contiguous-range
checks.

**F2 — Topology adjacency is already generic.** `icosphere_topology`
carries edge→2-tris, vertex→tri-fan, vertex→edges.  Incident *units*
of any sphere element follow by mapping its incident tris through
`unit()`.  The current per-ico-face ownership helpers
(`edge_owner_ico_face` etc.) are the m=0 special case.

**F3 — Angular unit & rank definition.**
- Choose patch level m and angular rank count A with A | 20·4^m.
  Units U = 20·4^m; angular rank a owns units
  [a·U/A, (a+1)·U/A) — a contiguous unit range ⇒ a contiguous tri
  range.  A is any divisor of 2^(2m+2)·5, i.e. A ∈ {2^j, 5·2^j}.
- Examples: A=4 (m=0, 5 faces/rank) × K=2 → one 8-GCD node.
  A=80 (m=1) × K=100, or A=320 (m=2) × K=25 → 8000 ranks.
- Ownership rule for shared sphere-edges/vertices: **lowest incident
  unit owns** (min over incident tris of unit(t)).  Reduces exactly to
  the current lowest-ico-face rule at m=0/A=20 — the bit-exactness
  regression anchor.
- World rank = radial_rank·A + angular_rank (generalizes ·20).
- Multi-unit ranks (A < U) may own geometrically disconnected unit
  sets depending on face ordering; this is *allowed* (plans are
  generic).  An optional unit-permutation table for connected groups
  is a later nicety, config-selectable, default identity.

**F4 — Ghost (halo) element sets, parameterized by depth class.**
Two stencil classes, chosen at partition-build time:
- `solver` (what exists today): d1/d1t edge-adjacent depth-1.
- `pic`: superset driven by the particle kernels —
  - T_own = owned tris; T_halo = all tris sharing ≥ 1 sphere-vertex
    with T_own (the 1-ring; covers barycentric walks around corners,
    including valence-5).
  - Ghost cochain sets = elements incident to T_halo prisms, on
    shells [k_lo−1, k_hi+1] (radial depth 1 both sides):
    h/v edges + vertices (deposit + E-gather targets of a particle
    that ends the step one cell over), tri/rect faces (crossing
    deposit; recovery fans of owned vertices are a subset).
  - The `pic` set contains the `solver` set, so a PIC run builds ONE
    layout per cochain and the field solver exchanges the slightly
    larger halo (a corner-elements increment; revisit only if
    profiles object).
- CFL contract: a particle may not exit T_halo in one step (field CFL
  keeps c·dt below the min edge length, and the 1-ring is ≥ one full
  cell deep everywhere).  The pusher asserts (debug) / absorbs with a
  loud counter (release) on violation.

**F5 — Local indexing scheme for the particle path (decision).**
The solver's cochain layouts ("owned ascending global, then ghosts")
are freshly validated — do NOT churn them.  Particle kernels instead
get a *tensor-product local sphere space*:
- Local sphere numbering: owned tris first (preserving global order),
  then T_halo ring; same for sphere-edges/vertices (owned then ghost,
  each sorted by global index).  Local sphere tables (`tri_verts`,
  `tri_edges_s`, `tri_neighbor`, vertex coords, edge endpoints,
  recovery weight rows) are the global tables remapped once at build;
  `tri_neighbor` entries beyond T_halo become −1.
- Kernels keep their arithmetic shape: tensor index =
  local_k·n_s_loc + local_s (identical structure to today's global
  k·N_s + s).
- One `int` map per cochain, `tensor_to_layout[]` (size = local
  cochain count), converts a tensor index to the solver-layout index
  used by the actual field buffers.  One extra indirection per access;
  memory = one int per local element.  This keeps a SINGLE field
  allocation per cochain, shared by solver and particles, with zero
  solver changes.
- Particle `cell` becomes LOCAL (local_layer·n_tri_loc + local_tri):
  smaller sort keys, no per-particle global decode.  Global cell ids
  appear only on the migration wire (F7).

**F6 — Deposit reduction (the real one).** Particles deposit into the
local J/rho buffers (owned + ghost slots, atomics as today).  New
`reduce()` path — the exact reverse of `exchange()`: ghost slots pack
→ MPI to owner → owner **+=** on unpack, then ghosts are zeroed.
Same plans, same packed device staging as the Phase-5 exchanger, same
wire-compatibility guarantee.  O(halo) traffic; the global allreduce
dies here.

**F7 — Migration under A·K.** Existing Alltoallv machinery survives.
Destination: unit = global_tri >> 2(L−m); angular rank = unit·A/U
(uniform ranges ⇒ arithmetic); radial rank via base/rem as today;
world = rad·A + ang.  Wire format carries GLOBAL cell ids
(rank-agnostic); pack translates local→global (table lookup), unpack
global→local (binary search in the owner's sorted sphere l2g +
layer arithmetic; host-side, per arrival — cheap at output-step
counts).  `prism_migrate_dest` generalizes from (20, tris_per_face)
to (A, m) parameters.

**F8 — Partition-aware mesh build (4.1a.4, now in scope).**
The global-per-rank memory splits into:
- Angular (sphere) tables: O(4^L) — ~10s of MB even at L8.  These
  STAY global on every rank (subdivision is serial and cheap); this
  is what makes local table remapping and sph output tractable.
  Distributed subdivision is explicitly out of scope (L9+ problem).
- 3D per-cochain arrays ((N_r+1)·N_s-sized: vertex positions,
  edge_length, face_area, hodge, d1 CSR, dual volumes): these are the
  GB-scale offenders and become local-only.  All are computed
  elementwise from sphere data × radii (analytic in r), so local
  construction is a loop-bounds change, not new math.
- End state: with a partition, `prismatic_mesh::build()` never
  allocates a global 3D array; the existing global path remains for
  single-rank.  `release_global_buffers()` becomes moot (nothing to
  release).

**F9 — What the output path needs.** After F8, no rank holds global
3D arrays:
- Exporter: already writes owned runs via parallel HDF5 — unaffected.
- mesh.h5: currently rank-0 serial from global arrays.  Becomes
  (a) parallel-written via the same owned-runs machinery for 3D
  datasets + rank-0 for sphere-level data, or (b) regenerated offline
  by a single-rank tool.  Choose (a); it is a mechanical reuse.
- sph output: rank-0 gather of global *cochains* stays (output
  cadence only), but its h1inv dual→primal conversion reads a global
  3D array — replace with the analytic per-edge evaluation (sphere
  tables × radii), which rank 0 can compute on the fly.  Parallel
  sph interpolation (each rank interpolating its owned grid points)
  is the eventual answer at 8000 ranks; keep it a separate,
  later work item — output cadence makes the gather tolerable
  meanwhile.

**F10 — Load balance.** Fields: exact by construction (units are
congruent, shells uniform per slab).  Particles: radially concentrated
(r < 5–10) and, for oblique rotators, latitude-concentrated ⇒ prefer
angular-major shapes (large A, modest K).  Config hook for
NON-uniform slab boundaries (explicit shell-list) is cheap and worth
adding while touching the factories; automatic rebalancing is out of
scope.

## Phasing (each phase leaves the tree green; commit ≈ one bullet)

### 7A — Generalized angular partition, fields only (~1 week)

1. `prismatic_partition`: replace {ico_face_lo, ico_face_hi} with
   {patch_level m, unit_lo, unit_hi} (+ derived tri range); ownership
   predicates via min-incident-unit through the topology tri fans;
   factories `combined(L, N_r, A, K, rank)` with A | 20·4^m
   validation.  A=20/m=0 must reproduce today's owned sets EXACTLY
   (unit test: compare layouts against the old rule on all cochains).
2. `icosphere_topology`: incident-unit queries (thin wrappers mapping
   incident tris / faces through `unit()`); drop nothing.
3. Generic `build_angular_halo_plan`: topology-driven, per-peer
   send/recv lists sorted by global index on BOTH sides (canonical
   order replaces today's matched per-ico-edge iteration — this is
   what makes arbitrary unit adjacency safe, including valence-5
   corners spanning 5 units and ordinary patch corners at valence-6
   vertices).  Ghost-set input comes from the depth-class spec (F4),
   defaulting to `solver`.
4. `prismatic_mpi_comm::create(world, A, K)`; angular sub-comms of
   size A; dist-graph neighbors from the partition's unit adjacency;
   world = rad·A + ang.
5. Validation: `test_prismatic_dec_dist` (in-process) and
   `test_prismatic_solver_multirank` (real MPI) extended to
   A ∈ {4, 20, 80} × K ∈ {1, 2} — bit-exact vs single-rank, all three
   scenarios.  8-rank (4×2) is the single-node Frontier shape and
   becomes a permanent CI configuration.

### 7B — PIC-depth ghosts + reduce() (~3–4 days)

1. Ghost-set builder for the `pic` depth class (T_halo 1-ring +
   shells ±1); layouts rebuilt; solver tests rerun (still bit-exact —
   ghosts are exact copies regardless of set size).
2. `reduce()` on `mpi_halo_backend` + exchanger (+ in-process backend
   for unit tests): reverse the plan direction, owner accumulates,
   ghosts zeroed after.  Packed device path from day one (the
   exchange scaffolding generalizes; keep the `halo_device_direct`
   fallback semantics).
3. Operator-level test (the 4.1b B0.2 pattern): synthetic per-rank
   deposits into owned+ghost slots, reduce, compare against a global
   single-rank sum — all cochains, A ∈ {4, 20, 80} × K, including
   valence-5 corners and slab boundaries.

### 7C — Local particle mesh + kernel conversion (~1.5–2 weeks, the core)

1. `prismatic_ptc_mesh_local` (+ POD ptrs): local sphere numbering,
   remapped sphere tables, localized recovery-weight rows,
   per-cochain `tensor_to_layout` maps, l2g/g2l for migration.
   Unit test: every remapped table entry equals its global
   counterpart under l2g.
2. Kernel conversion: `update_single_particle`, deposit, gathers,
   `find_triangle`/`cartesian_to_local` take the local ptrs bundle;
   `cell` goes local.  Landmark: single-rank local == single-rank
   global bit-exact (identity partition ⇒ identity maps — the same
   trick 4.1b used).
3. Recovery Bv goes distributed: compute Bv on owned vertices (fans
   ⊂ T_own ∪ T_halo faces), then exchange Bv as 3 scalar vertex-halo
   exchanges (avoids depth-2 face halos).  New sync point at the top
   of the updater.
4. Updater rewire: clear local J/rho → push+deposit (local) →
   `reduce()` J/rho/diagnostics → migrate (generalized dest + cell
   translation) → sort (max_cell now local).  Injector loops owned
   cells only (bounds already contiguous), reads local fields
   directly — the "*_ptc" replica indirection disappears.
5. Delete `prismatic_field_replicator`, `prismatic_field_sync`, the
   replica registrations and updater/injector replica branches;
   ns_rotator/test mains updated.
6. Acceptance: `test_prismatic_pic_multirank` extended to
   {1, 4×2, 20×2, 80×1(L≥3)} — count conservation, zero misowned,
   field/sph agreement vs serial at FP-reordering tolerance; plus a
   migration-stress config (large p0) to exercise multi-unit hops.

### 7D — Partition-aware mesh build (~1 week)

1. Split `prismatic_mesh::build()`: sphere stage (global, cheap) /
   3D stage (loop bounds from the partition; global path preserved
   when no partition given).  Solver + particle local builds consume
   it; assert (debug) that no (N_r+1)·N_s array is allocated under a
   partition.
2. mesh.h5 → parallel owned-runs for 3D datasets, rank-0 for sphere
   data; sph h1inv conversion → analytic per-edge evaluation.
3. Memory audit: per-rank device+host footprint logged at init;
   weak-scaling smoke (L fixed, A·K ∈ {8, 40, 160}) asserting
   footprint ~ 1/(A·K) + angular-table constant.

### 7E — Frontier readiness (~2–3 days)

1. Configs + launch notes for: 1 node (4×2), 5 nodes (20×2),
   40 nodes (320×1), 1000 nodes (320×25 or 80×100); rank-placement
   map so co-noded ranks are geometric neighbors (placement file,
   not code).
2. Scaling harness stub (timers already exist per system) — actual
   measurement campaign stays deferred per the 2026-07-18 decision.

## Risks / invariants to guard

- **Plan-order matching**: the canonical sort-by-global-index on both
  peers replaces implicit matched iteration; the in-process backend's
  positional matching makes any mismatch fail loudly in 7A tests.
- **Valence-5 corners at m ≥ 1**: a corner vertex's fan spans corner
  patches of 5 different faces (up to 5 distinct ranks); the generic
  min-incident-unit rule handles it, but tests must pin it
  explicitly (the historical bug magnet).
- **A=20 bit-exactness anchor**: every 7A/7B commit reruns the 4.1b
  acceptance tests; the generalized code must reproduce the current
  owned sets and wire traffic exactly at A=20/m=0.
- **Particle walk beyond T_halo**: asserted; the CFL argument (F4)
  makes it structurally impossible for physical dt — treat any hit as
  a bug, not a tolerance.
- **Two mesh_partition instances** (solver-internal + main-built for
  particle systems) must stay identical: replace the solver's
  internal build with an injected partition reference while touching
  its constructor in 7C.5 — removes the duplication instead of
  hoping.
- **sph rank-0 gather** becomes the known non-scalable output stage;
  acceptable at output cadence, flagged for parallel interpolation
  later.  Do not let it silently become a per-step cost.

## Deltas vs. prior plan documents

- PARALLELIZATION_PLAN Phase 6 ("particle depositor halo… defer") is
  implemented here as F4–F7; its reduce() sketch is 7B.
- The Phase-6 replicated-field implementation (`7f2ce6078`) is
  scaffolding: its acceptance tests, migration Alltoallv machinery,
  output path, and exchanger survive; replicator/field-sync die in
  7C.5.
- 4.1a.4 moves from "deferred, only needed at L≈7" to REQUIRED (7D):
  the replicated-mesh assumption is what capped the design.
