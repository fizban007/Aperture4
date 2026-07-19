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
the backbone of the generalized angular axis: ownership tests are
O(1) arithmetic plus the 20-entry face-path table (F3), and each
owned unit is a contiguous tri block.

**F2 — Topology adjacency is already generic.** `icosphere_topology`
carries edge→2-tris, vertex→tri-fan, vertex→edges.  Incident *units*
of any sphere element follow by mapping its incident tris through
`unit()`.  The current per-ico-face ownership helpers
(`edge_owner_ico_face` etc.) are the m=0 special case.

**F3 — Angular unit & rank definition.**
- Choose patch level m and angular rank count A with A | 20·4^m.
  Units U = 20·4^m; angular rank a owns units
  [a·U/A, (a+1)·U/A) *in the canonical (path) ordering below* — i.e.
  a contiguous range of PATH-ordered units, each unit being a
  contiguous tri block in global index space (the blocks themselves
  are scattered per the face path when a rank owns several faces).
  A is any divisor of 2^(2m+2)·5, i.e. A ∈ {2^j, 5·2^j}.
- Examples: A=4 (m=0, 5 faces/rank) × K=2 → one 8-GCD node.
  A=80 (m=1) × K=100, or A=320 (m=2) × K=25 → 8000 ranks.
- Ownership rule for shared sphere-edges/vertices: **lowest incident
  unit owns** (min over incident tris of unit(t)).  Reduces exactly to
  the current lowest-ico-face rule at m=0/A=20 — the bit-exactness
  regression anchor.
- World rank = radial_rank·A + angular_rank (generalizes ·20).
- **Canonical unit ordering (REQUIRED, decided 2026-07-18).**  Units
  are ordered by (face position along a fixed Hamiltonian cycle of
  the icosahedron's face-adjacency graph — the dodecahedral graph,
  which is Hamiltonian; a static 20-entry table) × (base-4 child
  order within the face).  Guarantees: any whole-face group
  (A ∈ {1,2,4,5,10,20}) is a CONNECTED band — including the 4×2
  single-node shape; any aligned power-of-4 range (A = 20·4^m) is a
  single connected patch; mixed cases (e.g. A=8 → ten quarter-face
  units) are connected up to at most one boundary sliver and correct
  regardless.  Rationale: disconnected groups cost halo surface
  (ghost-set size, peer count, traffic — potentially a few ×), never
  correctness or dense allocations; the ordering removes that tax
  for the configurations we will actually run.  Ownership and
  migration-destination stay O(1) arithmetic + the 20-entry table
  (`path_pos(face)·4^m + child_bits`); a rank's owned tris become a
  short list of per-unit contiguous blocks instead of one global
  range (injector bounds / local-mesh build generalize accordingly).

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
Destination: unit = global_tri >> 2(L−m); path unit =
path_pos(face)·4^m + child_bits (20-entry table); angular rank =
path_unit·A/U (uniform path ranges ⇒ arithmetic); radial rank via
base/rem as today; world = rad·A + ang.  Wire format carries GLOBAL cell ids
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

**F9 — Output architecture (decided 2026-07-18): mesh-native dumps +
post-processing; the sph system leaves the production path.**
- The exporter becomes the ONLY production output system.  It grows
  to write J, rho, rho_abs, gamma_wsum alongside E/B (the sph system
  was the only consumer of the moment fields), all via the existing
  parallel owned-runs machinery.
- **Downsampled output = TRUE coarse cochains by chain-map
  aggregation**, not fine-value sampling: coarse sphere edge = sum of
  its 2 fine half-edges (normalized midpoints lie on the parent arc,
  so the halves tile it exactly); coarse tri face = 4 children;
  coarse v-edge = R stacked fine v-edges; coarse rect = 2R fine
  rects; j angular levels = recursive base-4 child sums.  Because
  aggregation is the transpose of refinement it COMMUTES WITH THE
  DISCRETE d: the dump satisfies the coarse Gauss law / Faraday
  consistency / charge conservation exactly and is a bona fide
  level-(L−j) DEC field.  Vertex moments restrict with
  partition-of-unity hat weights (vanishing fine vertices
  redistribute charge to their coarse neighbors; total charge exactly
  conserved) — one documented convention.  Cost: a few adds per
  output element at output cadence.
- Ownership of aggregation sums: angular is automatically safe (a
  coarse tri's 4 children share a parent ⇒ same unit for any
  m ≤ L−j; a boundary coarse edge's halves lie on the same boundary
  arc ⇒ same owner).  Radial coarse elements can straddle slab
  boundaries: DEFAULT constraint = slab boundaries aligned to
  multiples of the radial stride when aggregated output is enabled
  (checked at init); documented fallback = owner-sums-partials
  reduce over the sparse boundary set at output cadence.
- Dump format becomes self-describing ("level L−j icosphere, radii
  subset [...]"); post tools build the coarse mesh themselves — the
  `output_*_idx` kept-index maps disappear.
- **Post-processing replaces in-code sph**: `sph_from_dump.py`
  (Whitney E/J + vertex-recovery B on the dump's mesh → (r,θ,φ)
  grids, reproducing today's sph datasets; prototypes exist in
  `python/prismatic_recovery.py` and the `prismatic_interp` pybind
  module) and a Cartesian-box variant for 3D visualization (accuracy
  ↔ viz-convenience trade, chosen in post, not at run time).
  Validation: in-code sph vs post-reconstruction on identical
  full-resolution single-rank dumps.
- The sph system's distributed gather path is DELETED (the one
  knowingly non-scalable stage goes away entirely); the system stays
  compiling for single-rank convenience runs only.
- mesh.h5: parallel-written via owned runs for 3D datasets + rank-0
  for sphere-level data (post-F8 no rank holds global 3D arrays).
- Quantitative diagnostics (luminosity, recurrence) keep running
  from full-resolution dumps at chosen cadence, as today.

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
   {patch_level m, unit_lo, unit_hi} in the CANONICAL unit ordering
   (Hamiltonian face path × base-4 children — the static table and
   its inverse land here); ownership predicates via min-incident-unit
   through the topology tri fans; factories
   `combined(L, N_r, A, K, rank)` with A | 20·4^m validation.
   A=20/m=0 must reproduce today's owned sets EXACTLY (unit test:
   compare layouts against the old rule on all cochains; note the
   identity-vs-path face order does not matter at A=20 since every
   face is its own rank).  Include a unit-connectivity unit test:
   every rank's owned unit set is connected for whole-face groups
   and single-patch configurations.
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

### 7D — Partition-aware mesh build + output rework (~1.5 weeks)

1. Split `prismatic_mesh::build()`: sphere stage (global, cheap) /
   3D stage (loop bounds from the partition; global path preserved
   when no partition given).  Solver + particle local builds consume
   it; assert (debug) that no (N_r+1)·N_s array is allocated under a
   partition.
2. Exporter: add J/rho/rho_abs/gamma_wsum datasets; aggregated
   coarse-cochain downsampling (F9) with the slab-alignment check;
   self-describing dump metadata (L_out, radii subset); mesh.h5 →
   parallel owned-runs for 3D datasets, rank-0 for sphere data.
3. Post tools: `sph_from_dump.py` (Whitney E/J + recovery B on the
   dump's mesh → sph datasets) + Cartesian-box variant; validation
   against in-code sph on identical single-rank full dumps.
   Aggregation unit test: coarse Gauss law / total charge hold
   EXACTLY on aggregated dumps (the property sampling never had).
4. Delete the sph system's distributed gather path (single-rank
   convenience use remains); ns_rotator/vacuum_dipole mains updated.
5. Memory audit: per-rank device+host footprint logged at init;
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
- **Aggregated-output radial alignment**: the slab-boundary ∝ stride
  constraint must fail loudly at init (not silently produce partial
  sums); the fallback boundary reduce, if implemented, needs its own
  unit test against a global aggregation reference.
- **Post-tool parity**: `sph_from_dump.py` must reproduce the in-code
  sph datasets on identical inputs before the distributed gather path
  is deleted — the diagnostics pipeline (luminosity, recurrence,
  rotator scorecards) depends on those datasets.

## Appendix — files touched per phase (orientation for fresh sessions)

- 7A: `prismatic_partition.h/.cpp` (unit descriptor, path table,
  ownership), `icosphere_topology.h/.cpp` (incident-unit queries),
  `prismatic_halo_plan.h/.cpp` (generic angular builder, canonical
  ordering), `prismatic_mpi_comm.h/.cpp` (A·K create, dist-graph),
  `prismatic_mesh_partition.cpp` (plan wiring), tests:
  `test_prismatic_partition`, `test_prismatic_halo_plan`,
  `test_prismatic_dec_dist`, `test_prismatic_solver_multirank`.
  Callers of `combined(...)`: `dec_field_solver_impl.hpp`,
  `ns_rotator.cpp`, `test_prismatic_pic_multirank.cpp`.
- 7B: `prismatic_halo_plan.*` (pic depth class),
  `prismatic_cochain_layout.cpp` (ghost sets),
  `prismatic_mpi_halo_backend.h/.cpp` + `prismatic_halo_exchanger.h`
  (reduce()), new operator test alongside `test_prismatic_d1_local`.
- 7C: new `prismatic_ptc_mesh_local.h/.cpp` (+ptrs),
  `prismatic_ptc_update_kernel.hpp`, `prismatic_ptc_updater.*`,
  `prismatic_ptc_injector.hpp`, `prismatic_surface_injector.h`,
  `prismatic_vertex_recovery.*` (local rows + Bv exchange),
  `prismatic_particles.h` (dest helper), DELETE
  `prismatic_field_replicator.h` + `prismatic_field_sync.h`,
  `dec_field_solver.*` (injected partition ref),
  `ns_rotator.cpp`, `test_prismatic_pic_multirank.cpp`.
- 7D: `prismatic_mesh.h/.cpp` (staged build),
  `prismatic_data_exporter.*` (moments + aggregation + metadata),
  `prismatic_sph_output.*` (gather-path deletion), new
  `python/sph_from_dump.py` / `python/cart_from_dump.py`
  (start from `python/prismatic_recovery.py` + `prismatic_interp`).
- Validation entry points: `./check_prismatic.sh`;
  `mpirun --oversubscribe -n <8|20|40|80> bin/test_prismatic_solver_multirank`;
  `mpirun --oversubscribe -n <N> bin/test_prismatic_pic_multirank -c
  tests/config_prismatic_pic_multirank.toml` (also run with -n 1 and
  compare `step_*/sph_*` HDF5 datasets between the two output dirs at
  ≤1e-4 relative; LIVE/MISOWNED/ESUM printed by the test must match).
  NOTE: new CMake test targets need a reconfigure
  (`cmake -S . -B build`) before `--build --target` finds them.

## Deltas vs. prior plan documents

- PARALLELIZATION_PLAN Phase 6 ("particle depositor halo… defer") is
  implemented here as F4–F7; its reduce() sketch is 7B.
- The Phase-6 replicated-field implementation (`7f2ce6078`) is
  scaffolding: its acceptance tests, migration Alltoallv machinery,
  output path, and exchanger survive; replicator/field-sync die in
  7C.5.
- 4.1a.4 moves from "deferred, only needed at L≈7" to REQUIRED (7D):
  the replicated-mesh assumption is what capped the design.
