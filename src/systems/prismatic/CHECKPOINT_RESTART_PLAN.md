# Checkpoint / restart for distributed prismatic PIC — implementation plan

> **Status: PLANNED 2026-07-19 (not started).**  Written to be executed
> in a fresh session.  Phase 7 (A–E) is complete: fields, particles and
> mesh geometry are fully local under an arbitrary canonical A·K
> decomposition; the exporter writes rank-agnostic global-indexed
> snapshots via parallel owned-runs; migration moves particles with
> GLOBAL cell ids over an Alltoallv.  This plan reuses all of that.
>
> **Motivation.** There is NO restart in the prismatic drivers.  The
> L6 production run t15 was SIGKILLed at 2.5 P and had to be declared
> dead — relaunch means from scratch (~3 h lost).  On clusters with
> wall-time limits this is untenable: checkpointing is required before
> any run longer than one allocation.  (See the t15 postmortem in the
> session memory / ROADMAP notes.)
>
> **Target scale: L10 (user decision 2026-07-19, revised down from an
> initial L12 ambition — L12 arithmetic kept in the appendix for
> reference).**  L10 means 20·4^10 ≈ 2.1×10^7 sphere tris (0.062° ≈
> 3.7 arcmin quasi-uniform spacing), N_r ≈ 3.3×10^3 shells (doubling
> per level from 204 at L6), ~6.9×10^10 prisms, O(10^12) macros —
> still a large step beyond current global pulsar PIC, and a
> realistic 1000–2000-node Frontier run (memory floor ~460 nodes at
> perfect packing).  Design consequences for THIS plan: (a) the
> on-disk global cell index must be 64-bit from day one — the uint32
> wall is at ~L9, BELOW the target (§D2); (b) checkpoint generations
> are ~10^2 TB — sizing and I/O notes in §D5; (c) per-rank
> O((N_r+1)·4^L) TIME loops are forbidden (hour-scale at L10), while
> per-rank O(4^L) MEMORY is ~GB at L10 — tolerable, which is what
> keeps the distributed sphere stage OFF the critical path (see the
> appendix ranking).

## Design decisions (agreed with the user 2026-07-19)

**D1 — Rank-agnostic, GLOBAL-indexed format.**  Everything in a
checkpoint is keyed by global indices (the exporter snapshot
convention), never by rank layout.  This is what buys the killer
cluster feature: **restart on a DIFFERENT rank count / decomposition**
than the writer (resubmit at whatever node count the queue offers).
Checkpoints are inspectable with the same tools as ordinary dumps.

**D2 — Store the minimal evolved state; recompute everything derived.**

Stored:

| dataset | shape | why |
|---|---|---|
| `Edelta`, `Bdelta` | global cochains | the solver's evolved state |
| `J`, `rho`, `rho_abs`, `gamma_wsum` | global cochains | not needed by the solver (recomputed each step), but the INJECTOR criteria read *last step's* J / rho_abs — storing them makes the first resumed step faithful |
| particles: `x1..3`, `p1..3`, `E`, `weight`, `cell` (GLOBAL encoding, **stored as uint64**), `flag`, `id` | one concatenated global dataset, per-rank slices | live particles only (skip `empty_cell` slots); cells translated by the existing `wire_cell()` migration encoding, WIDENED to 64-bit on disk — the in-memory encoding is `uint32_t` and global cell = k·N_tri + tri overflows 32 bits near L9 (5.2M tris × ~1.6k shells ≈ 8.6e9 > 2^32); baking uint32 into the format would poison every checkpoint written before the wire widening lands |
| rng states | per WRITING rank | exact streams on same-count restart (see D4) |
| per-rank particle `id` counters | per writing rank | id continuation |
| `step`, `time`, config fingerprint (`L`, `N_r`, `r_min`, `r_max`, `dt`, writer's `A`, `K`, world size) | scalars | resume point + loud sanity validation on load |

Recomputed at restart (deterministic from config): the mesh
(sphere-only under MPI), E0/B0 static background (analytic
quadratures), totals E = E0 + delta via `refresh_total_fields`,
layouts/plans/local meshes, recovery weights, Bv (first `sync_fields`
rebuilds it).  Solver Picard temporaries are per-step — nothing else is
stateful across steps.  Injector `m_throttled` is transient.
Exporter/sph `m_time` accumulators must be SEEDED from the restored
time (they integrate `+= dt`).

**D3 — Reuse existing machinery in both directions.**
- *Fields out:* the exporter's `write_parallel_runs` owned-runs path —
  identical pattern, different file (single-rank: plain global writes).
- *Particles out:* per-rank live count → `MPI_Exscan` → each rank
  writes its slice of one concatenated global dataset (parallel
  hyperslab with per-rank offset; the only NEW HDF5 pattern needed —
  add `write_parallel_slice(data, count, offset, total, name)` or
  equivalent to the hdf wrapper if absent).
- *Particles in (the elegant part):* each reading rank loads an
  arbitrary contiguous ~1/world chunk of the particle dataset, then
  ROUTES particles to owners with the migration machinery: destination
  = O(1) arithmetic from the global cell (unit → path → angular rank ×
  slab base/rem → logical world rank, exactly `migrate_dest`'s math on
  global ids), Alltoallv over `comm->world()` (the LOGICAL-order
  communicator — REQUIRED under 7E node tiling), unpack translates
  global → local cells via `tri_g2l` + `k0` exactly like migration
  arrivals.  This is what makes different-rank-count restart nearly
  free.  Factor the shared pack/exchange/append code out of
  `prismatic_ptc_updater::migrate` rather than duplicating it.

**D4 — RNG policy, stated honestly.**  Same-rank-count restart
restores exact per-rank streams (bit-faithful injection).
Different-rank-count restart necessarily reseeds (streams are per-rank
objects): physically irrelevant, statistically fresh.  The reader
compares the stored writer (A, K, world) with its own and LOGS which
case applies.  `id` counters: restore per-rank on same count; on count
change, continue every rank above the stored global max (ids only feed
tracking; document).

**D5 — Crash-safe rotation.**  Write to `<ckpt_dir>/tmp/`, flush and
close, then ATOMIC RENAME to `<ckpt_dir>/ckpt_<step>/`; delete the
oldest generation only after the rename succeeds.  Config
`checkpoint_keep` (default 2).  A SIGKILL mid-write can never destroy
the last good checkpoint.  Sizing at the L6 / t14 scale: ~0.5 GB
fields + ~16 GB particles (48 B × ~325 M live) per generation —
seconds-to-a-minute on Lustre, ~35 GB disk for two generations.

*Sizing at the target scale.*  State grows ~8× per level (4× angular
× 2× radial).  Scaling the measured L6 t15 numbers (654 M live
macros, 7.2 GB fields): a generation is ~2 TB at L8, ~15 TB at L9,
and **~100–200 TB at L10** (particle-dominated: ~2.7×10^12 macros ×
~60 B on disk with uint64 cells; the stored field subset is only a
few TB).  This is inherent — checkpoint size is O(evolved state) and
cannot be designed away.  Consequences: `checkpoint_keep = 2` at L10
is ~0.3–0.4 PB of Lustre — a real quota line item (document in
LAUNCH_SCALING.md) — and a write is minutes-to-tens-of-minutes at
realistic aggregate bandwidth, so checkpoint cadence must be chosen
against wall-time, not taken for free.  I/O pattern: the
single-shared-file collective write is fine through ~L8–L9; at L10's
~10^4 writers a single HDF5 file is BORDERLINE (metadata/lock
contention), so keep the format schema but expect to switch the file
layout to HDF5 **subfiling** (or per-N-rank shards with an index
dataset) for the target runs — a change confined to the
writer/reader pair, not the global-indexed schema.  Do NOT implement
sharding in the first pass; just don't let any code outside the
writer/reader assume "one file per checkpoint".

## Implementation order

**Step 0 — resolve the one framework unknown FIRST: run-loop offset.**
Can `sim_environment::run()` start from step S (and time T)?  The
framework has snapshot plumbing (`include_in_snapshot` exists) that may
already carry this.  If not, it is a small framework patch — every
time-dependent prismatic piece takes `time`/`step` as arguments, so
only the loop counter and the per-system `m_time` accumulators matter.
Check `src/framework/environment.*` for: loop start step, whether
`update(dt, step)` steps are absolute, any existing
`load_snapshot`/`resume` hooks.  Everything else in this plan is under
prismatic control.

**Step 1 — writer**: new system `prismatic_checkpointer`
(`prismatic_checkpoint.h/.cpp`), constructor `(mesh, mp*, comm*)` like
the exporter, registered LAST in the mains (captures end-of-step state
after the solver).  Config: `checkpoint_interval` (steps, 0 = off),
`checkpoint_dir` (default `<output_dir>/ckpt`), `checkpoint_keep`.
Fetches by name: "Edelta", "Bdelta", "J", "rho", "rho_abs",
"gamma_wsum", "particles", "rng_states".  Owned-runs writes for
cochains (build the run sets once at init, same as the exporter — the
vertex/edge/face run-set builder can be shared instead of copied);
particle slice write per D3; metadata per D2.  Single-rank path: plain
writes (same file schema).

**Step 2 — reader + redistribution**: free function or small class
`prismatic_restart::load(env, ckpt_dir, ...)` called from the mains
when config `restart_from` is set, AFTER `env.init()` and INSTEAD of
`set_initial_dipole()` (order matters: init builds layouts; ICs would
be overwritten anyway but skipping them saves the quadrature cost).
  - Validate the config fingerprint loudly (L, N_r, radii params, dt).
  - Fields: read owned slots (mirror of owned-runs — each rank reads
    its runs with hyperslab selections; single-rank plain read).
    Load into Edelta/Bdelta/J/rho/rho_abs/gamma_wsum; call the
    solver's total refresh (or rely on its existing start-of-step
    refresh — verify which; the updater's `sync_fields` handles ghost
    exchange on first use).
  - Particles: chunked read → route via the factored migration pass →
    local append.  Buffer-overflow check against `max_ptc_num` with a
    clear error (a different decomposition concentrates particles
    differently — the inner-region warning from LAUNCH_SCALING.md
    applies).
  - rng: same-count → restore per-rank; else reseed (log).
  - Seed step/time into the env loop (step 0's answer) and into the
    exporter/sph/checkpointer `m_time` accumulators.
Mains to wire: `ns_rotator.cpp` (primary), `vacuum_dipole.cpp`
(trivial, fields only), `test_prismatic_pic_multirank.cpp` (for the
tests below).

**Step 3 — rotation + atomicity** (D5): `std::filesystem::rename`,
generation scan on startup (`restart_from = auto` picks the newest
valid generation in `checkpoint_dir` — presence of a completed marker
dataset, e.g. `complete = 1` written last).

**Step 4 — tests** (the important part):
  - *Continuity (host policy, in-tree CI):* extend
    `test_prismatic_pic_multirank` (or a sibling) — run 2N steps
    straight vs N + checkpoint + fresh process + restart + N, same
    rank count → final step files BITWISE identical on host builds
    (host arithmetic is deterministic; this pins that no state was
    missed).  On GPU: LIVE/MISOWNED exact, fields at the usual
    FP-reorder tolerance (~1e-4) — GPU deposit atomics make even
    uninterrupted runs vary at ~1e-8, bitwise is unattainable there
    by construction.
  - *Elasticity:* checkpoint at 8 ranks (A=4×K=2), restart at 40
    (20×2) and at 1 → LIVE conserved exactly, ESUM to ~8 digits,
    continued evolution within cross-rank tolerance of an
    uninterrupted run.  Also restart WITH node tiling on
    (`ranks_per_node`) to pin the logical-comm routing.
  - *Crash-safety:* leave a truncated `tmp/` behind, verify startup
    picks the previous complete generation.
  - Reference commands land in the plan-doc appendix / this file when
    implemented.

**Step 5 — docs**: checkpoint section in `LAUNCH_SCALING.md` (sizing,
rotation, RNG caveat, different-count recipe, `restart_from = auto`);
update the on-machine checklist (a checkpoint/restart cycle at the
node shape becomes item 6).

Estimated effort: ~2 days.  Step 0 first — it is the only piece not
fully under prismatic control.

## Gotchas for the implementing session (hard-won context)

- Migration-style collectives MUST use `comm->world()` (LOGICAL-order
  communicator), not `MPI_COMM_WORLD` — they address logical ranks and
  the 7E node tiling permutes physical ones.
- The updater builds its local mesh at REGISTRATION time (injector
  init runs first and needs it); a checkpointer registered last is
  outside that minefield, but any restore that touches particle data
  must run after `env.init()`.
- Problem binaries need explicit `--target` rebuilds after library
  edits (stale-binary gotcha bit twice already).
- Fields registered by name are shared solver/updater/exporter — the
  checkpointer must `get_data`, never re-register with different
  sizing.
- "J" is the dual-2 cochain (edge_kind tag) — store raw + the flag,
  as the exporter snapshots already do.
- Single-rank runs use identity layouts, global-sized fields — the
  same schema falls out naturally; keep one code path where possible.

## Appendix — scale-ceiling items ranked against the L10 target

The checkpoint FORMAT (global-indexed, layout-agnostic) survives to
any L untouched; these are the surrounding pieces that don't.  Ranked
by the level at which they break.  Items 1 and 2 sit BELOW or AT the
L10 target and are prerequisites; item 3 sits above it and stays
deferred — this demotion is the main practical payoff of choosing
L10 over L12.

**1. `uint32_t` global cell overflow (~L9) — hard correctness wall,
PREREQUISITE.**  Global cell = k·N_tri + tri exceeds 2^32 near L9
(5.2M tris × ~1.6k shells ≈ 8.6×10^9; L10 is ~6.9×10^10, see D2).
The checkpoint stores uint64 from day one, so the FORMAT is safe, but
the in-memory migration wire, `wire_cell()`, `migrate_dest()`, sort
keys and every `uint32_t cell` in the particle structs must widen (or
move to a (tri, k) pair encoding) before any run above ~L8.  Silent
wraparound, not a crash — audit, don't wait for symptoms.

**2. `distributed_cochain_layout::build` O((N_r+1)·4^L) scans —
PREREQUISITE.**  Init discovers ownership by scanning all
(N_r+1)·N_s global indices per cochain per rank, re-answering the same
ANGULAR ownership question N_r+1 times (ownership factorizes:
owns(g) = owns_shell(k) && owns_sphere(s)).  ~1–3 s/rank at L6,
~1–2 min at L8; at L10 the scan is ~10^11 indices per cochain per
rank — HOUR-scale init, unacceptable (and paid again on every
restart, which is the whole point of this plan).  Fix
(constructor-only, no format or consumer changes): angular ownership
bitmap once per cochain kind (O(4^L)), then ENUMERATE the owned set as
(owned shell range × owned sphere list), ghosts from the halo plans —
O(4^L + local), where the O(4^L) part is ~10^7 at L10 — trivial.
Do this alongside the checkpoint work.

**3. Replicated O(4^L) angular stage — memory wall at ~L11+,
DEFERRED.**  The sphere-stage tables replicated per rank (tri/edge/
vertex geometry, fan CSR, tri_verts, recovery weights) are tens of MB
at L8 and an estimated **~2–4 GB per rank at L10** — a noticeable but
survivable slice of a 64 GB GCD next to the particle buffer.  At L11
that becomes ~10–15 GB and at L12 tens of GB — impossible.  So the
"distributed subdivision / patch-local sphere stage" project is
needed only if the target moves above L10 again; it stays parked as
out-of-scope, with one obligation now: at init, LOG the replicated-
table total alongside the existing memory audit so the estimate is
replaced by a measurement the first time an L9/L10 config is built.
Nothing in the checkpoint schema changes if it ever lands — which is
exactly why the schema is global-indexed.

Rule of thumb going forward: per-rank **O((N_r+1)·4^L) TIME is
disqualifying** at the target scale; per-rank **O(4^L) MEMORY (~GB at
L10) is a budgeted line item** — acceptable case-by-case, never
free.  New code should be O(local) + O(A·K) unless explicitly
justified.  (For reference, the abandoned L12 numbers: 3.4×10^8
tris, ~4×10^12 prisms, replicated tables tens of GB/rank, generations
0.1–1 PB — every item above becomes mandatory including item 3.)
