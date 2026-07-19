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
| particles: `x1..3`, `p1..3`, `E`, `weight`, `cell` (GLOBAL encoding), `flag`, `id` | one concatenated global dataset, per-rank slices | live particles only (skip `empty_cell` slots); cells translated by the existing `wire_cell()` migration encoding |
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

## Appendix — deferred, related: init-cost optimization (watch item)

Not part of this plan; recorded so it isn't lost.  Init is O(global)
in TIME (not memory) because `distributed_cochain_layout::build`
discovers ownership by scanning all (N_r+1)·N_s global indices per
cochain per rank — re-answering the same ANGULAR ownership question
N_r+1 times (ownership factorizes: owns(g) = owns_shell(k) &&
owns_sphere(s)).  Per-rank cost ~1–3 s at L6, ~5–10 s at L7, ~1–2 min
at L8.  Fix when a real L7/L8 init measurement hurts: compute the
angular ownership bitmap once per cochain kind (O(4^L)), then
ENUMERATE the owned set as (owned shell range × owned sphere list) and
take ghosts from the halo plans — O(4^L + local), no format or
consumer changes (only the layout constructor).  Tier-1 replicated
angular work (subdivision, recovery weight fits ~10–30 s at L8) is the
F8-budgeted floor and stays.
