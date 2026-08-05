# Frontier bring-up log — 2026-07-20

What actually happened when the distributed prismatic PIC was first run on
Frontier. Written as a handoff for a fresh debugging session.

**Read this before `FRONTIER_RUNBOOK.md`.** That document was written
without Frontier access (commit `171eb62e`, AI co-authored, same day) and
its machine-facing claims are unreliable — several are provably wrong. Its
*code*-facing claims held up under every check. §8 below lists the specific
corrections.

---

## 0. Current state — one sentence

The build works, all correctness gates pass, single-rank L6 runs fine, and
**the distributed (multi-rank) path faults on the first timestep that has
particles at L6**. That bug blocks L7 and blocks any honest scaling number.

---

## 1. The bug (start here)

> **RESOLVED 2026-07-20, later session — root cause: Cray MPICH GPU IPC,
> not our code.** Workaround: `export MPICH_GPU_IPC_ENABLED=0`.
>
> Evidence chain:
> 1. **Core dump** (job 5038053, task 1, 8 GB `core` in the run dir):
>    main thread inside `prismatic_ptc_updater::update` →
>    `reduce_edge(J)` → `run_packed_reduce` → `MPI_Waitall` when the HSA
>    event thread delivered the fault and aborted.  Under
>    `AMD_SERIALIZE_KERNEL=3` every app kernel is synchronized at launch,
>    so the push/deposit kernels had COMPLETED — the faulting GPU access
>    came from the MPI data movement, not an app kernel.
> 2. **Fault address is a peer-GPU IPC window**: `gdb info proc
>    mappings` on the core places `0x7ffbec80f000` inside a 2 MB region
>    backed by `/dev/dri/renderD130` — a *different GCD's* render node.
>    Rank 1's own GCD is `renderD133` (123 mappings, 6.3 GB: particle
>    buffer + fields).  The five other render nodes appear only as
>    ~2 MB × ~11 imported windows — the Cray GTL's intra-node GPU IPC
>    machinery.  The app's own buffers are innocent.
> 3. **Discriminators** (job 5038347, both 8-rank, 1e8, 100 steps):
>    Phase A `halo_device_direct = false` → clean.  Phase B
>    device-direct + `MPICH_GPU_IPC_ENABLED=0` → clean.  Three prior
>    runs with IPC on faulted at step 10, deterministically.
> 4. **Why step 10, why rank 1**: steps 0–9 have zero particles — all
>    ranks hit the halo calls in lockstep.  Step 10 is the first
>    injection, so per-rank push cost (and MPI arrival time) first
>    *desynchronizes* there; the IPC-path defect is timing-sensitive.
>    (Consistent with it: the acceptance test at L=2 passes — tiny
>    messages, near-lockstep.)
>
> Validation at production timing (no `AMD_SERIALIZE_KERNEL`, 30-min
> window, sorts + snapshot + throttle): job 5038362 / `diagnose6.sbatch`
> — check its `.out` before trusting this paragraph.
>
> Follow-ups: (a) put `MPICH_GPU_IPC_ENABLED=0` in the launch env
> (NOT the build env) for every distributed run until OLCF confirms a
> fix; (b) file an OLCF ticket with `core`, the two fault logs, and
> the 5038347 discriminator — cray-mpich/9.1.0 + rocm/7.0.2; (c)
> re-measure the intra-node halo cost with IPC off before sizing L7 —
> peer messages now route through the host/NIC path.

### Symptom

```
Memory access fault by GPU node-9 (Agent handle: 0x...) on address 0x7fec4e8c1000.
Reason: Unknown.
srun: error: frontier02934: task 1: Aborted (core dumped)
```

Binary `ns_rotator`, config `config_ns_rotator_L6_a60_frontier.toml`,
1 node = 8 ranks, A=8 angular x K=1 radial.

### What is established

| Fact | Evidence |
|---|---|
| Deterministic, always **rank 1** (task 1) | 3 independent runs, near-identical fault addresses |
| **Not** buffer size | identical fault at `max_ptc_num` 8e8 and 1e8 |
| **Not** OOM | `free=28.649/68.703` at 8e8; `free=62.246/68.703` at 1e8 |
| **Not** an async race | unchanged under `AMD_SERIALIZE_KERNEL=3` |
| **Not** the L6/HIP particle kernels as such | single-rank L6 runs 780+ steps clean |
| **Is** in the distributed path | the only variable changed between pass and fail was rank count (1 vs 8) |
| Scale/layout dependent | the L=2 distributed PIC acceptance test passes (`LIVE 630`, `MISOWNED 0`) |

### Where it dies

Step 0 completes fully with `ptc number: 0` — injector (no-op, 0.71 ms),
updater, solver, and the first parallel snapshot all succeed. The fault
comes on the first timestep with real particles:

```
=== Time step 10, Time is 0.04909 ===
>>> Time for prismatic_surface_injector is 6.71ms     <-- completes
<fault>
```

`inj_interval = 10`, so step 10 is the first injection. The injector
*finishes* (it printed its timing), so suspect what runs next: the particle
updater / push / deposit on freshly injected particles, under a decomposed
mesh.

### First place to look

Phase 7C moved particle cells to LOCAL indices
(`prismatic_ptc_mesh_local.h`, `prismatic_ptc_update_kernel.hpp`). At L6
with A=8 the local/global spans differ by ~7x:

```
Distributed particle updater: rank 0 owns 10240 tris (of 10881 local), layers [0, 204) at k0 = 0
Distributed DEC solver: rank (0, 0) of 8x1, local edges 4521618 / 33546648 global,
                                            local faces 5592933 / 41861120
```

At L=2 (the tested case) local and global are close enough that a
local/global confusion may not fault. Audit the local<->global cell
mapping, the halo/ghost tri range (`10240` owned vs `10881` local — the
641-tri ghost band), and `migrate_dest` / `wire_cell` in the injected-particle
path.

### Reproduce

~20 s on 1 node in the debug QOS:

```bash
cd /lustre/orion/ast229/proj-shared/alex/runs/ns_rotator_L6_a60
sbatch diagnose.sbatch      # 8 ranks, max_ptc_num 8e8 -> faults
sbatch diagnose2.sbatch     # 8 ranks, max_ptc_num 1e8 -> faults identically
sbatch diagnose3.sbatch     # 1 rank                   -> runs clean
```

`diagnose*.sbatch` set `stdbuf -o0 -e0` — **required**. The abort kills the
process before stdout flushes; the first failing run produced a 28-byte
`.out` and lost the footprint line and all step timings.

---

## 2. Build recipe that works

```bash
source machines/frontier.20260408      # cpe/26.03, rocm/7.0.2, cray-mpich/9.1.0
mkdir build_frontier && cd build_frontier
CXX=CC CC=cc HIPCC=hipcc cmake .. -DCMAKE_BUILD_TYPE=Release \
  -Duse_double=0 -Duse_hip=1 -Don_frontier=1 -Dbuild_python=0 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j 8
```

`-Duse_double=0` was a deliberate choice for this campaign (single
precision: 48 B/particle instead of 80 B). `-Dbuild_python=0` is required
on Frontier (no pybind11; system python is 3.6.15).

Two blockers had to be fixed before this worked at all — see §3.

Note: builds under `build/` (Mar 2026, ROCm 6.4.2) and `builds/Frontier`
(Dec 2023, ROCm 5.4) are dead. OLCF removed ROCm 5.x and CPE <= 23.12 on
2026-07-01, which also makes `machines/frontier.new` (cpe/23.05, amd/5.5.1)
obsolete.

---

## 3. Fixes committed

| commit | what |
|---|---|
| `5815c6df` | `build_python` option — `add_subdirectory(python)` was unconditional, so `find_package(pybind11 REQUIRED)` aborted configure |
| `9fad1f1a` | thrust/CCCL guard — ROCm 7 ships a CCCL-based Thrust but no CCCL headers, so host TUs reaching `<thrust/...>` via `GPU_ENABLED` failed with `'cuda/__cccl_config' file not found`. Invisible on CUDA. |
| `65abe082` | `check_prismatic_checkpoint.sh` — srun launcher, absolute binary path, preserve scratch on failure (3 bugs, see §4) |
| `e56d3bbd` | L7 a60 config (untested at scale — blocked by §1) |

**Uncommitted:** `problems/prismatic_dipole/config_ns_rotator_L6_a60_frontier.toml`
(the config that reproduces the bug — worth committing).

---

## 4. Bugs found in `check_prismatic_checkpoint.sh`

All three would bite anyone, not just on Frontier:

1. Hardcoded `mpirun --oversubscribe` — **mpirun does not exist on
   Frontier** (Cray MPICH launches via srun). Now resolves
   `$PRISM_MPI_LAUNCH` -> srun under Slurm -> mpirun otherwise.
2. `bin` used verbatim after `cd "$work"`, so the documented invocation
   (`... ./bin/test_prismatic_pic_multirank`, a relative path) could never
   have worked. Now resolved to absolute up front.
3. `trap 'rm -rf "$work"' EXIT` deleted the per-run logs on failure —
   destroying the evidence exactly when needed. Now cleans up only on
   success.

Under Slurm, `--gpus-per-task=1` means the 20-rank elasticity case needs
>= 20 GCDs, i.e. **>= 3 nodes**. `TMPDIR` must point at Lustre; `/tmp` is
RAM-backed tmpfs on compute nodes.

---

## 5. What passes (correctness gates, all on Frontier hardware)

| gate | job | result |
|---|---|---|
| host test suite | — | 248 cases, 9,241,445 assertions, all pass |
| solver bit-exactness | 5037247 | `0.000e+00` on all 8 scenarios, packed + staged, plain + node-tiled |
| distributed PIC acceptance | 5037247 | `LIVE 630 / MISOWNED 0 / ALL PASS`, matching the committed 2026-07-18 CUDA baseline **exactly** |
| checkpoint/restart | 5037448 | `ALL CHECKPOINT TESTS PASS` — continuity (1, 8 ranks), elasticity (8->1, 8->8 tiled, 8->20), crash safety |

The PIC acceptance reproducing the CUDA baseline exactly retires runbook
pitfall #1 (HIP particle kernels untested on AMD) **at L=2**. It clearly
does not cover L6 — see §1.

Checkpoint elasticity passing matters: `restart_from = "auto"` across
changing rank counts is the mechanism the whole chained-resubmission plan
depends on.

---

## 6. Measured performance (first real numbers)

Single GCD, L6 a60, **throttled** at ~73M particles (`max_ptc_num = 1e8`
was a diagnostic value, not a physics value):

```
step 1.50 s = particle updater 1.40 s (93%) + injector 45 ms + field solver 13 ms
per-particle-step: 19.2 ns
```

Per-rank footprint at L6, A=8 (measured, from the init line):

```
local 3D (fields+maps+Bv) ~ 131 MB
replicated angular tables ~ 9.7 MB
particle buffer            38.4 GB at max_ptc_num = 8e8
```

**This is a particle-throughput problem, not a field-solver problem** — the
DEC solver is ~1% of step time.

Extrapolating 19.2 ns/particle-step with the runbook's own census (8e9) and
imbalance (2.2x), for L7 5 P = 51200 steps:

| nodes | hot rank | s/step | wall |
|---|---|---|---|
| 40 | 55M | 1.05 | **15.0 h** |
| 80 | 28M | 0.53 | 7.5 h |
| 160 | 14M | 0.26 | 3.8 h |

The runbook claims **~3 h at 40 nodes** — off by ~5x in the expensive
direction. Caveats: derived from a throttled single-GCD run, assumes the
updater is linear in particle count, and inherits the unvalidated 8e9
census and 2.2x imbalance. Treat as an order-of-magnitude correction, not a
number to plan against. Redo it once the distributed path runs.

**UPDATE — unthrottled 8-rank numbers (job 5038835, 2026-07-20).** The
first healthy distributed production window (8 GCDs, `max_ptc_num = 8e8`
per rank, GPU-IPC workaround active) reached step ~16,290 (~3.2 P) in
the 2 h wall.  Never throttled: rank 0 plateaued near ~115M particles
(14% of buffer).  Plateau block [9501..10000], per step across ranks:

```
sync    ~10 ms   |  push  min 294 / mean 361 / max 424 ms
reduce  min 10 / mean 72 / max 140 ms   (mean/max is skew-wait, not wire)
migrate ~3.9 ms  |  sort ~2.6 ms  |  total step ~490 ms -> ~520 ms by 16k
```

Hot/cold push spread 1.45x (hot/mean 1.17x — milder than the runbook's
2.2x).  Checkpoint generation = 48 GB, writes in ~6 s.  These numbers
supersede the throttled single-GCD extrapolation below for L7 sizing.

**Throttle warning.** The single-rank run printed
`Surface injector throttled at 72348209 particles (buffer frac 0.9)`, and
the updater cost plateaued (531 -> 820 -> 1145 -> 1470 -> ~1500 ms flat).
That plateau is the signature of runbook pitfall #10: a throttled run is
particle-starved and **not physically comparable** to the workstation
result. Size `max_ptc_num` for physics before any comparison run.

---

## 7. Sizing notes (verified in code)

- Particle = **48 B** exactly at `use_double=0` (8 floats + `uint32 cell` +
  `uint64 id` + `uint32 flag`). Confirmed empirically: buffer 96.0 MB at
  `max_ptc_num = 2e6`.
- `max_ptc_num` is **per rank** in the distributed path, not the global
  census. The workstation L6 value (1.4e9) is a single-GPU whole-census
  number and must not be copied across.
- Buffer cost is mostly free: sort scratch is a **fixed** 10M-element
  segment (`particles.h:128`), not capacity-scaled; the checkpoint writes
  `n_live`, not capacity; the host mirror is plain `new T[]` (pageable,
  lazily backed — `buffer.hpp:69`), not pinned.
- Hard ceiling: `max_ptc_num` is read into an **`int`**
  (`prismatic_ptc_updater_impl.hpp:51`), so it must stay < 2.147e9.
- MI250X GCD reports `68.703` total by the code's own units; the label says
  GiB but the value is consistent with 64 GiB reported as bytes/1e9 — a
  cosmetic bug in the print, worth fixing to avoid confusion.
- `checkpoint_ptc_window` defaulted to **0** in the log, while the runbook
  documents 33554432. Worth confirming 0 does not mean "unwindowed", since
  windowing is what fixed the host-OOM-during-checkpoint issue.

---

## 8. Runbook corrections

| runbook says | reality |
|---|---|
| §2c `mpirun --oversubscribe` | mpirun does not exist on Frontier |
| §6 `-N 40 -t 06:00:00` | **rejected by the scheduler.** Verified with `sbatch --test-only`: 1-91 nodes cap at 2 h, 92-183 at 6 h |
| §1 build recipe | incomplete — omits both blockers (§3); does not build as written |
| §2c default `/tmp` | RAM-backed tmpfs on compute nodes; would not test Lustre at all |
| §3 "~3 h at 40 nodes" | measured extrapolation says ~15 h (§6) |
| ROADMAP refs `python/prismatic_memory_budget.py` | file does not exist |
| §5 "GCD has 64 GB" | ~correct, but the code's own print mislabels units (§7) |

Pattern: claims derivable from reading the repo are reliable; claims
requiring a Frontier shell are not. Verify anything in the second category
before acting on it.

---

## 9. Artifacts

Everything preserved on Lustre:

```
/lustre/orion/ast229/proj-shared/alex/runs/
  ns_rotator_L6_a60/           <- the bug lives here
    ns_a60_L6_frontier-5037969.{out,err}   8 ranks, 8e8  -> FAULT (28-byte out: stdout lost)
    ns_a60_L6_diag-5038014.{out,err}       8 ranks, 8e8  -> FAULT (full log)
    ns_a60_L6_diag2-5038053.{out,err}      8 ranks, 1e8  -> FAULT (reached step 10)
    ns_a60_L6_diag3-5038088.{out,err}      1 rank,  1e8  -> CLEAN to step 780 (cancelled)
    ns_a60_L6_diag45-5038347.{out,err}     8 ranks, 1e8, 100 steps
                                           phase A host-staged halos -> CLEAN
                                           phase B GPU IPC off       -> CLEAN
    ns_a60_L6_diag6-5038362.{out,err}      8 ranks, 1e8, IPC off, no serialize,
                                           30-min window (workaround validation)
    core                                   task 1 of 5038053; gdb backtrace =
                                           MPI_Waitall in run_packed_reduce(J)
    diag3_output/                          sph_000000.h5, sph_000640.h5, step_*.h5
    config_ns_rotator_L6_a60_frontier.toml, config_diag{,2,3}.toml
    diagnose{,2,3}.sbatch, submit.sbatch
  ns_rotator_L7_a60/           <- staged, never launched (blocked by §1)
    submit.sbatch, validate.sbatch, validate_checkpoint.sbatch
    config_ns_rotator_L7_a60.toml, bin/, output/, checkpoints/
```

`diag3_output/sph_*.h5` are single-rank spherical dumps, directly
comparable in format to the workstation L6 run — but throttled, so treat
any comparison as partial.

Note: distributed runs write mesh-native `step_XXXXXX.h5` only (the in-code
spherical output is single-rank); use `python/sph_from_dump.py` to compare
distributed output against workstation `sph_` dumps.

---

## 10. Suggested order

1. ~~**Fix the distributed particle path** (§1).~~ RESOLVED — Cray MPICH
   GPU IPC bug; run with `MPICH_GPU_IPC_ENABLED=0` (see §1 banner).
   Confirm the 5038362 validation run, then file the OLCF ticket.
2. Rerun L6 distributed with `max_ptc_num` sized for physics (~8e8 per rank
   at 8 ranks) — unthrottled census, real imbalance, and a genuine
   comparison against the workstation result.
3. Re-derive the L7 node count from *that* measurement, not from §6's
   throttled extrapolation and not from the runbook.
4. ~~Validate SIGUSR1 end-to-end (the handler installs; graceful
   checkpoint-and-exit has never actually fired).~~
   **First live test FAILED (job 5038835, 2026-07-20):** the batch trap
   used `scancel --signal=USR1 --batch`, which delivers USR1 back to the
   batch shell only — the ranks never saw it and the job died on the
   time limit at step ~16,290 with no final checkpoint (resume point =
   the last interval generation, ckpt_15361; ~930 steps lost).  Fixed in
   the run-dir `submit.sbatch`: `trap 'kill -USR1 $SRUN_PID' USR1`
   (srun forwards to all tasks).
   **VALIDATED (job 5041746, 2026-07-21, L7 40 nodes / 320 ranks):**
   dedicated wall-limited resume of the L7 run (`submit_sigtest.sbatch`;
   interval checkpoints disabled so the signal path is the only writer).
   USR1 delivered at T-120 s -> "Graceful stop requested at step 61538"
   (not a checkpoint-interval multiple) -> generation ckpt_61538 written
   in 49.6 s (6.08e9 macros) -> `srun exited with status 0` at 18:30 of
   the 20:00 window.  Margin: ~50 s write inside the @120s offset — keep
   @120 for L7-scale runs (the earlier "~6 s" figure was the 8-rank L6
   census; per-generation write time scales with census, not ranks).
   The same job also validated ckpt-elasticity item: resume from
   ckpt_51201 restored 5.86e9 macros in 49.0 s at identical
   decomposition.
   Separately, the L7 production window itself (job 5041012) finished
   ALL 51200 steps (5 P) in 80 min — census plateaued at 5.86e9
   (~18M/rank mean), ~95 ms/step — so §6's L7 sizing table is obsolete
   in the cheap direction: 40 nodes is ample for 5 P inside one 2 h
   debug window.
