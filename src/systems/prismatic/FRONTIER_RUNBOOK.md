# Frontier runbook — distributed prismatic pulsar PIC

Self-contained procedure for running the inclined-rotator magnetosphere
(`ns_rotator`) at scale on OLCF Frontier. Written to be executed with no
access to the local dev session — everything you need is here or in the
two companion docs it references (`LAUNCH_SCALING.md`, the
`CHECKPOINT_RESTART_PLAN.md` appendix).

Branch: `prismatic_mesh`. Reference commit for this runbook:
`ba16c81ee` (L6 a60 overnight settings). Binary: `ns_rotator`
(`problems/prismatic_dipole/src/ns_rotator.cpp`).

---

## 0. What this run is

A rotating (optionally inclined) magnetic dipole with volumetric,
E·B-triggered pair injection and a GCA/Boris-hybrid particle push. Fully
distributed: fields, particles, and mesh geometry are all local under a
canonical `A × K` angular×radial decomposition; deposits fold through
halo `reduce()`; particles migrate by cell ownership each step.

**Why the resolution matters (the science case, validated locally):** at
L5 the under-resolved current sheet numerically reconnects and dissipates
~60% of the Poynting flux between the light cylinder and the wave zone,
so the measured spin-down luminosity is only ~43% of the force-free
value. At L6 the sheet is resolved well enough that the flux survives to
the wave zone and the luminosity lands within a few percent of
force-free. L7+ is what resolves the sheet **down to its stellar
footpoints** — the target physics. Every level is 16× the previous
(8× macros × 2× steps).

---

## 1. Build

```bash
module load PrgEnv-amd amd-mixed cray-hdf5-parallel cmake   # site names vary
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}

cmake -S . -B build -Duse_hip=ON -Don_frontier=ON \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target ns_rotator \
      test_prismatic_solver_multirank \
      test_prismatic_pic_multirank \
      test_prismatic_mpi_backend_multirank tests
```

`on_frontier=ON` sets `--offload-arch=gfx90a` and the HIP flags. The
particle kernels are templated on the mesh-ptrs type and have only ever
been compiled/validated on CUDA — **the HIP compile of
`prismatic_ptc_update_kernel.hpp` is the single most likely place to hit
a first-contact surprise.** Build the tests too; they are your
acceptance gate (§2).

HDF5 must be the **parallel** build (`cray-hdf5-parallel`); the
checkpoint and exporter paths use collective MPI-IO.

---

## 2. First-contact validation (DO THIS BEFORE ANY PRODUCTION RUN)

Run on 2–4 nodes, in order. These catch the things that cannot be tested
off-machine. Do not skip to production on a fresh machine.

```bash
# (a) solver bit-exactness — must print 0.000e+00 for every scenario,
#     both packed and staged halo modes.  Second arg = ranks_per_node
#     exercises the node-tiling permutation.
srun -N1 -n8  --gpus-per-task=1 ./bin/test_prismatic_solver_multirank 4
srun -N1 -n8  --gpus-per-task=1 ./bin/test_prismatic_solver_multirank 4 8

# (b) distributed PIC acceptance — LIVE/MISOWNED must match the config
#     header baseline; step files agree across rank counts to ~1e-4.
srun -N1 -n8  --gpus-per-task=1 ./bin/test_prismatic_pic_multirank \
     -c tests/config_prismatic_pic_multirank.toml
srun -N1 -n8  --gpus-per-task=1 ./bin/test_prismatic_pic_multirank \
     -c tests/config_prismatic_pic_stress.toml       # migration stress

# (c) checkpoint/restart cycle on the parallel filesystem (Lustre):
#     continuity, cross-rank-count restart, crash-safety.  Delta mode
#     on GPU builds (atomics reorder deposits; host builds are bitwise).
tests/check_prismatic_checkpoint.sh ./bin/test_prismatic_pic_multirank
```

Then **one manual SIGUSR1** against a short running `ns_rotator` to
confirm the graceful-stop checkpoint fires under the site's signal
delivery (§6).

Pass criteria: (a) exactly `0.000e+00`; (b) `ALL PASS`, `MISOWNED 0`;
(c) `ALL CHECKPOINT TESTS PASS`.

---

## 3. Choosing the decomposition

`world_size = A × K`: `A` angular ranks (must be `2^j` or `5·2^j`, with
`A | 20·4^m` for some patch level `m ≤ L`), `K` radial slabs. Frontier =
8 GCDs/node, so `world_size = 8 × nodes`.

**Rule 1 — angular-major (K small).** Particles concentrate radially, so
radial slabs load-imbalance first. Measured on the real L5 a60 load map:
pure-angular `K=1` shapes hold hot-rank imbalance at ~1.9–2.4×, while
`K=4` blows up to 3.3–5.2×. **Use K=1** unless a shell count forces
otherwise. `N_r/K ≥ 2` is a hard requirement (pic radial halos read a
depth-2 upper shell).

**Rule 2 — node tiling.** Set `ranks_per_node = 8` so consecutive
world-rank blocks form compact tiles of the `A × K` grid (intra-node
peers become halo neighbors). Pure logical permutation; outputs are
identical with or without it.

**Rule 3 — let A auto-pick if unsure.** `n_angular_ranks = 0` calls the
angular-major heuristic for `(world, L, N_r)`. An explicit A is fine when
it satisfies the divisibility rule.

**Measured sizing (from the L5 a60 checkpoint load map, scaled to the L7
census ≈ 8×10⁹ macros; per-macro cost has a ±40% MI250X uncertainty
until §7 calibrates it):**

| level | census (a60) | nodes | shape (A×K) | hot-rank M | imbalance | hot GCD mem | wall (6 P) | node-h |
|------:|-----------:|------:|:-----------:|-----------:|:---------:|:-----------:|-----------:|-------:|
| **L7** | ~8×10⁹ | **40** | **320×1** | 54M | 2.2× | ~7 GB | ~3 h | ~116 |
| L7 | ~8×10⁹ | 80 | 640×1 | 29M | 2.3× | ~5 GB | ~1.5 h | ~123 |
| L8 | ~6×10¹⁰ | ~300 | angular-major | — | ~2.4× | <20 GB | overnight | ~2,000 |
| L9 | ~5×10¹¹ | ~2000 | angular-major | — | ~2.4× | <20 GB | ~15 h | ~30,000 |

Memory is **never** the binding constraint at these shapes (hot GCD ≤ 20
GB of 64). Throughput and the ~2.4× imbalance are. The imbalance is a
straight multiplier on node-hours and is the biggest remaining
optimization lever (census-weighted unit ranges could roughly halve it —
not yet implemented).

**Recommended L7 start: 40 nodes, `n_angular_ranks = 320`,
`ranks_per_node = 8`, `K = 1`.**

---

## 4. Config

Under MPI the driver builds the **sphere-only** mesh (no global 3D
arrays; per-element geometry is computed locally) and the `sph_output`
system is skipped — post-process the exporter's mesh-native dumps with
`python/sph_from_dump.py` (§8).

Scale from the committed L6 config
(`problems/prismatic_dipole/config_ns_rotator_L6_a60.toml`) by inverting
**only the level-dependent knobs** — everything else (Bp, Omega,
injection weights/thresholds, absorber radius, GCA settings) is physics
and stays fixed:

| knob | rule | L6 → L7 |
|---|---|---|
| `subdivision_level` | target | 6 → **7** |
| `N_r` | ×2 per level (same log shells, same r-range) | 204 → **408** |
| `dt` | ÷2 per level (CFL; ω_max ∝ 2^L) | 0.0049087 → **0.00245437** |
| `max_steps` | ×2 per level per period | (steps/P) → matched |
| `fld_output_interval` | ×2 (same physical cadence) | 640 → **1280** |
| `damping_length` | ×2 (same physical entrance ~19.8) | 44 → **88** |
| `max_ptc_num` | ~8× census + headroom | set from §3 hot-rank × safety |
| `n_angular_ranks`, `ranks_per_node` | §3 | 320, 8 |

`dt` is **CFL-set, not gyration-set** — GCA already banks the ~25×
gyration relaxation, so `dt` sits at ~73% of the leapfrog stability
limit and cannot be meaningfully increased (the next constraint,
ω_p·dt ≈ 0.8 at the surface, is also near its ceiling). Do not raise it.

**`dt` is part of the checkpoint fingerprint.** Changing `dt`, `L`,
`N_r`, or the radii means a *fresh run* — a restart across a `dt` change
is refused (it would corrupt the leapfrog staggering). `ptc_absorb_radius`
and `max_ptc_num` are **not** fingerprinted, so they can be edited
between resubmissions (see §6, §7).

Checkpoint/output block to add:

```toml
checkpoint_interval   = 2560     # steps between generations; ~0.5 P here
checkpoint_keep       = 2        # generations retained (1 to save quota)
checkpoint_dir        = "..."    # a Lustre path with room for keep×gen
restart_from          = "auto"   # newest complete generation, else fresh
checkpoint_ptc_window = 33554432 # particle staging slots (see §5); default fine
# ranks_per_node = 8
# n_angular_ranks = 320
halo_device_direct    = true     # GPU-aware MPI; set false to isolate MPI bugs
step_timer_interval   = 500      # min/mean/max per-phase timing; grep "step timing"
```

---

## 5. Memory sizing (per GCD)

- **Particle buffer**: `max_ptc_num × 48 B`, host+device. Size it so the
  **hot rank** (mean × imbalance from §3) sits at ~70% occupancy. The
  injector soft-throttles at `inj_buffer_frac × max_ptc_num` (default
  0.9). GCD has 64 GB; buffer + local fields/mesh/scratch must fit.
- **Replicated angular tables** (O(4^L), same on every rank): tens of MB
  through L8; estimated **~2–4 GB/rank at L10** (DEFERRED item — init
  logs the true value, watch the "replicated angular tables" line the
  first time an L9/L10 config builds). Not an issue at L7/L8.
- **Host mirror**: the particle buffer's host copy can exceed node RAM at
  large `max_ptc_num` and lives in swap untouched during stepping.
  Checkpoint particle I/O is **windowed** (`checkpoint_ptc_window`, ~2 GB
  staging) so it never faults the whole mirror in — this fix is why the
  writer no longer OOMs (see §9 pitfall). Leave the default unless a node
  is RAM-starved, in which case lower it.

Init prints a per-rank footprint line ("local 3D … replicated angular
tables … particle buffer …") — check it on the first production launch.

---

## 6. Launch & the resubmission chain

The killer feature: checkpoints are **rank-agnostic and global-indexed**,
so `restart_from = "auto"` lets the *same batch script* resume from the
newest complete generation at **any node count** the queue gives you. The
first submission (no generation yet) starts fresh; every resubmission
continues.

`ns_rotator` calls `MPI_Finalize()` itself at the end (the env singleton
never destructs), so a clean finish is normal — don't be alarmed by the
absence of a framework teardown message.

Example `submit.sbatch`:

```bash
#!/bin/bash
#SBATCH -A <PROJECT>
#SBATCH -J ns_a60_L7
#SBATCH -t 06:00:00
#SBATCH -N 40
#SBATCH --signal=B:USR1@120     # SIGUSR1 120 s before wall-time kill

CFG=problems/prismatic_dipole/config_ns_rotator_L7_a60.toml

# --signal above delivers SIGUSR1 to the batch shell; forward it to the
# job so the graceful-stop checkpoint fires before the wall-clock kill.
trap 'scancel --signal=USR1 --batch $SLURM_JOB_ID' USR1

srun -N40 -n320 --gpus-per-task=1 --gpu-bind=closest \
     ./bin/ns_rotator -c $CFG &
wait
```

**Chained resubmission** (each job resumes the last; requeue on the
timeout so a full run spans many wall-time windows automatically):

```bash
sbatch --dependency=singleton --job-name=ns_a60_L7 submit.sbatch
# or a self-resubmitting tail: at the end of submit.sbatch, if the run
# has not reached max_steps, `sbatch --dependency=afterany:$SLURM_JOB_ID
# submit.sbatch`.
```

**SIGUSR1 = graceful checkpoint-and-exit.** The `--signal=B:USR1@120` +
`trap` combination writes a final checkpoint at the resume point before
Slurm kills the job, so no steps between the last periodic checkpoint and
the wall-time limit are lost. Validate this once in first-contact (§2) —
signal forwarding is site-specific.

---

## 7. Cadence, cost, and the imbalance lever

- Pick `checkpoint_interval` against **wall-time, not for free**: a
  generation is written by collective MPI-IO in minutes at L8+ and is
  ~O(evolved state) on disk. One checkpoint per ~30 wall-minutes is a
  sane default; too frequent wastes I/O bandwidth.
- Run a **short timed segment first** (`step_timer_interval`, grep
  `step timing`) at the production shape to replace the ±40% MI250X
  cost estimate with a real number before committing the full campaign.
- If the census trends toward the throttle (`inj_buffer_frac`), the lever
  is **pull `ptc_absorb_radius` inward** (e.g. 13 → 11): it harvests the
  transported outer population and also cuts wall-time. It is *not*
  fingerprinted, so: let the checkpoint land, edit the config, resubmit
  — the run resumes with the new absorber and no lost work. Watch the
  per-period census in the checkpoint log lines to decide.

---

## 8. Output & diagnostics

Distributed runs write **only** the exporter's mesh-native dumps
(`step_XXXXXX.h5`: `E_e`, `B_f`, `J_e`, `rho`, `rho_abs`, `gamma_wsum` +
self-describing metadata) as single global-indexed files (collective
parallel HDF5), plus `mesh.h5` (sphere-level only under MPI). The in-code
spherical output is single-rank only.

Post-process on a login/analysis node:

```bash
python/sph_from_dump.py <run_dir> <step> ...     # -> sph_XXXXXX.h5
problems/prismatic_dipole/rotator_diagnostics.py  <run_dir> <step>   # scorecard
problems/prismatic_dipole/deutsch_luminosity.py   <run_dir> --omega 0.25 \
        --bp 1000 --alpha-deg 60                  # spin-down vs force-free
problems/prismatic_dipole/make_rotating_meridional_movie.py <run_dir> \
        --alpha-deg 60                            # co-rotating-frame movie
```

The **luminosity** is the headline convergence metric: wave-zone
Poynting flux / analytic vacuum-Deutsch value should approach the
force-free ratio `(3/2)(1+sin²α)/sin²α` (= 3.5 at α=60°) as the sheet
resolves. The **co-rotating movie** (meridional plane following
`φ_m = Ω t`) is the honest oblique diagnostic — φ-averaged panels smear
the tilted structure. Both default to the production normalization
(Ω=0.25, Bp=1000); override via flags/env for other runs.

---

## 9. Pitfalls & mitigations

| # | Pitfall | Symptom | Mitigation |
|---|---|---|---|
| 1 | **HIP compile of templated particle kernels** untested on AMD | build error or wrong PIC results only on GCD | Build + run the single-rank PIC acceptance on ONE GCD first (§2b). CUDA-validated only. |
| 2 | **GPU-aware MPI** not engaged / mis-handles device pointers | hang or garbage in halos | Init logs "packed, GPU-direct MPI". If it misbehaves set `halo_device_direct = false` (host-staged, wire-compatible — can even mix per-rank while debugging). |
| 3 | **Parallel HDF5 / Lustre** not striped | checkpoint & dump writes crawl or fail | `lfs setstripe -c <N>` on the output dir before the run. Use `cray-hdf5-parallel`. Confirm bit-identical vacuum output 1-vs-node-shape in §2. |
| 4 | **Wall-time SIGKILL** mid-run | job vanishes, log ends mid-line | This is EXPECTED on a scheduler. Checkpoints + `restart_from=auto` resume automatically; the `--signal=B:USR1@120` trap adds a final graceful checkpoint. Never run without checkpointing. |
| 5 | **SIGKILL mid-checkpoint-write** | partial `tmp/` generation | Harmless by design: generations are written to `<dir>/tmp/` then atomically renamed; `auto` skips any generation missing its `complete` marker. The last good generation is never touched. |
| 6 | **Host OOM during checkpoint** (large `max_ptc_num`) | node killed one minute into a checkpoint | Already fixed via windowed particle I/O (`checkpoint_ptc_window`); if a node is RAM-starved, lower the window. Do NOT revert to whole-buffer `copy_to_host`. |
| 7 | **Particle-buffer overflow on a DIFFERENT-count restart** | "particle buffer overflow" abort | A different decomposition concentrates particles differently. Size `max_ptc_num` for the *hot* rank of the restart shape, not the mean. |
| 8 | **Radial over-decomposition (`K` too large)** | one slab's ranks throttle while others idle; imbalance 3–5× | Keep K=1 / angular-major (§3). `N_r/K ≥ 2` is enforced (throws otherwise). |
| 9 | **`dt`/`L`/`N_r` change on a "restart"** | fingerprint-mismatch abort | Intended — those define a fresh run. Only `ptc_absorb_radius`, `max_ptc_num`, cadence, and rank shape may change between resubmissions. |
| 10 | **Injector silent throttle** | census plateaus flat at `0.9×max_ptc_num`; physics starved | Watch the per-period census in checkpoint log lines. If it pins at the throttle, raise `max_ptc_num` or pull `ptc_absorb_radius` in (§7). A hidden throttle produced physically-starved outer zones in early L5 tuning. |
| 11 | **int32 global-index overflow** | (historical) silent wraparound above ~L8 | CLOSED: global cell wire is uint64, global cochain/vertex indices are `int64` (`gidx_t`). Safe through L10+. New code must still write `gidx_t(k)*width+s`, never an int product. |
| 12 | **Replicated sphere-stage memory at L10** | per-rank angular tables ~2–4 GB | DEFERRED, survivable at L10. Init logs the real number — check it before an L10 campaign; a distributed/patch-local sphere stage is the fix if it ever bites. |
| 13 | **Single-shared-file HDF5 at ~10⁴ writers (L10)** | metadata/lock contention on the checkpoint file | BORDERLINE at L10 only. Schema is unaffected; switch the writer/reader pair to subfiling or per-N-rank shards. Nothing outside the writer/reader may assume "one file per generation". Fine through L8–L9. |

---

## 10. Quick reference — full launch sequence

```bash
# 1. build (§1)
cmake -S . -B build -Duse_hip=ON -Don_frontier=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target ns_rotator test_prismatic_solver_multirank \
      test_prismatic_pic_multirank tests

# 2. validate on 1–4 nodes (§2) — gate before production
tests/check_prismatic_checkpoint.sh ./bin/test_prismatic_pic_multirank

# 3. make config from L6 by the scaling table (§4); set decomposition (§3),
#    checkpoint block, max_ptc_num for the hot rank (§5)

# 4. lfs setstripe on the output + checkpoint dirs (§9.3)

# 5. short timed segment to calibrate cost (§7), then chained submit (§6)
sbatch --dependency=singleton --job-name=ns_a60_L7 submit.sbatch

# 6. monitor: per-period census + "step timing" in the run log;
#    post-process with sph_from_dump.py + the three diagnostics (§8)
```

Convergence expectation: L7's wave-zone luminosity should sit at or above
L6's ~97% of force-free, with the current sheet resolved progressively
further down toward the stellar surface — the reason for going to
Frontier at all.
