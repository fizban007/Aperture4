# Prismatic distributed runs — launch & scaling guide

The decomposition machinery is **cluster-agnostic**: no machine names,
placement files, or `#ifdef`s anywhere in the code.  Everything a
cluster needs is expressed through three config keys and the launcher's
tasks-per-node count.  This document is the one place specific machines
are mentioned (as worked examples at the end).

## 1. Choosing a decomposition

A run uses `world_size = A × K` MPI ranks: `A` angular ranks over
`U = 20·4^m` level-`m` patch units (`A = 2^j` or `5·2^j`, `m ≤ L`) and
`K` radial slabs.

Config keys (all drivers: `vacuum_dipole`, `ns_rotator`, the PIC
acceptance test):

| key               | meaning                                          | default |
|-------------------|--------------------------------------------------|---------|
| `n_angular_ranks` | A; **0 = auto** (angular-major suggestion)       | 20      |
| `ranks_per_node`  | node-tile size for rank placement (0 = off)      | 0       |
| `step_timer_interval` | per-phase timing report cadence (0 = off)    | 0       |

Auto mode (`n_angular_ranks = 0`) calls
`prismatic_partition::suggest_angular_ranks(world, L, N_r)`: the
LARGEST valid A — angular-major, because particles concentrate radially
(and in latitude for oblique rotators), so radial slabs load-imbalance
first (plan F10).  Constraints enforced either way:

- `A | world_size`, `A ∈ {2^j, 5·2^j}` with patch level `m ≤ L`;
- `K = world/A ≤ N_r`;
- **PIC runs need `N_r / K ≥ 2`** (the pic-depth radial halos read a
  depth-2 upper shell from the immediate neighbor).

Useful shapes at a glance (any machine):

| GCDs | shapes (A × K)                    | notes                       |
|------|-----------------------------------|-----------------------------|
| 8    | 4×2 (auto: 8×1 at L ≥ 1)          | single node                 |
| 40   | 20×2, 40×1                        |                             |
| 320  | 320×1 (L ≥ 2), 80×4, 20×16        |                             |
| 8000 | 320×25 (L ≥ 2), 80×100            | 5² never divides 20·4^m —   |
|      |                                   | the second 5 goes radial    |

## 2. Rank placement (node tiling)

Every launcher's default places consecutive world ranks on the same
node.  Set `ranks_per_node` to the launcher's tasks-per-node and the
comm internally permutes the world-rank → (angular, radial) assignment
so each consecutive block forms a compact `a_t × k_t` patch of the
A × K grid — intra-node peers become geometric halo neighbors.  The
tile shape is chosen automatically (squarest valid tile, angular-major
tie-break; e.g. 8 ranks on 80×100 → 4×2 tiles, on 320×25 → 8×1).  An
unsatisfiable tiling falls back to the identity assignment with a log
note.  This is a pure logical-rank permutation: global outputs are
IDENTICAL with or without it (validated bit-exact in
`test_prismatic_solver_multirank <A> <ranks_per_node>`).

Do NOT use launcher placement/rank files; keep block placement (the
default) and let the tiling do the geometry.

## 3. Memory estimate

At init every PIC rank logs

    Per-rank footprint: local 3D (fields+maps+Bv) ~ X, replicated
    angular tables ~ Y, particle buffer Z

- local 3D ≈ (fine cochain count / (A·K)) × ~30 B — scales as 1/(A·K)
  plus a halo-surface fraction;
- angular tables ≈ O(4^L) — REPLICATED per rank by design (~10s of MB
  at L8), flat in A·K;
- particle buffer = `max_ptc_num` × 48 B, host + device.

Measured weak scaling (L=3): 236.8 → 72.8 → 30.6 kB local 3D across
8 → 40 → 160 ranks against a flat 151.1 kB angular constant.

## 4. Timing harness

`step_timer_interval = N` reports min/mean/max wall time per step
across ranks for the particle phases (sync / push / reduce / migrate /
sort) every N steps — grep `step timing`.  The exchange-dominated
phases (`sync`, `reduce`) are the scaling observables; `push` is the
GPU compute floor.  The full measurement campaign remains deferred;
this is its instrument.

## 5. Launch examples

Generic Slurm (any GPU cluster, `G` GPUs per node):

    srun -N <nodes> --ntasks-per-node=G --gpus-per-task=1 \
         ./ns_rotator -c config.toml
    # config: n_angular_ranks = 0, ranks_per_node = G

Generic OpenMPI workstation / small cluster:

    mpirun -n <A*K> ./ns_rotator -c config.toml

GPU-aware MPI is detected at RUNTIME (`MPIX_Query_cuda_support`); no
build flag.  If a machine's MPI misbehaves with device pointers, set
`halo_device_direct = false` (wire-compatible per rank — it can be
flipped on a subset of ranks while debugging).

### Worked example: Frontier (8 GCDs/node, HIP)

    cmake -S . -B build -Duse_hip=ON -Don_frontier=ON
    srun -N 1000 --ntasks-per-node=8 --gpus-per-task=1 \
         ./ns_rotator -c config_L7.toml
    # config: n_angular_ranks = 320   (K = 25), ranks_per_node = 8
    #         -> 8x1 node tiles (25 is odd); or A = 80, K = 100 -> 4x2

### Worked example: generic CUDA cluster (4 GPUs/node)

    srun -N 10 --ntasks-per-node=4 --gpus-per-task=1 ...
    # config: n_angular_ranks = 40, ranks_per_node = 4 -> 2x... tiles

## 6. First-contact verification checklist (any new machine)

Things that cannot be validated off-machine; run in this order on a
few nodes before a production campaign:

1. `mpirun/srun -n <A*K> test_prismatic_solver_multirank <A>` — must
   print bit-exact 0.000e+00 for all scenarios, both packed and staged
   modes.  Repeat with `<A> <ranks_per_node>` for the tiling.
2. `test_prismatic_pic_multirank -c tests/config_prismatic_pic_multirank.toml`
   at 1 rank and at the node shape — LIVE/MISOWNED must match the
   config header's baseline; step files within 1e-4 across rank counts.
   Repeat with `tests/config_prismatic_pic_stress.toml` (migration).
3. GPU-aware MPI: confirm the init log says "packed, GPU-direct MPI";
   if the run misbehaves, retry with `halo_device_direct = false` to
   isolate the MPI library.
4. Parallel HDF5 on the parallel filesystem: exporter snapshots at the
   node shape vs 1 rank (bit-identical for vacuum runs).  Lustre
   striping is an environment concern (`lfs setstripe` on the output
   dir), not code.
5. HIP-specific: the templated particle kernels
   (`prismatic_ptc_update_kernel.hpp` on `prismatic_ptc_mesh_ptrs`)
   have been validated on CUDA; run the single-rank PIC acceptance on
   one GCD first to pin the HIP compile.
