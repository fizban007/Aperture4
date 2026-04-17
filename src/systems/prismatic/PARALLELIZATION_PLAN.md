# Prismatic Mesh MPI Parallelization Plan

## Goal

Enable the prismatic GR field solver to run across multiple GPUs and nodes
with MPI. Target working resolution: L=8 icosphere subdivision × N_r ≈ 200
radial layers (~260 M prism cells, ~1.2 B DOF), for production
Wald/BH-plasma simulations.

## Partition Strategy

Two-axis decomposition:

1. **Angular axis** — partition by the 20 original icosahedron faces.
   Each ico-face carries exactly `4^L` sub-triangles after L subdivisions
   (key property of the subdivide() algorithm: subdivision keeps the
   4 child triangles of triangle `i` at indices `[4i, 4i+4)` in the new
   list, so after L subdivisions ico-face `k` owns sub-triangles
   `[k · 4^L, (k+1) · 4^L)` — contiguous range in the global index).
   Perfect load balance at this level.

2. **Radial axis** — partition N_r shells into K radial slabs.
   One-dimensional Cartesian decomposition with one-shell ghost layers
   on each interior slab boundary. Shell-contiguous in the global index.

Total ranks: `20 · K` (K radial × 20 angular). For L=8 N_r=200 with K=5,
that's 100 ranks, ~10 M DOF/rank — reasonable GPU memory footprint.

## MPI Topology

Two sub-communicators:

- **`comm_radial`**: K ranks sharing the same ico-face; built via
  `MPI_Cart_create(1D, periodic=false)`.
- **`comm_angular`**: 20 ranks sharing the same radial slab; built via
  `MPI_Dist_graph_create_adjacent`, with the ico-face dual graph.

### Angular topology (dodecahedron + vertex-diagonals)

Each ico-face has **9 topological neighbors** under this mesh's DEC stencil:

- 3 **edge-neighbors** (ico-edge sharing) — 3-regular dodecahedron graph.
  Carry the bulk of the halo data (`~N_r · 2^L` values per interface).
- 6 **vertex-diagonal neighbors** — at each of F's 3 corners (valence-5
  icosahedron vertices), F meets 4 other faces via the vertex; 2 of those
  are already edge-neighbors, leaving 2 diagonal per corner, 6 total.
  Carry very small halos (1-2 corner elements).

MPI_Dist_graph_create_adjacent primitive is the right choice — arbitrary
graph, neighbor collectives (`MPI_Neighbor_alltoallv`) give one-call halo
exchange per step.

Attach communication weights proportional to halo byte-count so MPI can
place edge-neighbors on the same node when reordering.

### Radial topology

Standard 1D Cartesian. Two nearest-slab neighbors per rank (one if at
radial boundary). Ghost = one shell on each side.

## Ownership Conventions

Rule: each mesh element is owned by **exactly one** rank; all other ranks
access via halo.

- **Sub-triangle (face 2-cochain)**: owned by the rank whose ico-face
  contains it. Index range `[k · 4^L, (k+1) · 4^L)` per ico-face k.
  No sharing — the sub-triangle's angular position is strictly inside
  one ico-face.

- **Sphere vertex (vertex 0-cochain)**: interior vertices owned by their
  ico-face. Vertices on ico-edges owned by the ico-face with smaller
  global index. The 12 valence-5 icosahedron vertices owned by the
  lowest-index ico-face incident to them.

- **Sphere edge (edge 1-cochain)**: interior edges owned by their
  ico-face. Edges along an ico-edge boundary (the `2^L` sphere-edges
  lying on one ico-edge) owned by the ico-face with smaller global index.

- **Radial slab boundary**: shell `k` sits on the slab boundary between
  slabs owning `[k_lo, k)` and `[k, k_hi)`. Ownership convention:
  shell `k` is owned by the lower-indexed slab; upper slab receives it
  as ghost.

- **Vertical edge (at a valence-5 vertex, spanning slab boundary)**:
  owned by the rank owning both the sphere-vertex and the lower shell.

## Halo Structure

### Angular halos (per-ico-edge)

For each of a rank's 3 edge-neighbors, the halo strip consists of:

- `N_r · 2^L` tri-face values (the `2^L` sub-triangles along the shared
  ico-edge on each of N_r shells, inner side of the boundary)
- `N_r · (2^L - 1)` horizontal sphere-edges on the interior side (not on
  the ico-edge itself, which is owned by one side)
- `N_r` rect faces bridging sphere-edges that cross the ico-edge
- `N_r + 1` sphere-vertex values for each shell-interface

### Angular halos (per-vertex-diagonal-neighbor)

Per valence-5 corner, per diagonal neighbor: ~O(1) values per shell.
Much smaller than edge halos but still required for the vertical-edge
dual construction at the valence-5 vertex (which spans 5 ico-faces).

### Radial halos

Entire ico-face triangulation at the slab-boundary shell:
- `4^L` tri-face values
- `1.5 · 4^L` horizontal edges (roughly) at that shell
- `4^L / 2 + 2` sphere-vertices at that shell

All contiguous in the global index within an ico-face.

## Global Indexing (Stable Across Partition Shape)

Independent of partition — every rank agrees on global indices.
Downstream (parallel I/O, post-processing) reads the same file
regardless of rank count.

```
// Sub-triangle global index (unchanged from subdivide() order):
//   global_tri = ico_face_idx * (4^L) + local_sub_tri_idx_within_face
//   where ico_face_idx ∈ [0, 20) and local_sub_tri_idx ∈ [0, 4^L).
// This is already the natural order; no remap needed.

global_tri_face_cochain(shell_k, tri_idx)       = shell_k * N_tri_global + tri_idx
global_rect_face_cochain(slab_k, sphere_edge)   = N_tri_faces_global
                                                 + slab_k * N_edge_s_global
                                                 + sphere_edge_idx
global_horiz_edge_cochain(shell_k, sph_edge)    = shell_k * N_edge_s_global
                                                 + sphere_edge_idx
global_vert_edge_cochain(slab_k, sph_vert)      = N_h_edges_global
                                                 + slab_k * N_vert_s_global
                                                 + sphere_vert_idx
global_vertex_cochain(shell_k, sph_vert)        = shell_k * N_vert_s_global
                                                 + sphere_vert_idx
```

## Parallel HDF5 Output

- `H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL)` at file creation.
  Pass the full `20·K`-rank comm (or a radial-bcast-gather one for small
  files).
- `H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE)` for data writes.
- Global 1D datasets of size `N_global` per cochain type.
- Each rank writes a single hyperslab covering its owned indices — a
  contiguous range per shell per ico-face, so a rectangular hyperslab.
- Ghost/halo indices excluded via hyperslab selection (don't write them).
- Empty-selection ranks still call `H5Dwrite` with `H5Sselect_none` to
  satisfy collective-mode requirements.

## Implementation Phases

### Phase 1 — Partition descriptor (no MPI, header API only)

1.1. Define `prismatic_partition` struct with:
     - `ico_face_range`: which ico-faces this rank owns (for future multi-
       face-per-rank configurations; default single-face).
     - `radial_slab`: `[k_lo, k_hi)` owned radial shells + `[k_lo-1, k_hi]`
       ghost range.
     - Ownership queries: `owns_tri(global_tri_idx)`, etc.
     - Halo-index tables: for each neighbor, which global indices are
       sent/received.

1.2. Single-rank default construction (`MPI_COMM_SELF`-equivalent):
     rank owns everything, no halos, no neighbors. Lets us wire the API
     in without an MPI dependency yet.

1.3. Unit tests: construct partition at L=2, N_r=4 (both 1-rank and
     20-rank-simulated), verify ownership partition is a disjoint union
     of owned sets covering the full global mesh, verify halo tables
     are consistent (what A sends to B matches what B receives from A).

### Phase 2 — Halo exchange abstraction (still non-MPI)

2.1. `halo_exchange<T>` class that knows how to pack/unpack buffers for
     a given cochain type (face / edge / vertex).
2.2. Non-MPI backend: `memcpy` between ranks running in the same process
     (for testing on single-node single-process).

### Phase 3 — MPI backend

3.1. Build `comm_radial` and `comm_angular` with proper topology.
3.2. Implement halo exchange via `MPI_Neighbor_alltoallv` on
     `comm_angular`, and `MPI_Sendrecv` or similar on `comm_radial`.
3.3. Handle valence-5 corners (vertical-edge dual construction at
     icosahedron vertices): gather all 5 incident ico-face's corner
     data before computing `hodge1_inv[vert_edge]`.

### Phase 4 — Solver integration

4.1. Update `prismatic_mesh` / `prismatic_mesh_metric` `build()` and
     `compute_metric()` to restrict to the rank's owned + halo regions.
4.2. Drop halo exchanges into `compute_dB_dt` and `compute_dD_dt` at the
     appropriate points (before shift-term averaging, before curl).
4.3. `apply_damping`, `apply_inner_boundary`, `apply_outer_boundary` —
     examine for cross-rank dependencies (should be purely radial; OK).
4.4. Semi-implicit iteration loops need halo exchanges each iteration.

### Phase 5 — Output & post-processing

5.1. Parallel HDF5 for `prismatic_data_exporter` and `prismatic_sph_output`.
     Global-indexed hyperslab writes.
5.2. The `prismatic_sph_output` angular grid → triangle finder needs to
     route each grid point to its owning rank. Simple: each rank
     contributes the grid points falling in its owned ico-face range,
     and participates in a collective gather before the HDF5 write.
5.3. Validate bit-identical output vs 1-rank baseline at low resolution.

### Phase 6 — Particles (deferred)

Particle depositor halo needs more thought (particles move between
ranks; momentum buffers need migration). Defer until fields work.

## Test Matrix

| Test | Ranks | L | N_r | Purpose |
|------|-------|---|-----|---------|
| Smoke | 1 | 2 | 4 | Single-rank degenerate fallback still works |
| Angular only | 20 | 3 | 4 | All ico-face partitions, no radial slabs |
| Radial only | 4 | 3 | 16 | All ranks share angular, just radial halos |
| Product | 20×4=80 | 4 | 16 | Combined decomposition |
| Production | 20×5=100 | 8 | 200 | L=8 baseline target |
| Extreme | 20×10=200 | 9 | 400 | Stress test |

Each row: bit-identical vs 1-rank up to rounding, stationary Kerr-Wald
drift rate convergence, one full evolution to t=100 M matching the
single-rank result within discretization tolerance.

## Known Pitfalls (Record for Future Sessions)

- **Valence-5 vertices** (12 of them, at original icosahedron corners).
  Source of most partition-boundary complexity. Every cross-rank bug
  shows up here first. Always verify with targeted unit tests.
- **`edge_tris` / `vert_tris` adjacency tables** were built globally on
  host pre-scatter. After partitioning, each rank needs only local +
  halo; keep the same layout convention so the existing Hodge kernels
  don't need changing.
- **Owner convention matters for Hodge construction**:
  - Vertical-edge `hodge1_inv` (polygon fan around vertex) spans 5 faces
    at valence-5 corners — owner must gather halos before computing.
  - Tri-face `hodge2` at ico-edge-adjacent triangles needs the neighbor
    triangle's circumcenter — which is in the halo.
- **`MPI_Dist_graph_create_adjacent` inputs are per-rank and local** —
  each rank computes its own adjacency list from the ico-face index,
  no global graph needed.
- **Empty selections in collective HDF5 writes**: use `H5Sselect_none()`
  but still call `H5Dwrite` from every rank holding the file comm.
- **CUDA-aware MPI**: use it. `cudaMemcpyDeviceToHost` → `MPI_Send` →
  `cudaMemcpyHostToDevice` is ~10× slower than
  `MPI_Send(device_ptr)` with CUDA-aware transport.
- **Checkpoint/restart format** must be partition-independent. Use the
  global-index scheme above, so a checkpoint from a 20-rank run can be
  restarted with 100 ranks.
