# Dual-Grid B Interpolation for Continuous Fields at Particle Positions

## Problem

Whitney 2-forms on the primal prismatic mesh give discontinuous B when
evaluated pointwise.  Specifically, the tangential (parallel) component
of B jumps when a particle crosses a prism face.  Whitney 1-forms for E
have the complementary problem: E_normal jumps across faces.  These
jumps cause numerical scattering in the Boris push, leading to artificial
pitch-angle diffusion and energy errors over long integrations.

On Cartesian (Yee) grids this issue does not arise because the tensor-
product hat functions are fully continuous.  On simplicial or prismatic
meshes it is inherent to lowest-order Whitney forms.

## Proposed Solution

Interpolate B using **Whitney 1-forms on the dual grid**, where B
naturally lives as a dual 1-cochain (H = ★₂B).  By the continuity
property of Whitney 1-forms, the tangential component — which is the
one that's discontinuous on the primal — becomes continuous on the dual.

Similarly, E can be interpolated using dual Whitney 2-forms to fix the
E_normal discontinuity, though this is less critical for the Boris push
since B dominates particle dynamics.

## Why the Dual Grid is Tractable

The prismatic mesh is a tensor product of an icosahedral sphere
triangulation (angular) and a 1D radial grid.  The dual inherits this
tensor-product structure:

- **Angular dual**: Voronoi tessellation of the sphere triangulation.
  Each Voronoi cell is a polygon (5 sides at the 12 icosahedral
  vertices, 6 sides everywhere else).  Each polygon decomposes into
  5-6 triangles by connecting edges to the polygon center (the
  primal vertex = dual cell center).  This gives a dual angular
  triangulation that our existing Whitney machinery can operate on.

- **Radial dual**: The dual shells sit at the midpoints of primal
  radial layers: r*_k = 0.5 * (radii[k] + radii[k+1]).  The dual
  radial intervals span from one primal shell midpoint to the next.

So each dual cell (polygon × interval) decomposes into 5-6 triangular
prisms — exactly the same element type as the primal mesh.  All existing
infrastructure (Whitney forms, barycentric coordinates, point location)
can be reused on the dual.

## Dual Grid Geometry

### Dual vertices
One per prism: the circumcenter at (angular_circumcenter, radial_midpoint).
Already computed in `compute_geometric_dual()` but not persisted.

- Store: `dual_vx[N_prisms]`, `dual_vy[N_prisms]`, `dual_vz[N_prisms]`
- Prism index: `pid = k * N_tri + t` for triangle t in layer k
- N_prisms = N_tri * N_r

### Dual angular mesh (Voronoi polygons → triangles)
For each primal vertex s on the sphere, the Voronoi polygon has vertices
at the circumcenters of the triangles surrounding s.  The vertex
valence is 5 (icosahedral vertices) or 6 (all others).

Decompose each polygon into triangles by connecting the circumcenters
to the primal vertex (which serves as the "center"):
- Dual triangle vertices: (primal_vertex_s, circumcenter_t1, circumcenter_t2)
  where t1 and t2 are adjacent triangles sharing vertex s.

Need to store:
- `dual_tri_verts[N_dual_tri * 3]`: vertex indices into the dual
  vertex array (circumcenter indices)
- `dual_tri_edges[N_dual_tri * 3]`: dual edge indices
- Number of dual triangles: sum of valences = 2 * N_edge_s
  (each sphere edge contributes one dual triangle on each side)
  Actually: each primal vertex with valence v contributes v dual
  triangles.  Total = sum_v valence_v = 2 * N_edge_s (by handshaking
  lemma, since each edge contributes 1 to each endpoint's valence).

### Dual edges
One per primal face.  A dual edge connects the circumcenters of the
two prisms sharing that face.

- Triangular faces (top/bottom of prisms): dual edge is radial,
  connecting circumcenters of prisms in layers k and k+1 through
  the same triangle.  Length = |r*_{k+1} - r*_k|.

- Rectangular faces (sides of prisms): dual edge is angular,
  connecting circumcenters of two prisms sharing a sphere edge in
  the same layer.  Length already computed as part of hodge2.

### Dual 1-cochain values (H = ★₂B)
For each primal face f, the dual edge e* carries:
```
H_e* = hodge2[f] * B_f[f]
```
This is trivially computed from existing data — no new solve needed.

## Point Location on the Dual Grid

### Radial
The dual shells are at r*_k = midpoint(radii[k], radii[k+1]).
Binary search as before, but on the dual radii array.

### Angular
The dual triangulation on the sphere is different from the primal.
Need a separate `find_dual_triangle()` walk algorithm with dual
adjacency data.

Alternative: since we know the particle's primal triangle, and the
dual triangles are the Voronoi cells of the primal vertices, we can
determine the dual triangle from the barycentric coordinates.  The
particle is in the Voronoi cell of the nearest primal vertex, which
corresponds to the vertex with the largest barycentric coordinate.
The specific dual triangle within that Voronoi cell can be determined
by the angular position relative to the surrounding circumcenters.

## Interpolation Formula

Once the dual prism is identified and dual barycentric coordinates
(μ₁, μ₂, μ₃) and dual ζ* are computed, the Whitney 1-form
interpolation on the dual is identical in structure to the primal:

```
B(x) = Σ_e* H_e* W¹_e*(x)
```

where the sum is over the 9 dual edges of the dual prism, and
W¹_e* are Whitney 1-forms defined using the dual vertex positions
and dual barycentric coordinates.

The implementation can reuse `interpolate_fields()` with dual
mesh_ptrs.

## Implementation Steps

1. **Persist circumcenters** in `prismatic_mesh`:
   - Store `dual_cx[N_prisms]`, `dual_cy`, `dual_cz` (circumcenter
     positions in 3D, currently computed but discarded)
   - Store dual angular positions on unit sphere: `dual_sx`, `dual_sy`,
     `dual_sz` (circumcenter projected to sphere)

2. **Build dual angular triangulation**:
   - For each primal vertex, find surrounding triangles in order
   - Create dual triangles: (vertex_s, circumcenter_t1, circumcenter_t2)
     for each pair of adjacent triangles sharing vertex s
   - Build dual edge list and adjacency (dual_tri_neighbor)
   - Store in mesh alongside primal data

3. **Build dual_mesh_ptrs** (analogous to prismatic_mesh_ptrs):
   - Dual vertex positions, dual triangle connectivity, dual radii
   - Dual edge indices per dual prism
   - Same HD_INLINE methods: compute_barycentric, find_triangle, etc.

4. **Compute dual 1-cochain at runtime**:
   - Before particle interpolation: `H_e*[f] = hodge2[f] * B_f[f]`
   - This is a simple elementwise multiply, O(N_faces)

5. **Dual interpolation function**:
   - `interpolate_B_dual(dual_mp, dual_tri, dual_layer, μ, ζ*, H, Bx, By, Bz)`
   - Same structure as primal `interpolate_fields` but using dual geometry

6. **Update particle pusher**:
   - Before Boris push, interpolate B from dual grid instead of primal
   - E can still use primal interpolation (E_tangential is continuous)
   - Or: also interpolate E from dual for full continuity

## Cost Estimate

- Storage: ~3 × N_prisms floats for circumcenter positions,
  ~4 × N_dual_tri ints for connectivity ≈ modest overhead
- Runtime per particle: one extra point location on dual grid +
  one Whitney 1-form evaluation ≈ 2× current interpolation cost
- Runtime per step for H computation: O(N_faces) elementwise multiply ≈ negligible

## Alternatives Considered

- **Second-order Whitney forms**: Fully continuous but requires
  non-diagonal mass matrix (CG solve per timestep), ~10× slower field
  solver.  Major rewrite.

- **Simple averaging**: Blend interpolation from neighboring prisms
  near faces.  Cheap, easy to implement, reduces jumps from O(h) to
  O(h²) but not mathematically exact.  Good as a quick fix.

- **Smoothed projection**: Project B onto a vertex-based field
  (volume-weighted average to vertices), then interpolate from
  vertices using Whitney 0-forms (fully continuous).  Loses accuracy
  but simple.  Common in FEM post-processing.
