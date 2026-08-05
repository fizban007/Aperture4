# Fast spherical-grid interpolation for prismatic dumps

How to turn the exporter's mesh-native dumps (`step_XXXXXX.h5` + `mesh.h5`)
into `sph_XXXXXX.h5` files on an (r, θ, φ) grid, using the compiled
interpolation module `prismatic_interp` — and how to fan the conversion out
across workers on an analysis cluster (Andes).

## What it is

`prismatic_interp.cpp` is a standalone pybind11 module (no Aperture library
dependency) that provides the two hot kernels of `sph_from_dump.py`:

| binding         | replaces (pure Python)          | what it does |
|-----------------|---------------------------------|--------------|
| `locate_points` | `prismatic_recovery.locate`     | triangle walk on the unit sphere, central-projection barycentric λ |
| `gather_shells` | `sph_from_dump.gather_at_shell` | Whitney-form E/B (or J/B) gather at fixed angular points for a list of shells with explicit (layer, ζ) |

All math is float64 and follows the Python conventions exactly (the
per-triangle inverse matrices for the barycentric predicate are computed in
numpy and passed in), so results are **bit-exact** against the pure-Python
path — verified on the L6 a60 run, step 16000, every dataset.

`sph_from_dump.py` auto-detects the module: if
`prismatic_interp*.so` sits next to it, the fast path is used; otherwise it
silently falls back to pure Python. No flags needed.

Speed (L6: N_r = 204, 40962 sphere vertices, 64×128 angular grid, one core):

- compiled: **~19 s/step** end-to-end (~2 s in the gathers; the rest is
  mesh setup + reading the ~500 MB dump)
- pure Python: ~30 min/step

There is also an older `interpolate_slice` binding (E/B on (x, z) points in
the y = 0 plane, float32, production perpendicular-projection barycentric)
kept for the original recovery-study workflow.

## Building the module

Requires a C++17 compiler and pybind11 (`pip install pybind11`). Build
against the **same python you will run with** (the `.so` is
interpreter-ABI-specific — rebuild when switching machines or envs):

```sh
cd python/
PY=/path/to/your/python          # must have numpy + h5py
$PY -m pip install pybind11      # once
g++ -O3 -std=c++17 -shared -fPIC \
    $($PY -m pybind11 --includes) \
    prismatic_interp.cpp \
    -o prismatic_interp$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
```

Notes:

- On the Frontier login nodes the system `g++` is 7.5, which is too old for
  pybind11 ≥ 3; use `g++-13` (present in `/usr/bin`). On Andes,
  `module load gcc` gives a recent one.
- The CMake route also exists (`python/CMakeLists.txt`,
  `pybind11_add_module`), but the one-liner above is all it takes.
- The already-built `prismatic_interp.cpython-310-x86_64-linux-gnu.so` in
  this directory was compiled on a Frontier login node for the
  `aperture_post` venv (python 3.10); rebuild for anything else.

## Converting dumps

```sh
python sph_from_dump.py DATA_DIR --no-recovery                # every step_*.h5
python sph_from_dump.py DATA_DIR --no-recovery --steps 0 640  # explicit steps
```

- `--no-recovery` uses the primal Whitney B gather. **Recommended**: the
  default C0 vertex recovery runs `build_recovery` — hours of pure Python at
  L6 — and only affects the smoothness of B (E, J, ρ, ρ_abs, γ are
  identical either way).
- Grid resolution comes from an existing `sph_grid.h5`, else defaults to
  64×128; override with `--ntheta/--nphi`. If you change the resolution,
  **delete the old `sph_grid.h5` first** — the writer skips it when present
  (so parallel workers don't race), and stale grid metadata would disagree
  with the new files.
- `--suffix X` writes `sphX_XXXXXX.h5` / `sphX_grid.h5` — useful for
  A/B comparisons without clobbering existing output.

### Fanning out on Andes

Per-process setup (mesh load, analytic Hodge/dual-volume tables, point
location) is ~15 s, so give each worker a *chunk* of steps rather than one
step. Workers with disjoint `--steps` are independent; the shared
`sph_grid.h5` is written once thanks to the existence guard. E.g. 4 workers
× ~7 steps for a 26-dump run:

```sh
seq 0 640 16000 | xargs -n 7 | \
  xargs -P 4 -I{} sh -c \
    'python sph_from_dump.py DATA_DIR --no-recovery --steps {}'
```

(or the same split as a Slurm job array). Memory is ~2–3 GB per worker
(float64 copies of the cochains). The conversion is I/O-heavy — reading the
dump usually dominates the compiled compute, so past ~8 workers the
filesystem is the limit, not cores.

## Output conventions

Same as the retired in-code single-rank `prismatic_sph_output` (see the
`sph_from_dump.py` docstring for the full story):

- `Er/Eth/Eph`, `Jr/Jth/Jph` — covariant coordinate-basis components
  (E_r̂, E_θ̂·r, E_φ̂·r·sinθ)
- `Br/Bth/Bph` — contravariant (B_r̂, B_θ̂/r, B_φ̂/(r·sinθ)); exact-pole
  B^φ gets a 0 sentinel at **both** poles (the in-code writer left a float
  sin(π) residual at the south pole)
- `rho`, `rho_abs` — vertex densities / lumped dual volume;
  `gamma_mean = gamma_wsum / rho_abs`
- shells sample layer boundaries: shell k interpolates in the layer
  *starting* at the shell (top shell: last layer, ζ = 1) — part of the
  output definition, since Whitney normal components jump across shells

Expected agreement with in-code `sph_` dumps: ~1e-5 for ρ/γ and the
in-plane components, few-percent for B and the φ̂-components (that writer
computes in float32, uses the C0 recovery for B, and the
perpendicular-projection barycentric).

## Direct API use

For custom sampling (other grids, slices, field lines), call the bindings
directly; `sph_from_dump._fast_mesh_args(mesh)` marshals and caches the
mesh arrays in the right dtypes:

```python
from prismatic_recovery import Mesh
from sph_from_dump import _fast_mesh_args
import prismatic_interp, numpy as np

mesh = Mesh("DATA_DIR/mesh.h5")
fm = _fast_mesh_args(mesh)

s_hat = np.ascontiguousarray(pts_unit)            # (N, 3) float64, |s|=1
tri, lam = prismatic_interp.locate_points(*fm, s_hat)

layers = np.array([...], dtype=np.int32)          # (NS,) radial layer per shell
zetas  = np.array([...])                          # (NS,) ζ in [0, 1]
E, B = prismatic_interp.gather_shells(*fm, tri, lam, layers, zetas, E_e, B_f)
# E, B: (NS, N, 3) float64 Cartesian vectors
```

`E_e`/`B_f` are the float64 cochain arrays from a dump; for currents pass
`(J_e, B_f)` and keep the first return (remember the
`hodge1_inv` conversion first when the dump flags `J_kind_dual2` — see
`sph_from_dump.process_step`). Angular location is radius-independent, so
one `locate_points` call serves all shells.

## Movie pipeline

Corotating-frame movie from the converted files (needs matplotlib ≥ 3.6
for `AsinhNorm`, and ffmpeg on PATH):

```sh
python ../problems/prismatic_dipole/make_rotating_meridional_movie.py DATA_DIR \
    [--omega 0.25 --bp 1000 --alpha-deg 60 --rmax 8 --fps 8 --out movie.mp4]
```

The movie title uses the basename of `DATA_DIR`; symlink the output
directory to a descriptive name if you want a better title than `output`.
