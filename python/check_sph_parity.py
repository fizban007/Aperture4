#!/usr/bin/env python3
"""7D parity check: sph_from_dump.py output vs the in-code sph files.

Agreement criteria (established 2026-07-19 on the L2 PIC acceptance run):
  - every dataset matches to <= 1e-4 relative (measured ~2e-6, the C++
    float pipeline's roundoff) EXCEPT at
  - grid points lying EXACTLY on mesh facet boundaries (min barycentric
    < 1e-12), where the primal Whitney interpolant's normal component is
    discontinuous and the two implementations legitimately pick
    different one-sided limits (E/J only; the recovery B is C0 and has
    no such points), and
  - gamma_mean in vacuum regions (rho_abs below `--vacuum-frac` of its
    max), where the weight ratio is 0/0 noise, and
  - the exact pole rows of Bph (the in-code path leaks a float sin(pi)
    residual at the south pole; the tool writes the intended 0).

Usage: check_sph_parity.py DATA_DIR STEP [--suffix _post]
Exit code 0 = parity holds.
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prismatic_recovery import Mesh, locate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("step", type=int)
    ap.add_argument("--suffix", default="_post")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--vacuum-frac", type=float, default=1e-6)
    args = ap.parse_args()

    d = args.data_dir
    a = h5py.File(os.path.join(d, f"sph_{args.step:06d}.h5"), "r")
    b = h5py.File(os.path.join(d, f"sph{args.suffix}_{args.step:06d}.h5"),
                  "r")
    g = h5py.File(os.path.join(d, "sph_grid.h5"), "r")
    nth, nph = int(g["N_theta"][()]), int(g["N_phi"][()])
    nr = int(g["N_r"][()])

    mesh = Mesh(os.path.join(d, "mesh.h5"))
    theta = np.pi * np.arange(nth) / (nth - 1)
    phi = 2 * np.pi * np.arange(nph) / nph
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    s = np.stack(
        [(np.sin(th) * np.cos(ph)).ravel(),
         (np.sin(th) * np.sin(ph)).ravel(), np.cos(th).ravel()], axis=1)
    r_loc = 0.5 * (mesh.radii[0] + mesh.radii[1])
    _, _, lam, _ = locate(mesh, r_loc * s)
    tie = np.tile(lam.min(axis=1) < 1e-12, nr + 1)

    pole = np.zeros((nth, nph), dtype=bool)
    pole[0] = pole[-1] = True
    pole = np.tile(pole.ravel(), nr + 1)

    ra = a["rho_abs"][...].astype(np.float64) if "rho_abs" in a else None

    ok = True
    for k in sorted(set(a.keys()) & set(b.keys()) - {"step", "time"}):
        x = a[k][...].astype(np.float64)
        y = b[k][...].astype(np.float64)
        mask = np.zeros(x.size, dtype=bool)
        if k in ("Er", "Eth", "Eph", "Jr", "Jth", "Jph"):
            mask |= tie
        if k == "Bph":
            mask |= pole
        if k == "gamma_mean" and ra is not None:
            mask |= np.abs(ra) < args.vacuum_frac * np.abs(ra).max()
        scale = max(np.max(np.abs(x)), 1e-30)
        rel = np.abs(x - y) / scale
        rel[mask] = 0.0
        status = "OK" if rel.max() < args.tol else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{k:12s} max rel {rel.max():.3e} "
              f"(masked {int(mask.sum())}/{x.size})  {status}")
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
