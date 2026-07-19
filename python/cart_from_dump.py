#!/usr/bin/env python3
"""Cartesian-box sampling of the exporter's mesh-native dumps (7D F9).

The visualization-convenience companion of sph_from_dump.py: samples
E/B (and J/rho when present) on a regular Cartesian grid — the accuracy
vs viz-convenience trade is made in POST, not at run time.  Points
outside the radial domain get zeros.

Writes cart_XXXXXX.h5 with Ex/Ey/Ez, Bx/By/Bz (recovery-gathered
unless --no-recovery), optionally Jx/Jy/Jz and rho, plus the grid
axes.  Shapes are (nz, ny, nx), C-order.

Usage:
  python cart_from_dump.py DATA_DIR [--steps ...] [-n 64]
         [--extent R]  (default: the outer mesh radius)
"""
import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prismatic_recovery import (  # noqa: E402
    Mesh,
    build_recovery,
    locate,
    primal_gather,
    recovery_gather,
    vertex_field,
)
from sph_from_dump import hodge1_inv, vert_dual_vol  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("--steps", type=int, nargs="*", default=None)
    ap.add_argument("-n", type=int, default=64, help="points per axis")
    ap.add_argument("--extent", type=float, default=None)
    ap.add_argument("--no-recovery", action="store_true")
    args = ap.parse_args()

    mesh = Mesh(os.path.join(args.data_dir, "mesh.h5"))
    ext = args.extent or float(mesh.radii[-1])
    ax = np.linspace(-ext, ext, args.n)
    Z, Y, X = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    r = np.linalg.norm(pts, axis=1)
    inside = (r >= mesh.radii[0]) & (r <= mesh.radii[-1])

    h1inv = (mesh.hodge1_inv.astype(np.float64)
             if hasattr(mesh, "hodge1_inv") else hodge1_inv(mesh))
    dvol = (mesh.vert_dual_vol.astype(np.float64)
            if hasattr(mesh, "vert_dual_vol") else vert_dual_vol(mesh))

    rec = None
    if not args.no_recovery:
        print("Building vertex recovery weights ...")
        rec = build_recovery(mesh)

    if args.steps is not None:
        paths = [os.path.join(args.data_dir, f"step_{s:06d}.h5")
                 for s in args.steps]
    else:
        paths = sorted(glob.glob(os.path.join(args.data_dir, "step_*.h5")))

    shape = (args.n, args.n, args.n)
    p_in = pts[inside]
    hints = np.zeros(len(p_in), dtype=int)
    for path in paths:
        m = re.search(r"step_(\d+)\.h5$", path)
        if m is None:
            continue
        step = int(m.group(1))
        with h5py.File(path, "r") as f:
            E_e = f["E_e"][()].astype(np.float64)
            B_f = f["B_f"][()].astype(np.float64)
            J_e = f["J_e"][()].astype(np.float64) if "J_e" in f else None
            rho = f["rho"][()].astype(np.float64) if "rho" in f else None
            time = float(f["time"][()])
            if J_e is not None and "J_kind_dual2" in f and f["J_kind_dual2"][()]:
                J_e = h1inv * J_e

        E, B = primal_gather(mesh, E_e, B_f, p_in, hints=hints)
        if rec is not None:
            Bv = vertex_field(mesh, rec, B_f)
            B = recovery_gather(mesh, Bv, p_in, hints=hints)
        out = {}
        for name, arr in (("Ex", E[:, 0]), ("Ey", E[:, 1]), ("Ez", E[:, 2]),
                          ("Bx", B[:, 0]), ("By", B[:, 1]), ("Bz", B[:, 2])):
            full = np.zeros(len(pts))
            full[inside] = arr
            out[name] = full.reshape(shape)
        if J_e is not None:
            J, _ = primal_gather(mesh, J_e, B_f, p_in, hints=hints)
            for name, arr in (("Jx", J[:, 0]), ("Jy", J[:, 1]),
                              ("Jz", J[:, 2])):
                full = np.zeros(len(pts))
                full[inside] = arr
                out[name] = full.reshape(shape)
        if rho is not None:
            tri, layer, lam, zeta = locate(mesh, p_in, hints=hints)
            val = np.zeros(len(p_in))
            for lev in range(2):
                w_lev = (1.0 - zeta) if lev == 0 else zeta
                for vi in range(3):
                    sv = mesh.tri_verts[tri, vi]
                    v_idx = (layer + lev) * mesh.N_vert_s + sv
                    val += rho[v_idx] / dvol[v_idx] * lam[:, vi] * w_lev
            full = np.zeros(len(pts))
            full[inside] = val
            out["rho"] = full.reshape(shape)

        opath = os.path.join(args.data_dir, f"cart_{step:06d}.h5")
        with h5py.File(opath, "w") as f:
            for name, arr in out.items():
                f.create_dataset(name, data=arr.astype(np.float32))
            f.create_dataset("axis", data=ax)
            f.create_dataset("step", data=np.int32(step))
            f.create_dataset("time", data=np.float64(time))
        print(f"  cart_{step:06d}.h5: time={time:.4f}")


if __name__ == "__main__":
    main()
