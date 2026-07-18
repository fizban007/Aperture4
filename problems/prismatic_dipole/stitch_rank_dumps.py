#!/usr/bin/env python3
"""Stitch distributed per-rank field dumps into global cochain arrays.

The distributed dec_field_solver (4.1b) writes one file per rank and
step:  <dir>/rank<R>_step_<S>.h5  containing this rank's OWNED slots of
the total E/B cochains in four blocks (E_h, E_v, B_tri, B_rect) plus
the matching global-index maps (*_g, each within its own cochain range).
This script reassembles the combined global [h|v] edge and [tri|rect]
face arrays — the same layout the single-rank exporter writes as
E_e / B_f in step_<S>.h5 — and optionally compares against such a
reference run.

Usage:
  python stitch_rank_dumps.py <rank_dump_dir> <step> [--ref <single_run_dir>]
                              [-o stitched.h5]
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np


def stitch(dump_dir, step):
    files = sorted(glob.glob(os.path.join(dump_dir, f"rank*_step_{step:06d}.h5")))
    if not files:
        sys.exit(f"no rank dumps for step {step} in {dump_dir}")

    # Infer mesh dimensions from the union of global indices.
    n_h = n_v = n_t = n_r = 0
    for fn in files:
        with h5py.File(fn, "r") as f:
            n_h = max(n_h, int(np.max(f["E_h_g"])) + 1)
            n_v = max(n_v, int(np.max(f["E_v_g"])) + 1)
            n_t = max(n_t, int(np.max(f["B_tri_g"])) + 1)
            n_r = max(n_r, int(np.max(f["B_rect_g"])) + 1)

    E = np.full(n_h + n_v, np.nan, dtype=np.float64)
    B = np.full(n_t + n_r, np.nan, dtype=np.float64)
    time = None
    for fn in files:
        with h5py.File(fn, "r") as f:
            E[np.asarray(f["E_h_g"])] = np.asarray(f["E_h"])
            E[n_h + np.asarray(f["E_v_g"])] = np.asarray(f["E_v"])
            B[np.asarray(f["B_tri_g"])] = np.asarray(f["B_tri"])
            B[n_t + np.asarray(f["B_rect_g"])] = np.asarray(f["B_rect"])
            time = float(np.asarray(f["time"]))
    if np.isnan(E).any() or np.isnan(B).any():
        sys.exit("ownership gap: some global slots were not covered by any rank")
    return E, B, time, (n_h, n_v, n_t, n_r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("step", type=int)
    ap.add_argument("--ref", help="single-rank run dir with step_<S>.h5 (exporter)")
    ap.add_argument("-o", "--out", help="write stitched E_e/B_f to this h5 file")
    args = ap.parse_args()

    E, B, time, dims = stitch(args.dump_dir, args.step)
    print(f"stitched step {args.step} (t = {time:.6f}): "
          f"{len(E)} edges ({dims[0]} h + {dims[1]} v), "
          f"{len(B)} faces ({dims[2]} tri + {dims[3]} rect)")

    if args.ref:
        with h5py.File(os.path.join(args.ref, f"step_{args.step:06d}.h5")) as f:
            Es, Bs = np.asarray(f["E_e"]), np.asarray(f["B_f"])
        for name, a, b in [("E", E, Es), ("B", B, Bs)]:
            d = np.abs(a - b).max()
            print(f"  vs ref {name}: maxdiff = {d:.3e} "
                  f"(rel {d / max(np.abs(b).max(), 1e-30):.3e})")

    if args.out:
        with h5py.File(args.out, "w") as f:
            f.create_dataset("E_e", data=E)
            f.create_dataset("B_f", data=B)
            f.create_dataset("time", data=time)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
