#!/usr/bin/env python3
"""A2.0 stationarity analysis for the Deutsch benchmark.

The run (config_deutsch_L*.toml) starts from the exact retarded Deutsch
solution with the Deutsch inner BC, so the true solution is periodic
with P = 2*pi/Omega and dt is chosen with an integer number of steps per
period.  The recurrence error

    err(t) = ||F(t + P) - F(t)|| / ||F(t)||

on the raw cochains is then pure solver error (discretization drift +
imperfect outer-layer absorption).  Reported over the full domain and
restricted to r < r_cut (inside the damping layer), for both B_f and E_e.

Usage: deutsch_stationarity.py <run_dir> --steps-per-period N [--r-cut 9]
Compare err across L=4/5/6 runs for the convergence figure.
"""

import argparse
import glob
import os
import re

import h5py
import numpy as np


def load_run(run_dir):
    snaps = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "step_*.h5"))):
        m = re.search(r"step_(\d+)\.h5", path)
        if m:
            snaps[int(m.group(1))] = path
    with h5py.File(os.path.join(run_dir, "mesh.h5"), "r") as f:
        mesh = {
            "radii": f["radii"][()],
            "edge_layer": f["edge_radial_layer"][()],
            "face_layer": f["face_radial_layer"][()],
        }
    return snaps, mesh


def rel_err(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--steps-per-period", type=int, required=True)
    ap.add_argument("--r-cut", type=float, default=9.0)
    args = ap.parse_args()

    snaps, mesh = load_run(args.run_dir)
    spp = args.steps_per_period
    radii = mesh["radii"]
    emask = radii[mesh["edge_layer"]] < args.r_cut
    fmask = radii[mesh["face_layer"]] < args.r_cut

    pairs = [(s, s + spp) for s in sorted(snaps) if s + spp in snaps]
    if not pairs:
        raise SystemExit("no snapshot pairs one period apart")

    print(f"# {args.run_dir}: {len(pairs)} period-separated pairs, "
          f"r_cut={args.r_cut}")
    print(f"# {'step':>7} {'errB_full':>11} {'errB_rcut':>11} "
          f"{'errE_full':>11} {'errE_rcut':>11}")
    results = []
    for s0, s1 in pairs:
        with h5py.File(snaps[s0], "r") as f0, h5py.File(snaps[s1], "r") as f1:
            B0, B1 = f0["B_f"][()], f1["B_f"][()]
            E0, E1 = f0["E_e"][()], f1["E_e"][()]
        row = (s0, rel_err(B1, B0), rel_err(B1, B0, fmask),
               rel_err(E1, E0), rel_err(E1, E0, emask))
        results.append(row)
        print(f"  {row[0]:7d} {row[1]:11.4e} {row[2]:11.4e} "
              f"{row[3]:11.4e} {row[4]:11.4e}")

    # headline number: last available pair (most settled)
    last = results[-1]
    print(f"# headline (t={last[0]}dt -> +P): "
          f"errB(r<{args.r_cut}) = {last[2]:.4e}, "
          f"errE(r<{args.r_cut}) = {last[4]:.4e}")


if __name__ == "__main__":
    main()
