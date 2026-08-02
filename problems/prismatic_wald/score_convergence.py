#!/usr/bin/env python3
"""Assemble the Kerr-Wald convergence table from the _ana / _relax families.

Three separate quantities, deliberately kept apart because they have different
error floors:

  1. analytic(mesh) vs continuum   -- how well the discrete cap-flux observable
                                      represents the continuum Wald value on
                                      this mesh.  No evolution involved.
  2. relaxed vs analytic(mesh)     -- how far the evolved state ends up from
                                      the scheme's own sampled stationary
                                      state.  This is what the relaxation test
                                      actually exercises.
  3. relaxed vs continuum          -- the end-to-end number.  Its apparent
                                      order is NOT meaningful when (1) and (2)
                                      are comparable in size and opposite in
                                      sign, which is exactly what happens here.

Usage: score_convergence.py 3 4 5 [--window 150 220]
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FLUX = os.path.join(HERE, "flux_cap.py")


def _run(args):
    r = subprocess.run([sys.executable, FLUX] + args, capture_output=True,
                       text=True)
    if r.returncode != 0:
        sys.exit(f"flux_cap.py failed on {args}:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def analytic(rundir):
    out = _run([rundir, "--file", "ic_aux.h5"])
    line = [l for l in out.splitlines() if "Phi_cap(horizon)/" in l][0]
    cont = [l for l in out.splitlines() if "continuum Wald" in l][0]
    return float(line.split()[-1]), float(cont.split()[-1])


def relaxed(rundir, lo, hi):
    out = _run([rundir, "--series"])
    rows = [[float(x) for x in l.split()]
            for l in out.splitlines()[2:] if l.strip()]
    d = np.array(rows)
    t, r = d[:, 1], d[:, 2]
    m = (t >= lo) & (t <= hi)
    if m.sum() < 3:
        sys.exit(f"{rundir}: only {m.sum()} snapshots in [{lo},{hi}] M")
    w = r[m]
    return w.mean(), (w.max() - w.min()), int(m.sum()), t.max()


def order(e_coarse, e_fine):
    if e_fine == 0 or np.sign(e_coarse) != np.sign(e_fine):
        return np.nan
    return np.log2(abs(e_coarse) / abs(e_fine))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("levels", nargs="+", type=int)
    ap.add_argument("--window", nargs=2, type=float, default=[150.0, 220.0])
    ap.add_argument("--suffix", default="",
                    help="run-dir suffix, e.g. _whitney for the Whitney "
                         "Galerkin Hodge family (GRPIC_PLAN B1b)")
    args = ap.parse_args()
    lo, hi = args.window

    rows = []
    cont = None
    for L in args.levels:
        ana_dir = os.path.join(HERE, f"Data_conv_L{L}_ana{args.suffix}")
        rlx_dir = os.path.join(HERE, f"Data_conv_L{L}_relax{args.suffix}")
        a, c = analytic(ana_dir)
        cont = c
        entry = dict(L=L, ana=a, cont=c)
        if os.path.isdir(rlx_dir) and glob.glob(os.path.join(rlx_dir,
                                                             "step_*.h5")):
            mu, ring, n, tmax = relaxed(rlx_dir, lo, hi)
            entry.update(relax=mu, ring=ring, n=n, tmax=tmax)
        rows.append(entry)

    print(f"Kerr-Wald cap-flux ratio  Phi_cap(r=1.10569) / Phi_cap(r=8.016)")
    print(f"continuum Wald value: {cont:.6e}    "
          f"relaxation window: {lo:g}-{hi:g} M\n")

    hdr = (f"{'L':>3} {'analytic(mesh)':>15} {'(1) vs cont':>12} "
           f"{'relaxed':>14} {'ring':>7} {'(2) vs ana':>11} {'(3) vs cont':>12}")
    print(hdr)
    print("-" * len(hdr))
    for e in rows:
        e1 = e["ana"] / e["cont"] - 1.0
        if "relax" in e:
            e2 = e["relax"] / e["ana"] - 1.0
            e3 = e["relax"] / e["cont"] - 1.0
            print(f"{e['L']:>3} {e['ana']:15.6e} {e1:+11.4%} "
                  f"{e['relax']:14.6e} {e['ring']/e['ana']:6.3%} "
                  f"{e2:+10.4%} {e3:+11.4%}")
        else:
            print(f"{e['L']:>3} {e['ana']:15.6e} {e1:+11.4%} "
                  f"{'--':>14} {'--':>7} {'--':>11} {'--':>12}")

    print("\nobserved orders (log2 of successive error ratios):")
    print(f"  {'L->L+1':>8} {'(1) observable':>15} {'(2) relaxation':>15} "
          f"{'(3) end-to-end':>15}")
    for i in range(len(rows) - 1):
        c_, f_ = rows[i], rows[i + 1]
        o1 = order(c_["ana"] / c_["cont"] - 1, f_["ana"] / f_["cont"] - 1)
        if "relax" in c_ and "relax" in f_:
            o2 = order(c_["relax"] / c_["ana"] - 1, f_["relax"] / f_["ana"] - 1)
            o3 = order(c_["relax"] / c_["cont"] - 1,
                       f_["relax"] / f_["cont"] - 1)
        else:
            o2 = o3 = np.nan
        def fmt(v):
            return f"{v:+15.2f}" if np.isfinite(v) else f"{'n/a':>15}"
        label = f"{c_['L']}->{f_['L']}"
        print(f"  {label:>8} {fmt(o1)} {fmt(o2)} {fmt(o3)}")


if __name__ == "__main__":
    main()
