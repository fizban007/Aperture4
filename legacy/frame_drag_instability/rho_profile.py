#!/usr/bin/env python
"""Phi-mean rho, rho_abs, E_par and multiplicity vs r at chosen colats."""
import sys
import numpy as np
import h5py, os

OMEGA = 0.25
NT, NP = 64, 128
fn = sys.argv[1]
colats = [float(c) for c in sys.argv[2:]] or [60.0, 75.0, 90.0]
with h5py.File(fn, "r") as f:
    d = {k: f[k][()].reshape(205, NT, NP) for k in
         ("Er", "Eth", "Eph", "Br", "Bth", "Bph", "rho", "rho_abs")}
    t = float(f["time"][()])
run = os.path.dirname(fn)
with h5py.File(os.path.join(run, "mesh.h5"), "r") as g:
    radii = g["radii"][()].astype(np.float64)
theta = (np.arange(NT) + 0.5) * np.pi / NT
print(f"# {fn.split('/')[-3]} t={t:.2f}")
for c in colats:
    it = int(np.argmin(np.abs(theta - np.radians(c))))
    print(f"## colat {np.degrees(theta[it]):.1f}")
    print(f"{'k':>3} {'r':>7} {'rho':>9} {'rho_abs':>9} {'|rho|/abs':>9} {'E_par/B':>9}")
    for k in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        rho = d["rho"][k, it].mean()
        rab = d["rho_abs"][k, it].mean()
        Br, Bth, Bph = d["Br"][k, it], d["Bth"][k, it], d["Bph"][k, it]
        B = np.sqrt(Br**2 + Bth**2 + Bph**2)
        epar = ((d["Er"][k, it]*Br + d["Eth"][k, it]*Bth +
                 d["Eph"][k, it]*Bph) / B**2).mean()
        print(f"{k:3d} {radii[k]:7.4f} {rho:9.3g} {rab:9.3g} "
              f"{abs(rho)/max(rab,1e-30):9.3f} {epar:9.5f}")
