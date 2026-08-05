#!/usr/bin/env python
"""Phi-averaged E x B drift rate omega(r) at chosen colatitudes.

omega_FIDO = (E x B)_phi / (B^2 r sin(theta)), phi-averaged.
Compare to Omega and Omega - w_lt(r).
"""
import sys
import numpy as np
import h5py, os

OMEGA, W0, LTP = 0.25, 0.05, 3
NT, NP = 64, 128

fn = sys.argv[1]
colats = [float(c) for c in sys.argv[2:]] or [60.0, 75.0, 90.0]
with h5py.File(fn, "r") as f:
    d = {k: f[k][()].reshape(205, NT, NP) for k in
         ("Er", "Eth", "Eph", "Br", "Bth", "Bph")}
    t = float(f["time"][()])
run = os.path.dirname(fn)
with h5py.File(os.path.join(run, "mesh.h5"), "r") as g:
    radii = g["radii"][()].astype(np.float64)
theta = (np.arange(NT) + 0.5) * np.pi / NT

print(f"# {fn.split('/')[-3]} t={t:.2f} t/P={t*OMEGA/2/np.pi:.2f}")
hdr = "colat: " + " ".join(f"{c:>8.0f}" for c in colats)
print(f"{'k':>3} {'r':>7} {'MT_om':>8} " +
      " ".join(f"{'om@'+str(int(c)):>8}" for c in colats))
for k in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40]:
    r = radii[k]
    row = []
    for c in colats:
        it = int(np.argmin(np.abs(theta - np.radians(c))))
        Er, Eth, Eph = d["Er"][k, it], d["Eth"][k, it], d["Eph"][k, it]
        Br, Bth, Bph = d["Br"][k, it], d["Bth"][k, it], d["Bph"][k, it]
        ExB_ph = Er * Bth - Eth * Br
        B2 = Br**2 + Bth**2 + Bph**2
        om = (ExB_ph / B2).mean() / (r * np.sin(theta[it]))
        row.append(om)
    print(f"{k:3d} {r:7.4f} {OMEGA - W0/r**LTP:8.4f} " +
          " ".join(f"{v:8.4f}" for v in row))
