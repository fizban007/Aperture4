#!/usr/bin/env python
"""Early-time seed structure: where does the residual E first grow?"""
import sys
import numpy as np
import h5py, os

OMEGA, W0, LTP = 0.25, 0.05, 3
NT, NP = 64, 128

def load(fn):
    with h5py.File(fn, "r") as f:
        d = {k: f[k][()].reshape(205, NT, NP) for k in
             ("Er", "Eth", "Eph", "Br", "Bth", "Bph")}
        t = float(f["time"][()])
    return d, t

run = sys.argv[1]
with h5py.File(os.path.join(run, "mesh.h5"), "r") as g:
    radii = g["radii"][()].astype(np.float64)
theta = (np.arange(NT) + 0.5) * np.pi / NT
sinth = np.sin(theta)[None, :, None]
r3 = radii[:, None, None]
om = OMEGA - W0 / r3**LTP
v = om * r3 * sinth

for step in sys.argv[2:]:
    d, t = load(os.path.join(run, f"sph_{int(step):06d}.h5"))
    dEr = d["Er"] - v * d["Bth"]
    dEth = d["Eth"] + v * d["Br"]
    dEph = d["Eph"]
    dmag = np.sqrt(dEr**2 + dEth**2 + dEph**2)
    print(f"\n### step {step}, t={t:.2f}, t/P={t*OMEGA/2/np.pi:.2f}")
    print("# rows: colat(deg); cols: shells k=1..12; entries phi-mean |dE|")
    ksel = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    print("colat " + " ".join(f"{'k'+str(k):>7}" for k in ksel))
    for it in range(0, NT, 3):
        print(f"{np.degrees(theta[it]):5.1f} " +
              " ".join(f"{dmag[k, it].mean():7.1f}" for k in ksel))
    # phi spectrum at the fastest-growing row, k=4
    k = 4
    it = int(np.argmax([dmag[k, i].mean() for i in range(NT)]))
    sp = np.abs(np.fft.rfft(dEth[k, it]))
    top = np.argsort(sp[1:])[::-1][:8] + 1
    print(f"peak row colat={np.degrees(theta[it]):.1f} k={k}; top m: " +
          ", ".join(f"m{m}={sp[m]:.3g}" for m in sorted(top)))
