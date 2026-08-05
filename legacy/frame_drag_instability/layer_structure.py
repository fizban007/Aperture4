#!/usr/bin/env python
"""Spatial structure of the near-surface E layer from a sph_*.h5 file.

Residual dE = E - E_corot(Omega - w_lt(r), measured B); report by shell:
|dE_par|, |dE_perp|, and at a chosen shell the colat profile and phi
m-spectrum of the residual.
"""
import sys
import numpy as np
import h5py

OMEGA = 0.25
W0 = 0.05
LTP = 3

fn = sys.argv[1]
NT, NP = 64, 128
with h5py.File(fn, "r") as f:
    d = {k: f[k][()].reshape(205, NT, NP) for k in
         ("Er", "Eth", "Eph", "Br", "Bth", "Bph", "Jr", "rho")}
    t = float(f["time"][()])
import glob, os
run = os.path.dirname(fn)
with h5py.File(os.path.join(run, "mesh.h5"), "r") as g:
    radii = g["radii"][()].astype(np.float64)

theta = (np.arange(NT) + 0.5) * np.pi / NT
sinth = np.sin(theta)[None, :, None]
r3 = radii[:, None, None]

om = OMEGA - W0 / r3**LTP
v = om * r3 * sinth                      # v_phi of MT rotation
Ec_r = v * d["Bth"]
Ec_th = -v * d["Br"]
dEr = d["Er"] - Ec_r
dEth = d["Eth"] - Ec_th
dEph = d["Eph"]

Bmag = np.sqrt(d["Br"]**2 + d["Bth"]**2 + d["Bph"]**2) + 1e-30
dpar = (dEr * d["Br"] + dEth * d["Bth"] + dEph * d["Bph"]) / Bmag
dmag2 = dEr**2 + dEth**2 + dEph**2
dperp = np.sqrt(np.maximum(dmag2 - dpar**2, 0))

band = (theta > np.radians(10)) & (theta < np.radians(25))
print(f"# t={t:.2f}; colat band 10-25 deg, means over band x phi")
print(f"{'k':>3} {'r':>7} {'|dE_par|':>10} {'|dE_perp|':>10} {'|E_corot|':>10} {'E/B':>8}")
for k in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]:
    ec = np.sqrt(Ec_r[k]**2 + Ec_th[k]**2)[band].mean()
    print(f"{k:3d} {radii[k]:7.4f} {np.abs(dpar[k][band]).mean():10.4g} "
          f"{dperp[k][band].mean():10.4g} {ec:10.4g} "
          f"{(np.sqrt(dmag2[k])/Bmag[k])[band].mean():8.4f}")

# colat profile at k=4
k = 4
print(f"\n# colat profile at k={k} (r={radii[k]:.3f}): phi-mean |dE_perp|, |dE_par|, rho")
for it in range(0, NT, 2):
    print(f"{np.degrees(theta[it]):6.1f} {dperp[k, it].mean():10.4g} "
          f"{np.abs(dpar[k, it]).mean():10.4g} {d['rho'][k, it].mean():10.4g}")

# phi m-spectrum of dEth at k=4, at colat rows inside band
print(f"\n# phi m-spectrum |FFT(dEth)| at k={k}, colat 14/17/20 deg rows")
for it in [np.argmin(np.abs(theta - np.radians(c))) for c in (14, 17, 20)]:
    sp = np.abs(np.fft.rfft(dEth[k, it]))
    top = np.argsort(sp[1:])[::-1][:6] + 1
    print(f"colat {np.degrees(theta[it]):5.1f}: m0={sp[0]:.3g}; top m: " +
          ", ".join(f"m{m}={sp[m]:.3g}" for m in sorted(top)))
