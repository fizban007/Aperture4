#!/usr/bin/env python
"""Time history of mean |E_e| on shell-tangential h-edges, colat 10-20 deg."""
import sys, glob
import numpy as np
import h5py

run = sys.argv[1]
ks = [1, 3, 5, 10, 20, 40]
with h5py.File(f"{run}/output/mesh.h5", "r") as g:
    V = np.stack([g["sphere_vx"][()], g["sphere_vy"][()],
                  g["sphere_vz"][()]], -1).astype(np.float64)
    e0 = g["sphere_edge_v0"][()]; e1 = g["sphere_edge_v1"][()]
    radii = g["radii"][()]; NEs = int(g["N_edge_s"][()])
mid = V[e0] + V[e1]
mid /= np.linalg.norm(mid, axis=1, keepdims=True)
colat = np.degrees(np.arccos(np.clip(mid[:, 2], -1, 1)))
sel = ((colat > 10) & (colat < 20)) | ((colat > 160) & (colat < 170))
print("#", " ".join(f"r{k}={radii[k]:.3f}" for k in ks))
print(f"{'t/P':>6} " + " ".join(f"{'k='+str(k):>8}" for k in ks))
for fn in sorted(glob.glob(f"{run}/output/step_*.h5")):
    with h5py.File(fn, "r") as f:
        E_e = f["E_e"]
        t = float(f["time"][()])
        vals = []
        for k in ks:
            Ek = E_e[k * NEs:(k + 1) * NEs]
            vals.append(np.abs(Ek[sel]).mean())
    print(f"{t*0.25/2/np.pi:6.2f} " + " ".join(f"{v:8.3g}" for v in vals))
