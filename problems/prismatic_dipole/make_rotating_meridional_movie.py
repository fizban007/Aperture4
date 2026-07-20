#!/usr/bin/env python3
"""Co-rotating meridional movie from prismatic sph dumps.

For each sph snapshot, slices the meridional plane that CONTAINS the
instantaneous magnetic axis (phi_m = Omega * t, the rotating-dipole
convention m = Bp (sin a cos Wt, sin a sin Wt, cos a)): the right half
of each panel is the half-plane phi = phi_m, the left half is
phi = phi_m + pi.  In this frame an oblique rotator's structure —
current sheet, Y-point, polar-cap pattern — should appear quasi-steady
once the magnetosphere is established.

Panels:
  1. |J| sign(J_r) / J_GJ(r)         current density (sheet tracer)
  2. B_phihat r^2 / Bp               out-of-plane field (asinh scale)
  3. rho / rho_GJ(r)                 charge density (asinh scale)
  4. <gamma>                         weighted-mean Lorentz factor

Component conventions (see prismatic_sph_output.cpp): E, J covariant
coord-basis, B contravariant:
  J_orth = (Jr, Jth/r, Jph/(r sin th)),  B_phihat = Bph * r * sin th.

Usage:
  make_rotating_meridional_movie.py <data_dir> [--omega 0.25]
      [--bp 1000] [--alpha-deg 60] [--rmax 8] [--fps 8]
      [--out movies/<dirname>_corotating.mp4]
"""

import argparse
import glob
import os
import subprocess
import tempfile

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm

ap = argparse.ArgumentParser()
ap.add_argument("data_dir")
ap.add_argument("--omega", type=float, default=0.25)
ap.add_argument("--bp", type=float, default=1000.0)
ap.add_argument("--alpha-deg", type=float, default=60.0)
ap.add_argument("--rmax", type=float, default=8.0)
ap.add_argument("--fps", type=int, default=8)
ap.add_argument("--out", default=None)
args = ap.parse_args()

OMEGA, BP, RL = args.omega, args.bp, args.rmax
R_LC = 1.0 / OMEGA
ALPHA = np.radians(args.alpha_deg)
P = 2 * np.pi / OMEGA

out = args.out or os.path.join(
    "movies", os.path.basename(os.path.normpath(args.data_dir)) +
    "_corotating.mp4")
os.makedirs(os.path.dirname(out), exist_ok=True)

with h5py.File(os.path.join(args.data_dir, "sph_grid.h5")) as f:
    th, ph, radii = f["theta"][:], f["phi"][:], f["radii"][:]
N_th, N_ph = len(th), len(ph)
TH = th[1:-1]  # drop the pole rows (sentinel/singular columns)
r = radii[:, None]
sin_th = np.sin(TH)[None, :]

# Meridional mesh: right half x = +r sin th, left half x = -r sin th.
Rg, THg = np.meshgrid(radii, TH, indexing="ij")
Xr, Z = Rg * np.sin(THg), Rg * np.cos(THg)
Xl = -Xr

rho_gj = 2.0 * OMEGA * BP / radii[:, None] ** 3   # (N_r, 1) broadcast

files = sorted(f for f in glob.glob(os.path.join(args.data_dir, "sph_0*.h5"))
               if "grid" not in f)
if not files:
    raise SystemExit(f"no sph dumps under {args.data_dir}")


def slice_phi(a, phi_t):
    """Linear interpolation of a (N_r, N_th-2, N_ph) array at azimuth
    phi_t (periodic)."""
    x = (phi_t - ph[0]) / (2 * np.pi / N_ph)
    j0 = int(np.floor(x)) % N_ph
    j1 = (j0 + 1) % N_ph
    w = x - np.floor(x)
    return (1 - w) * a[:, :, j0] + w * a[:, :, j1]


def panel_fields(path):
    with h5py.File(path) as f:
        t = float(f["time"][()])
        step = int(f["step"][()])
        shp = (len(radii), N_th, N_ph)
        d = {k: f[k][:].reshape(shp)[:, 1:-1, :]
             for k in ["Br", "Bph", "Jr", "Jth", "Jph", "rho", "gamma_mean"]}
    phi_m = (OMEGA * t) % (2 * np.pi)
    halves = {}
    for side, phi_t in (("R", phi_m), ("L", (phi_m + np.pi) % (2 * np.pi))):
        Jr = slice_phi(d["Jr"], phi_t)
        Jth = slice_phi(d["Jth"], phi_t)
        Jph = slice_phi(d["Jph"], phi_t)
        Jmag = np.sqrt(Jr**2 + (Jth / r) ** 2 + (Jph / (r * sin_th)) ** 2)
        halves[side] = dict(
            j=np.sign(Jr) * Jmag / rho_gj,
            bphi=slice_phi(d["Bph"], phi_t) * r * sin_th * r**2 / BP,
            rho=slice_phi(d["rho"], phi_t) / rho_gj,
            gam=slice_phi(d["gamma_mean"], phi_t),
        )
    return t, step, halves


# Fixed color scales from a late frame (the structure is established
# well before the end; fixed norms keep the movie steady to the eye).
_, _, ref = panel_fields(files[-1])
ref_all = {k: np.concatenate([np.abs(ref["R"][k]), np.abs(ref["L"][k])])
           for k in ["j", "bphi", "rho", "gam"]}
vj = float(np.percentile(ref_all["j"], 99))
vb = float(np.percentile(ref_all["bphi"], 99.5))
vr = float(np.percentile(ref_all["rho"], 99))
vg = max(float(np.percentile(ref_all["gam"], 99.5)), 2.0)

specs = [
    ("j", r"$|J|\,\mathrm{sign}(J_r)\,/\,J_{\rm GJ}(r)$", "RdBu_r",
     AsinhNorm(linear_width=0.5 * vj, vmin=-3 * vj, vmax=3 * vj)),
    ("bphi", r"$B_{\hat\phi}\, r^2 / B_p$ (out of plane)", "PuOr_r",
     AsinhNorm(linear_width=0.08 * vb, vmin=-2 * vb, vmax=2 * vb)),
    ("rho", r"$\rho\,/\,\rho_{\rm GJ}(r)$", "RdBu_r",
     AsinhNorm(linear_width=0.5 * vr, vmin=-3 * vr, vmax=3 * vr)),
    ("gam", r"$\langle\gamma\rangle$", "inferno",
     matplotlib.colors.Normalize(vmin=1.0, vmax=vg)),
]

tmp = tempfile.mkdtemp(prefix="rotmov_")
for i, path in enumerate(files):
    t, step, h = panel_fields(path)
    fig, axs = plt.subplots(1, 4, figsize=(21, 6.2), constrained_layout=True)
    for ax, (key, title, cmap, norm) in zip(axs, specs):
        pcm = ax.pcolormesh(Xr, Z, h["R"][key], cmap=cmap, norm=norm,
                            shading="gouraud", rasterized=True)
        ax.pcolormesh(Xl, Z, h["L"][key], cmap=cmap, norm=norm,
                      shading="gouraud", rasterized=True)
        fig.colorbar(pcm, ax=ax, shrink=0.85)
        # Star, light cylinder, rotation axis, magnetic axis (fixed in
        # this co-rotating frame: at angle alpha in the right half).
        ax.fill(np.cos(np.linspace(0, 2 * np.pi, 100)),
                np.sin(np.linspace(0, 2 * np.pi, 100)), color="0.2")
        for s in (+1, -1):
            ax.axvline(s * R_LC, color="k", ls="--", lw=0.7, alpha=0.5)
        ax.plot([0, 0], [-RL, RL], color="0.4", ls=":", lw=0.8)
        ax.plot([-2.2 * np.sin(ALPHA), 2.2 * np.sin(ALPHA)],
                [-2.2 * np.cos(ALPHA), 2.2 * np.cos(ALPHA)],
                color="k", ls="-", lw=1.0, alpha=0.7)
        ax.set_xlim(-RL, RL)
        ax.set_ylim(-RL, RL)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
    fig.suptitle(
        rf"{os.path.basename(os.path.normpath(args.data_dir))}   "
        rf"co-rotating meridional plane ($\phi_m = \Omega t$, "
        rf"$\alpha = {args.alpha_deg:.0f}^\circ$)   step {step}   "
        rf"$t/P$ = {t / P:.3f}",
        fontsize=13)
    fig.savefig(os.path.join(tmp, f"f{i:04d}.png"), dpi=110)
    plt.close(fig)
    if i % 10 == 0:
        print(f"frame {i + 1}/{len(files)}")

subprocess.run(
    ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
     "-i", os.path.join(tmp, "f%04d.png"),
     "-c:v", "libx264", "-pix_fmt", "yuv420p",
     "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out],
    check=True)
print(f"movie: {out}  ({len(files)} frames @ {args.fps} fps)")
