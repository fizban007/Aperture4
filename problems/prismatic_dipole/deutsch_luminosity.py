#!/usr/bin/env python3
"""A2.1 Poynting-luminosity diagnostic from prismatic_sph_output grids.

Integrates the radial Poynting flux over every output shell:

    L(r_k) = r_k^2 * Int [ sin^2(th) E_th B^ph  -  E_ph B^th ] dth dph

using the sph_output storage convention (E covariant coord-basis,
B contravariant coord-basis; see prismatic_sph_output.cpp).  In flat
space and orthonormal components this is Int (E x B)·rhat r^2 dOmega.

Health checks this provides:
  - L(r) should be flat between the light cylinder and the damping-layer
    onset (wave-zone energy conservation; ripples = reflections);
  - for the Deutsch run, L should match (8*pi/3) mu^2 Omega^4 sin^2(alpha)
    to leading order in R_*/R_LC.  The 8*pi/3 (not 2/3) is the code's
    rationalized units: S = E x B with no 1/(4*pi), and the analytic IC
    is B ~ [3n(n.m)-m]/r^3 with mu = Bp.

Usage: deutsch_luminosity.py <run_dir> [--omega 0.2] [--bp 1.0]
       [--alpha-deg 60] [--r-probe 7.0]
"""

import argparse
import glob
import os
import re

import h5py
import numpy as np


def shell_luminosity(f, N_theta, N_phi, N_shells, theta, radii):
    """L(r_k) for one sph snapshot file handle."""
    shp = (N_shells, N_theta, N_phi)
    Eth = f["Eth"][()].reshape(shp).astype(np.float64)
    Eph = f["Eph"][()].reshape(shp).astype(np.float64)
    Bth = f["Bth"][()].reshape(shp).astype(np.float64)
    Bph = f["Bph"][()].reshape(shp).astype(np.float64)
    integrand = (np.sin(theta)[None, :, None] ** 2) * Eth * Bph - Eph * Bth
    dphi = 2 * np.pi / N_phi
    az = integrand.sum(axis=2) * dphi                    # (N_shells, N_theta)
    L = radii**2 * np.trapezoid(az, theta, axis=1)       # (N_shells,)
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--omega", type=float, default=0.2)
    ap.add_argument("--bp", type=float, default=1.0)
    ap.add_argument("--alpha-deg", type=float, default=60.0)
    ap.add_argument("--r-probe", type=float, default=7.0)
    args = ap.parse_args()

    with h5py.File(os.path.join(args.run_dir, "sph_grid.h5"), "r") as f:
        N_theta = int(f["N_theta"][()])
        N_phi = int(f["N_phi"][()])
        theta = f["theta"][()].astype(np.float64)
        radii = f["radii"][()].astype(np.float64)
    N_shells = len(radii)
    k_probe = int(np.argmin(np.abs(radii - args.r_probe)))

    alpha = np.deg2rad(args.alpha_deg)
    L_dipole = (8.0 * np.pi / 3.0) * args.bp**2 * args.omega**4 * np.sin(alpha) ** 2
    r_lc = 1.0 / args.omega

    files = sorted(glob.glob(os.path.join(args.run_dir, "sph_*.h5")))
    files = [p for p in files if "grid" not in p]
    print(f"# {args.run_dir}: {len(files)} snapshots; "
          f"R_LC = {r_lc:.2f}, L_dipole = {L_dipole:.4e}")
    print(f"# {'time':>9} {'L(r_probe)':>12} {'L/L_dip':>9}")

    L_last = None
    for path in files:
        with h5py.File(path, "r") as f:
            t = float(f["time"][()])
            L = shell_luminosity(f, N_theta, N_phi, N_shells, theta, radii)
        L_last = (t, L)
        print(f"  {t:9.3f} {L[k_probe]:12.4e} {L[k_probe]/L_dipole:9.4f}")

    # radial profile of the last snapshot: wave-zone flatness check
    t, L = L_last
    print(f"# L(r) at t = {t:.3f}:")
    print(f"# {'r':>8} {'L':>12} {'L/L_dip':>9}")
    for k in range(0, N_shells, max(1, N_shells // 24)):
        print(f"  {radii[k]:8.3f} {L[k]:12.4e} {L[k]/L_dipole:9.4f}")
    wave = (radii > r_lc) & (radii < 0.9 * radii[-1])
    if wave.sum() > 2:
        Lw = L[wave]
        print(f"# wave-zone flatness (R_LC < r < 0.9 r_max): "
              f"mean {Lw.mean():.4e}, spread {(Lw.max()-Lw.min())/Lw.mean():.2%}")


if __name__ == "__main__":
    main()
