#!/usr/bin/env python3
"""Measure the absorber reflection from sph dumps via the standing wave.

A reflected wave of relative amplitude rho leaves a spatial-period
lambda/2 imprint on the period-averaged wave intensity:

    W(r) = <Var_t E_orthonormal>(r) * r^2 = A * (1 + 2 rho cos(2 k r + phi))

Subtracting the time mean per pixel (the variance) removes the static
m=0 fields; converting the sph covariant components (E_th ~ r,
E_ph ~ r sin th) to orthonormal and multiplying by r^2 makes the
outgoing-wave contribution flat in r.  Fitting (A, rho, phi) over the
clean zone gives the reflected amplitude fraction rho directly and the
reflection phase phi (which localizes the reflector modulo lambda/2).

Used by the 2026-07-16 A2 absorber study to identify the layer-entrance
near-field mismatch rho ~ 1.7/(k r_in)^2 and the cavity-resonance
luminosity shifts delta_L ~ 2 rho cos(2 k d_in); see ROADMAP A2 item 2.

Usage: absorber_standing_wave.py <run_dir> <r_lo> <r_hi> [--omega 0.2]
"""
import argparse
import glob
import os

import h5py
import numpy as np
from scipy.optimize import curve_fit


def wave_energy_profile(run_dir, period):
    with h5py.File(os.path.join(run_dir, "sph_grid.h5"), "r") as f:
        N_theta = int(f["N_theta"][()])
        N_phi = int(f["N_phi"][()])
        theta = f["theta"][()].astype(np.float64)
        radii = f["radii"][()].astype(np.float64)
    files = [p for p in sorted(glob.glob(os.path.join(run_dir, "sph_*.h5")))
             if "grid" not in p]
    with h5py.File(files[-1], "r") as f:
        t_end = float(f["time"][()])
    use = []
    for p in files:
        with h5py.File(p, "r") as f:
            if float(f["time"][()]) > t_end - period + 1e-6:
                use.append(p)
    shp = (len(radii), N_theta, N_phi)
    w = np.sin(theta)[None, :, None]
    s2 = np.maximum(np.sin(theta)[None, :, None] ** 2, 1e-12)
    prof = np.zeros(len(radii))
    for name in ("Eth", "Eph"):
        arr = np.stack([h5py.File(p, "r")[name][()].reshape(shp)
                        .astype(np.float64) for p in use])
        var = arr.var(axis=0)
        if name == "Eph":
            var = var / s2
        prof += (var * w).sum(axis=(1, 2))
    return radii, prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("r_lo", type=float)
    ap.add_argument("r_hi", type=float)
    ap.add_argument("--omega", type=float, default=0.2)
    args = ap.parse_args()

    k = args.omega
    radii, prof = wave_energy_profile(args.run_dir, 2 * np.pi / args.omega)
    m = (radii >= args.r_lo) & (radii <= args.r_hi)
    r, y = radii[m], prof[m]

    def model(r, A, rho, phi):
        return A * (1 + 2 * rho * np.cos(2 * k * r + phi))

    popt, _ = curve_fit(model, r, y, p0=(y.mean(), 0.02, 0.0), maxfev=20000)
    A, rho, phi = popt
    resid = np.sqrt(np.mean((y - model(r, *popt)) ** 2)) / A
    print(f"{args.run_dir}: fit over r in [{args.r_lo}, {args.r_hi}]")
    print(f"  rho = {rho:+.4f} (reflected amplitude fraction), "
          f"phi = {phi % (2 * np.pi):.3f} rad, rms resid/A = {resid:.2%}")
    print(f"  W(r) profile:")
    for rr, pp in zip(radii, prof):
        if args.r_lo - 5 <= rr <= args.r_hi + 15:
            print(f"    {rr:8.2f} {pp:12.5e}")


if __name__ == "__main__":
    main()
