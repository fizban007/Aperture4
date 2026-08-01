#!/usr/bin/env python3
"""Meridional movie of the Kerr-Wald relaxation.

Colour   : H_phi, the toroidal auxiliary field, in an orthonormal frame.
Contours : poloidal field lines (contours of the flux function Psi).

Why H_phi is the right thing to watch.  In a stationary axisymmetric vacuum
state Ampere gives curl H = 0, so H = grad(chi); the phi-component of a
gradient vanishes under axisymmetry.  So H_phi -> 0 is an exact signature of
having reached the stationary state -- unlike B_phi, which stays finite
(toroidal/poloidal ~ 0.35 near the horizon for a = 0.998).

Constitutive relation (dec_field_solver_gr_ks_impl.hpp:229-287), H = alpha*B
- beta x D with a purely radial shift:

    H_phi = alpha * (g_rphi B^r + g_phiphi B^phi)  -  sqrt(g) beta^r D_theta / Sigma

prismatic_sph_output writes B^i CONTRAVARIANT (Br/Bth/Bph) and D_i COVARIANT
(Er/Eth/Eph) -- see prismatic_sph_output.cpp:293-300.  The sign is verified in
verify(): on the on-shell analytic state H_phi must vanish.

Usage:
    make_wald_movie.py RUNDIR [--out wald_L5.mp4] [--rmax 8] [--verify]
"""
import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

A_SPIN = 0.998


def ks_geometry(r, th, a=A_SPIN):
    """Kerr-Schild 3-metric pieces on an (r, theta) grid."""
    R = r[:, None]
    T = th[None, :]
    Sigma = R ** 2 + a ** 2 * np.cos(T) ** 2
    f = 2.0 * R / Sigma
    return dict(
        Sigma=Sigma, f=f,
        alpha=1.0 / np.sqrt(1.0 + f),
        beta_r=f / (1.0 + f),
        g_rphi=-a * np.sin(T) ** 2 * (1.0 + f),
        g_phiphi=(R ** 2 + a ** 2 + f * a ** 2 * np.sin(T) ** 2) * np.sin(T) ** 2,
        sqrt_g=np.sin(T) * np.sqrt(Sigma * (Sigma + 2.0 * R)),
        sin_t=np.sin(T), cos_t=np.cos(T), R=R,
    )


def read_slice(path, Nr, Nt, Np, ip):
    with h5py.File(path, "r") as f:
        out = {k: np.array(f[k]).reshape(Nr, Nt, Np)[:, :, ip]
               for k in ("Br", "Bth", "Bph", "Er", "Eth", "Eph")}
        out["time"] = float(np.array(f["time"]))
    return out


def h_phi_orthonormal(sl, geo):
    """Toroidal H in an orthonormal frame: H_phi / sqrt(g_phiphi)."""
    B_phi_cov = geo["g_rphi"] * sl["Br"] + geo["g_phiphi"] * sl["Bph"]
    H_phi_cov = (geo["alpha"] * B_phi_cov
                 - geo["sqrt_g"] * geo["beta_r"] * sl["Eth"] / geo["Sigma"])
    return H_phi_cov / np.sqrt(np.maximum(geo["g_phiphi"], 1e-300))


def flux_function(sl, geo, th):
    """Psi(r,theta) = 2 pi \\int_0^theta B^r sqrt(g) dtheta' -- poloidal field
    lines are its contours, and Psi(r, pi/2) is the polar-cap flux."""
    integrand = sl["Br"] * geo["sqrt_g"]
    dth = np.diff(th)
    seg = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * dth[None, :]
    return 2.0 * np.pi * np.concatenate(
        [np.zeros((integrand.shape[0], 1)), np.cumsum(seg, axis=1)], axis=1)


def verify(rundir, Nr, Nt, Np, ip, geo):
    """On the on-shell analytic state H_phi must vanish; report how well."""
    p = os.path.join(rundir, "sph_000000.h5")
    sl = read_slice(p, Nr, Nt, Np, ip)
    H = h_phi_orthonormal(sl, geo)
    lapse_only = (geo["alpha"] * (geo["g_rphi"] * sl["Br"]
                                 + geo["g_phiphi"] * sl["Bph"])
                  / np.sqrt(np.maximum(geo["g_phiphi"], 1e-300)))
    m = geo["R"][:, 0] > 1.2
    num = np.sqrt(np.mean(H[m, :] ** 2))
    den = np.sqrt(np.mean(lapse_only[m, :] ** 2))
    print(f"  verify: rms|H_phi| = {num:.4e}, rms|alpha B_phi| = {den:.4e}, "
          f"ratio = {num/den:.3e}")
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rmax", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--frames-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with h5py.File(os.path.join(args.rundir, "sph_grid.h5"), "r") as g:
        r = np.array(g["radii"], dtype=np.float64)
        th = np.array(g["theta"], dtype=np.float64)
        ph = np.array(g["phi"], dtype=np.float64)
    Nr, Nt, Np = len(r), len(th), len(ph)
    ip_r = int(np.argmin(np.abs(ph - 0.0)))
    ip_l = int(np.argmin(np.abs(ph - np.pi)))
    geo = ks_geometry(r, th)

    if args.verify:
        verify(args.rundir, Nr, Nt, Np, ip_r, geo)
        return

    # Filter BEFORE sorting: sph_grid.h5 also matches the glob and has no
    # step number, so the sort key would raise on it.
    files = [f for f in glob.glob(os.path.join(args.rundir, "sph_*.h5"))
             if re.search(r"sph_(\d+)\.h5$", f)]
    files.sort(key=lambda p: int(re.search(r"sph_(\d+)\.h5$", p).group(1)))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no sph_*.h5 in {args.rundir}")

    r_plus = 1.0 + np.sqrt(max(0.0, 1.0 - A_SPIN ** 2))
    # Cover the CORNERS of the square view, not just a disk of radius rmax:
    # the plot box spans |R|,|z| <= rmax, whose corners sit at rmax*sqrt(2).
    keep = r <= args.rmax * 1.45
    rk = r[keep]

    # Meridional coordinates for each half-plane.  Build from th directly --
    # geo["sin_t"]/["cos_t"] are (1, N_theta) broadcast rows, not (N_r, ...).
    Xr = rk[:, None] * np.sin(th)[None, :]
    Zr = rk[:, None] * np.cos(th)[None, :]
    Xl = -Xr

    # Fixed symmetric colour limits from the first frame, and fixed field-line
    # levels from the LAST frame (the relaxed state) so the lines are
    # comparable throughout and label the final configuration.
    sl0 = read_slice(files[0], Nr, Nt, Np, ip_r)
    H0 = h_phi_orthonormal(sl0, geo)[keep]
    vmax = float(np.percentile(np.abs(H0), 99.0))
    if vmax <= 0:
        vmax = 1.0
    slN = read_slice(files[-1], Nr, Nt, Np, ip_r)
    PsiN = flux_function(slN, geo, th)[keep]
    # Space the levels QUADRATICALLY in Psi.  A uniform B_z field has
    # Psi ~ R^2, so linearly spaced levels bunch up at large R and leave the
    # axis region empty; sqrt-spacing puts the lines at even intervals in R.
    levels = float(np.nanmax(PsiN)) * 0.99 * np.linspace(0.0, 1.0, 29) ** 2
    levels = levels[1:]

    outdir = os.path.join(args.rundir, "frames")
    os.makedirs(outdir, exist_ok=True)
    for old in glob.glob(os.path.join(outdir, "f_*.png")):
        os.remove(old)

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    print(f"  {len(files)} frames, colour limits +-{vmax:.3e}")

    # Pre-pass: rms|H_phi| vs time for the inset.  With a fixed colour scale
    # the late frames correctly wash out to white, so the inset is what
    # carries the quantitative tail of the decay.
    mask_r = r > 1.2
    times, rms = [], []
    for path in files:
        sl = read_slice(path, Nr, Nt, Np, ip_r)
        H = h_phi_orthonormal(sl, geo)
        times.append(sl["time"])
        rms.append(float(np.sqrt(np.mean(H[mask_r, :] ** 2))))
    times = np.array(times)
    rms = np.array(rms)
    floor = None
    ana_dir = args.rundir.rstrip("/") + "_ana"
    ana_f = os.path.join(ana_dir, "sph_000000.h5")
    if os.path.exists(ana_f):
        sl = read_slice(ana_f, Nr, Nt, Np, ip_r)
        floor = float(np.sqrt(np.mean(
            h_phi_orthonormal(sl, geo)[mask_r, :] ** 2)))
        print(f"  analytic on-shell floor rms|H_phi| = {floor:.3e}")

    for i, path in enumerate(files):
        fig = plt.figure(figsize=(7.4, 8.6), dpi=110)
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(2, 1, height_ratios=[4.4, 1.0], hspace=0.30,
                              left=0.11, right=0.86, top=0.90, bottom=0.08)
        ax = fig.add_subplot(gs[0])
        axt = fig.add_subplot(gs[1])
        for ip, X in ((ip_r, Xr), (ip_l, Xl)):
            sl = read_slice(path, Nr, Nt, Np, ip)
            H = h_phi_orthonormal(sl, geo)[keep]
            Psi = flux_function(sl, geo, th)[keep]
            Z = Zr
            ax.pcolormesh(X, Z, H, cmap="RdBu_r", norm=norm,
                          shading="gouraud", rasterized=True)
            ax.contour(X, Z, Psi, levels=levels, colors="#222222",
                       linewidths=0.75, alpha=0.85)
            t = sl["time"]

        ax.add_patch(plt.Circle((0, 0), r_plus, color="#111111", zorder=5))
        ax.add_patch(plt.Circle((0, 0), r_plus, fill=False, lw=1.0,
                                color="#666666", zorder=6))
        ax.set_xlim(-args.rmax, args.rmax)
        ax.set_ylim(-args.rmax, args.rmax)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$R\ [M]$")
        ax.set_ylabel(r"$z\ [M]$")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#999999")
        ax.tick_params(colors="#555555", labelsize=9)
        ax.set_title(f"Kerr–Wald relaxation, $a = {A_SPIN}$, L5\n"
                     f"$t = {t:7.1f}\\ M$", fontsize=12, color="#222222")
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02,
                          extend="both")
        cb.set_label(r"toroidal $\hat{H}_\phi$   (→ 0 when stationary)",
                     fontsize=10, color="#333333")
        cb.ax.tick_params(labelsize=8, colors="#555555")
        ax.text(0.02, 0.98, "black lines: poloidal field lines",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8.5, color="#444444")

        axt.plot(times, rms, color="#8c8c8c", lw=1.4)
        axt.plot(times[i], rms[i], "o", ms=6, color="#b2182b", zorder=5)
        if floor is not None:
            axt.axhline(floor, ls="--", lw=1.1, color="#2166ac")
            axt.text(0.995, floor, "analytic on-shell floor ",
                     transform=axt.get_yaxis_transform(), fontsize=7.5,
                     color="#2166ac", va="bottom", ha="right")
        axt.set_xlim(times[0], times[-1])
        axt.set_ylim(0.0, rms.max() * 1.08)
        axt.set_xlabel("$t\\ [M]$", fontsize=9, color="#555555")
        axt.set_ylabel(r"rms$|\hat{H}_\phi|$", fontsize=9, color="#555555")
        axt.tick_params(labelsize=8, colors="#666666")
        axt.grid(True, which="major", axis="y", lw=0.5, color="#e6e6e6")
        axt.set_axisbelow(True)
        for sp in ("top", "right"):
            axt.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            axt.spines[sp].set_color("#bbbbbb")

        fig.savefig(os.path.join(outdir, f"f_{i:04d}.png"),
                    facecolor="white")
        plt.close(fig)
        if i % 20 == 0:
            print(f"    frame {i}/{len(files)}  t={t:.1f}M")

    print(f"  frames in {outdir}")
    if args.frames_only:
        return
    out = args.out or os.path.join(args.rundir, "wald_relax.mp4")
    cmd = (f'ffmpeg -y -loglevel error -framerate {args.fps} '
           f'-i "{outdir}/f_%04d.png" -c:v libx264 -pix_fmt yuv420p '
           f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "{out}"')
    rc = os.system(cmd)
    print(("  wrote " + out) if rc == 0 else f"  ffmpeg failed (rc={rc})")


if __name__ == "__main__":
    main()
