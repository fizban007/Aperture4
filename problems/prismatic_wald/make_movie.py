#!/usr/bin/env python3
"""
Render a movie of the prismatic Wald run on the meridional plane.

Each frame shows:
  - streamlines of the poloidal magnetic field (B_r, B_theta)
    averaged over phi, on a Cartesian (R, Z) grid
  - color: B_phi (axisymmetric-averaged)

Requires: Data_<tag>/ directory with sph_grid.h5 and sph_XXXXXX.h5 files.
"""
import argparse, os, glob
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_grid(data_dir):
    with h5py.File(f"{data_dir}/sph_grid.h5", "r") as f:
        Nth = int(f["N_theta"][()])
        Nph = int(f["N_phi"][()])
        Nr = int(f["N_r"][()])
        theta = np.array(f["theta"])
        phi = np.array(f["phi"])
        radii = np.array(f["radii"])
    return Nth, Nph, Nr, theta, phi, radii


def load_frame(path, Nr, Nth, Nph):
    with h5py.File(path, "r") as f:
        t = float(f["time"][()])
        Br = np.array(f["Br"]).reshape(Nr + 1, Nth, Nph)
        Bth = np.array(f["Bth"]).reshape(Nr + 1, Nth, Nph)
        Bph = np.array(f["Bph"]).reshape(Nr + 1, Nth, Nph)
    return t, Br, Bth, Bph


def render_frame(outpath, t, radii, theta, Br_m, Bth_m, Bph_m, rmax, Ngrid,
                 bph_cmax, a):
    # Average over phi: passed in already as ..._m (meridional slice)
    # Poloidal field in Cartesian cyl:  B_R = Br sinθ + Bθ cosθ,
    #                                   B_Z = Br cosθ − Bθ sinθ
    sth = np.sin(theta)[None, :]
    cth = np.cos(theta)[None, :]
    B_R = Br_m * sth + Bth_m * cth   # (Nr+1, Nth)
    B_Z = Br_m * cth - Bth_m * sth

    # Sample points in spherical: R_sph = r sinθ, Z_sph = r cosθ
    rr = radii[:, None] * np.ones_like(theta)[None, :]
    R_sph = rr * sth
    Z_sph = rr * cth

    # Regular Cartesian (R, Z) grid in the right half-plane; mirror to left.
    Rgrid = np.linspace(0, rmax, Ngrid)
    Zgrid = np.linspace(-rmax, rmax, 2 * Ngrid - 1)
    Rm, Zm = np.meshgrid(Rgrid, Zgrid, indexing="xy")

    # Build spherical (r, theta) for each (R, Z)
    r_from_rz = np.sqrt(Rm**2 + Zm**2)
    th_from_rz = np.arctan2(Rm, Zm)

    # Interpolate with RegularGridInterpolator on (r, theta) domain.
    # Points outside radii[0] .. radii[-1] are masked.
    inside = (r_from_rz >= radii[0]) & (r_from_rz <= radii[-1])

    def interp(fld):
        rgi = RegularGridInterpolator(
            (radii, theta), fld, bounds_error=False, fill_value=np.nan
        )
        pts = np.stack([r_from_rz, th_from_rz], axis=-1)
        return rgi(pts)

    BR_xy = interp(B_R)
    BZ_xy = interp(B_Z)
    Bph_xy = interp(Bph_m)

    # Horizon for a=0.998 KS
    r_plus = 1.0 + np.sqrt(max(0.0, 1.0 - a * a))
    r_minus = 1.0 - np.sqrt(max(0.0, 1.0 - a * a))

    fig, ax = plt.subplots(figsize=(7, 10))
    im = ax.pcolormesh(
        Rm, Zm, Bph_xy, cmap="RdBu_r", vmin=-bph_cmax, vmax=bph_cmax,
        shading="auto", rasterized=True
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r"$B_\phi$ (orthonormal)")

    # Streamlines.  Scale linewidth by magnitude so field lines are visible.
    mag_pol = np.sqrt(BR_xy**2 + BZ_xy**2)
    lw = 0.5 + 2.0 * (mag_pol / (np.nanmax(mag_pol) + 1e-12))
    try:
        ax.streamplot(
            Rm, Zm, BR_xy, BZ_xy, color="k", linewidth=lw, density=1.4,
            arrowsize=0.8,
        )
    except Exception:
        pass

    # Horizon ring (projected onto meridional half-plane = circle of radius r_+)
    angs = np.linspace(-np.pi / 2, np.pi / 2, 200)
    ax.fill(r_plus * np.cos(angs), r_plus * np.sin(angs) * 0 +
            r_plus * np.sin(angs), color="0.35", alpha=0.6, zorder=5)
    ax.plot(r_plus * np.cos(angs), r_plus * np.sin(angs), color="k", lw=0.8,
            zorder=6)
    if r_minus > 0:
        ax.plot(r_minus * np.cos(angs), r_minus * np.sin(angs), color="k",
                lw=0.5, ls="--", zorder=6)

    ax.set_xlim(0, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R / M$")
    ax.set_ylabel(r"$Z / M$")
    ax.set_title(f"Prismatic Wald — a={a} — t = {t:6.2f} M")

    fig.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data_L5", help="data directory")
    ap.add_argument("--frames", default="frames", help="PNG output dir")
    ap.add_argument("--movie", default="wald_L5.mp4", help="output MP4 path")
    ap.add_argument("--rmax", type=float, default=20.0)
    ap.add_argument("--Ngrid", type=int, default=200)
    ap.add_argument("--a", type=float, default=0.998)
    ap.add_argument("--bph_cmax", type=float, default=0.2)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--skip", type=int, default=1,
                    help="render every N-th frame")
    args = ap.parse_args()

    os.makedirs(args.frames, exist_ok=True)
    Nth, Nph, Nr, theta, phi, radii = load_grid(args.data)

    files = sorted(glob.glob(f"{args.data}/sph_*.h5"))
    # Exclude the grid file if glob caught it
    files = [f for f in files if "sph_grid" not in f]
    files = files[:: args.skip]

    for i, fn in enumerate(files):
        t, Br, Bth, Bph = load_frame(fn, Nr, Nth, Nph)
        if not np.all(np.isfinite(Br)) or not np.all(np.isfinite(Bth)):
            print(f"skip {fn}: non-finite")
            continue
        # phi average
        Br_m = np.nanmean(Br, axis=-1)
        Bth_m = np.nanmean(Bth, axis=-1)
        Bph_m = np.nanmean(Bph, axis=-1)
        out = os.path.join(args.frames, f"frame_{i:04d}.png")
        render_frame(out, t, radii, theta, Br_m, Bth_m, Bph_m, args.rmax,
                     args.Ngrid, args.bph_cmax, args.a)
        if i % 10 == 0:
            print(f"  frame {i+1}/{len(files)}  t={t:.2f}")

    # Encode
    pattern = os.path.join(args.frames, "frame_%04d.png")
    cmd = (f"ffmpeg -y -framerate {args.fps} -i {pattern} -c:v libx264 "
           f"-pix_fmt yuv420p -crf 20 {args.movie}")
    print("encoding:", cmd)
    os.system(cmd)
    print("done:", args.movie)


if __name__ == "__main__":
    main()
