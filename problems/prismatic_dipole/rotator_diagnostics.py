#!/usr/bin/env python3
"""A3 rotator scorecard diagnostics from sph dumps.

Usage: python rotator_diagnostics.py <data_dir> [step]

Panels (written to <data_dir>/diagnostics_<step>.png):
  1. rho meridional map (GJ pattern: negative over poles, positive
     near the equator inside the light cylinder for aligned rotators
     with Omega parallel to m).
  2. Plasma angular velocity Omega_pl = (E x B)_phi / (B^2 r sin th),
     phi-averaged meridional map, with the corotation target Omega and
     the light cylinder marked.
  3. |E.B|/B^2 meridional map (screening monitor).
  4. Radial profile of Omega_pl/Omega along theta = 55 deg (a closed
     field line footprint) and along the equator.

All sph components are coord-basis (E covariant, B contravariant):
  E_orth = (Er, Eth/r, Eph/(r sin th)),
  B_orth = (Br, Bth*r, Bph*r sin th).
"""

import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

data_dir = sys.argv[1] if len(sys.argv) > 1 else "Data_ns_rotator_L5_t1"
OMEGA = 0.2
R_LC = 1.0 / OMEGA


def load(step=None):
    import glob
    files = sorted(f for f in glob.glob(os.path.join(data_dir, "sph_0*.h5"))
                   if "grid" not in f)
    path = files[-1] if step is None else os.path.join(
        data_dir, f"sph_{step:06d}.h5")
    with h5py.File(os.path.join(data_dir, "sph_grid.h5")) as f:
        th, ph, radii = f["theta"][:], f["phi"][:], f["radii"][:]
    out = {}
    with h5py.File(path) as f:
        for k in ["Er", "Eth", "Eph", "Br", "Bth", "Bph", "rho"]:
            if k in f:
                out[k] = f[k][:].reshape(len(radii), len(th), len(ph))
        out["step"] = int(f["step"][()])
        out["time"] = float(f["time"][()])
    return th, ph, radii, out


def main():
    step = int(sys.argv[2]) if len(sys.argv) > 2 else None
    th, ph, radii, d = load(step)
    N_r, N_th = len(radii), len(th)
    TH = th[1:-1]  # drop poles
    R, THg = np.meshgrid(radii, TH)
    X, Z = R * np.sin(THg), R * np.cos(THg)
    sin_th = np.sin(TH)[None, :, None]
    r = radii[:, None, None]

    # Orthonormal fields (phi-average where needed)
    Er, Eth, Eph = d["Er"][:, 1:-1], d["Eth"][:, 1:-1], d["Eph"][:, 1:-1]
    Br, Bth, Bph = d["Br"][:, 1:-1], d["Bth"][:, 1:-1], d["Bph"][:, 1:-1]
    E_o = np.stack([Er, Eth / r, Eph / (r * sin_th)])
    B_o = np.stack([Br, Bth * r, Bph * r * sin_th])
    B2 = (B_o**2).sum(axis=0)

    # (E x B)_phi = E_r B_th... orthonormal: (ExB)_ph = E_r B_th - E_th B_r
    ExB_ph = E_o[0] * B_o[1] - E_o[1] * B_o[0]
    Omega_pl = ExB_ph / (B2 * r * sin_th + 1e-30)
    EdotB = (E_o * B_o).sum(axis=0) / (B2 + 1e-30)

    om_map = Omega_pl.mean(axis=2).T / OMEGA        # (N_th-2, N_r)
    eb_map = np.abs(EdotB).mean(axis=2).T
    rho_map = d["rho"][:, 1:-1, :].mean(axis=2).T if "rho" in d else None

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    rl = 8.0

    def style(ax, title):
        ax.plot(np.cos(np.linspace(0, 2*np.pi, 100)),
                np.sin(np.linspace(0, 2*np.pi, 100)), 'k-', lw=1)
        ax.axvline(R_LC, color='k', ls='--', lw=0.7, alpha=0.5)
        ax.set_xlim(0, rl); ax.set_ylim(-rl, rl)
        ax.set_aspect('equal'); ax.set_title(title, fontsize=11)

    if rho_map is not None:
        vm = np.percentile(np.abs(rho_map), 99) + 1e-30
        im = axes[0].pcolormesh(X, Z, rho_map, cmap='RdBu_r', vmin=-vm,
                                vmax=vm, shading='gouraud', rasterized=True)
        fig.colorbar(im, ax=axes[0], shrink=0.8)
        style(axes[0], r'$\langle\rho\rangle_\phi$')

    im = axes[1].pcolormesh(X, Z, om_map, cmap='viridis', vmin=0, vmax=1.2,
                            shading='gouraud', rasterized=True)
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    style(axes[1], r'$\Omega_{\rm pl}/\Omega$ ($E\times B$ rotation)')

    im = axes[2].pcolormesh(X, Z, eb_map, cmap='magma',
                            norm=matplotlib.colors.LogNorm(1e-6, 1e-1),
                            shading='gouraud', rasterized=True)
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    style(axes[2], r'$\langle|E\cdot B|\rangle_\phi / B^2$')

    for th_deg, c in [(55, 'C0'), (89, 'C1')]:
        it = np.argmin(np.abs(np.degrees(TH) - th_deg))
        axes[3].plot(radii, Omega_pl[:, it, :].mean(axis=1) / OMEGA, c,
                     label=rf'$\theta = {np.degrees(TH[it]):.0f}^\circ$')
    axes[3].axhline(1.0, color='k', ls=':', lw=1)
    axes[3].axvline(R_LC, color='k', ls='--', lw=0.7, alpha=0.5)
    axes[3].set_xlim(1, rl); axes[3].set_ylim(-0.2, 1.4)
    axes[3].set_xlabel(r'$r/r_\star$')
    axes[3].set_ylabel(r'$\Omega_{\rm pl}/\Omega$')
    axes[3].legend(); axes[3].set_title('corotation profile', fontsize=11)

    fig.suptitle(f'{data_dir}   step {d["step"]}   t = {d["time"]:.2f} '
                 f'(t/P = {d["time"] * OMEGA / (2*np.pi):.3f})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(data_dir, f'diagnostics_{d["step"]:06d}.png')
    fig.savefig(out, dpi=130)
    print(f'saved {out}')


if __name__ == "__main__":
    main()
