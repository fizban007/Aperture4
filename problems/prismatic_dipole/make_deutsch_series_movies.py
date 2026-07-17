#!/usr/bin/env python3
"""Movies for the L6 vacuum Deutsch rotator obliquity series.

Per obliquity: meridional Br*r^2 slice + equatorial (down-the-axis)
Br*r^2 slice + wave-zone shell map.  Combined movie: 2x4 grid of
meridional (top) and equatorial (bottom) slices for all four angles.

All movies span exactly one rotation period: the Deutsch IC + BC make
the solution periodic, so the last frame (t = P) is dropped and the
movie loops seamlessly.

The slices use an arcsinh color normalization (linear inside
+/-ASINH_WIDTH, logarithmic beyond): the near-star dipole Br*r^2 ~ 2/r
would otherwise dominate a linear scale and hide the wave-zone
stripes (amplitude ~0.2-0.4).

The displayed region is r <= R_LIM = 12: with the causally clean
protocol (r_max = 45, no absorber) the outer boundary contaminates only
r > r_max - P = 13.6 by the end of the period.
"""

import glob
import os

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm

ANGLES = ["00", "30", "60", "90"]
DATA_DIR = "Data_deutsch_L6_movie_a{}"
R_LIM = 12.0          # display radius (keep < 13.6, see module docstring)
R_LC = 5.0            # light cylinder, c/Omega with Omega = 0.2
R_SHELL = 10.0        # shell-map radius (wave zone, kr = 2)
FPS = 20
DPI = 130
STEPS_PER_PERIOD = 6400
VMAX = 2.0            # full range of Br*r^2 (2 Bp at the polar surface)
ASINH_WIDTH = 0.25    # linear width of the arcsinh color scale

FFMPEG_ARGS = dict(writer='ffmpeg', fps=FPS, dpi=DPI,
                   extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])


def make_norm():
    return AsinhNorm(linear_width=ASINH_WIDTH, vmin=-VMAX, vmax=VMAX)


CBAR_TICKS = [-2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2]



def add_slice_cbar(fig, im, ax, shrink=0.85):
    cb = fig.colorbar(im, ax=ax, shrink=shrink, label=r'$B_r\, r^2$')
    cb.set_ticks(CBAR_TICKS)
    cb.set_ticklabels([f'{t:g}' for t in CBAR_TICKS])
    return cb

def load_grid(data_dir):
    with h5py.File(os.path.join(data_dir, "sph_grid.h5"), "r") as f:
        theta = f["theta"][:]
        phi = f["phi"][:]
        radii = f["radii"][:]
    return theta, phi, radii


def snapshot_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "sph_0*.h5")))
    files = [f for f in files if "grid" not in f]
    # Drop the t = P frame: identical to t = 0 up to solver error, and
    # dropping it makes the loop seamless.
    return files[:-1]


def read_br(path, n_shell, n_th, n_ph):
    with h5py.File(path, "r") as f:
        br = f["Br"][:].reshape(n_shell, n_th, n_ph)
        step = int(f["step"][()])
    # The sph output writes exact zeros on the theta = 0, pi rows (the
    # sqrt(gamma) = r^2 sin(theta) clamp at the coordinate singularity).
    # Br is single-valued at the poles; patch each pole row with the
    # phi-average of its neighbor so gouraud shading doesn't show a
    # white streak along the rotation axis.
    br[:, 0, :] = br[:, 1, :].mean(axis=-1, keepdims=True)
    br[:, -1, :] = br[:, -2, :].mean(axis=-1, keepdims=True)
    return br, step


class SliceGeometry:
    """Meridional / equatorial / shell-map geometry for r <= R_LIM."""

    def __init__(self, theta, phi, radii):
        self.n_shell, self.n_th, self.n_ph = len(radii), len(theta), len(phi)
        self.k_max = int(np.searchsorted(radii, R_LIM * 1.05)) + 1
        self.k_shell = int(np.searchsorted(radii, R_SHELL))
        r_cut = radii[:self.k_max]

        R, TH = np.meshgrid(r_cut, theta)
        self.X = R * np.sin(TH)
        self.Z = R * np.cos(TH)
        self.R2 = R**2

        # Equatorial plane: no theta node sits exactly on the equator
        # (theta = pi*i/63), so average the two central rows.  Close the
        # phi loop by appending the phi = 2*pi column.
        self.i_eq = (self.n_th // 2 - 1, self.n_th // 2)
        phi_closed = np.append(phi, 2 * np.pi)
        PHI, Req = np.meshgrid(phi_closed, r_cut)
        self.Xe = Req * np.cos(PHI)
        self.Ye = Req * np.sin(PHI)
        self.R2e = Req**2

        self.PHI_deg, self.TH_deg = np.meshgrid(np.degrees(phi),
                                                np.degrees(theta))
        self.r_shell_actual = radii[self.k_shell]

    def meridional(self, br, ip):
        """Br * r^2 on the phi = phi_ip half-plane, (n_th, k_max)."""
        return br[:self.k_max, :, ip].T * self.R2

    def equatorial(self, br):
        """Br * r^2 on the z = 0 plane, (k_max, n_ph + 1)."""
        eq = 0.5 * (br[:self.k_max, self.i_eq[0], :] +
                    br[:self.k_max, self.i_eq[1], :])
        return np.concatenate([eq, eq[:, :1]], axis=1) * self.R2e

    def shell(self, br):
        return br[self.k_shell, :, :]


def draw_star(ax, lc_style='meridional'):
    tc = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(tc), np.sin(tc), 'k-', lw=1)
    if lc_style == 'meridional':
        for x in (-R_LC, R_LC):
            ax.axvline(x, color='k', ls='--', lw=0.7, alpha=0.5)
    else:  # equatorial: the light cylinder is a circle
        ax.plot(R_LC * np.cos(tc), R_LC * np.sin(tc), 'k--', lw=0.7,
                alpha=0.5)


def style_slice_ax(ax, xlabel, ylabel):
    ax.set_xlim(-R_LIM, R_LIM)
    ax.set_ylim(-R_LIM, R_LIM)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)


def make_single_movie(tag, geom, files, out_path):
    n_frames = len(files)
    br0, _ = read_br(files[0], geom.n_shell, geom.n_th, geom.n_ph)

    # Fixed shell-map scale from the mid-simulation frame (a per-frame
    # scale flickers).
    br_mid, _ = read_br(files[n_frames // 2], geom.n_shell, geom.n_th,
                        geom.n_ph)
    vmax_sh = max(np.percentile(np.abs(geom.shell(br_mid)), 99), 1e-10)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8),
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})

    im_mr = axes[0].pcolormesh(geom.X, geom.Z, geom.meridional(br0, 0),
                               cmap='RdBu_r', norm=make_norm(),
                               shading='gouraud', rasterized=True)
    im_ml = axes[0].pcolormesh(-geom.X, geom.Z,
                               geom.meridional(br0, geom.n_ph // 2),
                               cmap='RdBu_r', norm=make_norm(),
                               shading='gouraud', rasterized=True)
    draw_star(axes[0], 'meridional')
    style_slice_ax(axes[0], r'$x / r_\star$', r'$z / r_\star$')
    axes[0].set_title(r'meridional slice ($\phi = 0,\ \pi$)', fontsize=12)
    add_slice_cbar(fig, im_mr, axes[0])

    im_eq = axes[1].pcolormesh(geom.Xe, geom.Ye, geom.equatorial(br0),
                               cmap='RdBu_r', norm=make_norm(),
                               shading='gouraud', rasterized=True)
    draw_star(axes[1], 'equatorial')
    style_slice_ax(axes[1], r'$x / r_\star$', r'$y / r_\star$')
    axes[1].set_title(r'equatorial slice ($z = 0$)', fontsize=12)
    add_slice_cbar(fig, im_eq, axes[1])

    im_sh = axes[2].pcolormesh(geom.PHI_deg, geom.TH_deg, geom.shell(br0),
                               cmap='RdBu_r', vmin=-vmax_sh, vmax=vmax_sh,
                               shading='gouraud', rasterized=True)
    axes[2].set_xlabel(r'$\phi$ (deg)', fontsize=11)
    axes[2].set_ylabel(r'$\theta$ (deg)', fontsize=11)
    axes[2].invert_yaxis()
    axes[2].set_title(rf'$B_r$ at $r = {geom.r_shell_actual:.1f}\,r_\star$',
                      fontsize=12)
    fig.colorbar(im_sh, ax=axes[2], shrink=0.85, label=r'$B_r$')

    title = fig.suptitle('', fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    def update(i):
        br, step = read_br(files[i], geom.n_shell, geom.n_th, geom.n_ph)
        im_mr.set_array(geom.meridional(br, 0).ravel())
        im_ml.set_array(geom.meridional(br, geom.n_ph // 2).ravel())
        im_eq.set_array(geom.equatorial(br).ravel())
        im_sh.set_array(geom.shell(br).ravel())
        title.set_text(
            rf'Vacuum Deutsch rotator, $L=6$, $\alpha = {int(tag)}^\circ$'
            rf'   —   $t/P = {step / STEPS_PER_PERIOD:.3f}$')
        print(f"  [{tag}] frame {i + 1}/{n_frames}", end='\r')
        return [im_mr, im_ml, im_eq, im_sh, title]

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(out_path, **FFMPEG_ARGS)
    plt.close(fig)
    print(f"\n  saved {out_path}")


def make_combined_movie(geoms, file_lists, out_path):
    n_frames = min(len(f) for f in file_lists.values())
    fig, axes = plt.subplots(2, 4, figsize=(21, 11))
    ims = {}
    for col, tag in enumerate(ANGLES):
        geom = geoms[tag]
        br0, _ = read_br(file_lists[tag][0], geom.n_shell, geom.n_th,
                         geom.n_ph)
        ax_m, ax_e = axes[0, col], axes[1, col]
        ims[tag] = (
            ax_m.pcolormesh(geom.X, geom.Z, geom.meridional(br0, 0),
                            cmap='RdBu_r', norm=make_norm(),
                            shading='gouraud', rasterized=True),
            ax_m.pcolormesh(-geom.X, geom.Z,
                            geom.meridional(br0, geom.n_ph // 2),
                            cmap='RdBu_r', norm=make_norm(),
                            shading='gouraud', rasterized=True),
            ax_e.pcolormesh(geom.Xe, geom.Ye, geom.equatorial(br0),
                            cmap='RdBu_r', norm=make_norm(),
                            shading='gouraud', rasterized=True),
        )
        draw_star(ax_m, 'meridional')
        draw_star(ax_e, 'equatorial')
        style_slice_ax(ax_m, '', r'$z / r_\star$' if col == 0 else '')
        style_slice_ax(ax_e, r'$x / r_\star$',
                       r'$y / r_\star$' if col == 0 else '')
        ax_m.set_title(rf'$\alpha = {int(tag)}^\circ$', fontsize=14)

    add_slice_cbar(fig, ims[ANGLES[-1]][0], axes, shrink=0.75)
    title = fig.suptitle('', fontsize=16, y=0.98)

    def update(i):
        artists = []
        for tag in ANGLES:
            geom = geoms[tag]
            br, step = read_br(file_lists[tag][i], geom.n_shell, geom.n_th,
                               geom.n_ph)
            ims[tag][0].set_array(geom.meridional(br, 0).ravel())
            ims[tag][1].set_array(
                geom.meridional(br, geom.n_ph // 2).ravel())
            ims[tag][2].set_array(geom.equatorial(br).ravel())
            artists.extend(ims[tag])
        title.set_text(
            r'Vacuum Deutsch rotator, $L=6$ — $B_r\, r^2$, meridional (top)'
            r' and equatorial (bottom)'
            rf'   —   $t/P = {step / STEPS_PER_PERIOD:.3f}$')
        print(f"  [2x4] frame {i + 1}/{n_frames}", end='\r')
        artists.append(title)
        return artists

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(out_path, **FFMPEG_ARGS)
    plt.close(fig)
    print(f"\n  saved {out_path}")


def main():
    geoms, file_lists = {}, {}
    for tag in ANGLES:
        data_dir = DATA_DIR.format(tag)
        theta, phi, radii = load_grid(data_dir)
        geoms[tag] = SliceGeometry(theta, phi, radii)
        file_lists[tag] = snapshot_files(data_dir)
        print(f"alpha={tag}: {len(file_lists[tag])} frames in {data_dir}")

    os.makedirs("movies", exist_ok=True)
    for tag in ANGLES:
        make_single_movie(tag, geoms[tag], file_lists[tag],
                          f"movies/deutsch_L6_a{tag}.mp4")
    make_combined_movie(geoms, file_lists,
                        "movies/deutsch_L6_series_2x4.mp4")


if __name__ == "__main__":
    main()
