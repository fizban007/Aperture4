#!/usr/bin/env python3
"""Generate a movie from pre-gridded spherical output (sph_NNNNNN.h5)."""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os
import sys

data_dir = sys.argv[1] if len(sys.argv) > 1 else "Data_L5_sph"
r_lim = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0

# ---- Load grid ----
with h5py.File(os.path.join(data_dir, "sph_grid.h5"), "r") as f:
    theta = f["theta"][:]
    phi = f["phi"][:]
    radii = f["radii"][:]
    N_th, N_ph, N_r = len(theta), len(phi), len(radii)
print(f"Grid: {N_th}x{N_ph}x{N_r}, radii [{radii[0]:.2f}, {radii[-1]:.2f}]")

# Meridional slice coordinates: right half (phi=0), left half (phi=pi)
R, TH = np.meshgrid(radii, theta)
X = R * np.sin(TH)
Z = R * np.cos(TH)

# Find snapshots
snap_files = sorted(glob.glob(os.path.join(data_dir, "sph_*.h5")))
snap_files = [f for f in snap_files if "grid" not in f]
print(f"Found {len(snap_files)} snapshots")

# Determine color scale from mid-simulation
with h5py.File(snap_files[len(snap_files)//2], "r") as f:
    Br = f["Br"][:].reshape(N_r, N_th, N_ph)
Br_slice = Br[:, :, 0].T * R**2
vmax = np.percentile(np.abs(Br_slice), 97)
print(f"Color scale: +/-{vmax:.4f}")

# ---- Set up figure ----
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5),
                         gridspec_kw={"width_ratios": [1, 1.3]})

# Left: full meridional slice (phi=0 right, phi=pi left), gouraud shading
im_right = axes[0].pcolormesh(X, Z, Br_slice, cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax, shading='gouraud',
                               rasterized=True)
im_left = axes[0].pcolormesh(-X, Z, Br_slice, cmap='RdBu_r',
                              vmin=-vmax, vmax=vmax, shading='gouraud',
                              rasterized=True)
th_c = np.linspace(0, 2*np.pi, 200)
axes[0].plot(np.cos(th_c), np.sin(th_c), 'k-', lw=1)
axes[0].set_xlim(-r_lim, r_lim)
axes[0].set_ylim(-r_lim, r_lim)
axes[0].set_aspect('equal')
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('z', fontsize=12)

# Right: shell map, gouraud shading
k_shell = np.searchsorted(radii, 3.0)
PHI_deg, TH_deg = np.meshgrid(np.degrees(phi), np.degrees(theta))
im_shell = axes[1].pcolormesh(PHI_deg, TH_deg, np.zeros((N_th, N_ph)),
                               cmap='RdBu_r', shading='gouraud',
                               rasterized=True)
axes[1].set_xlabel(r'$\phi$ (deg)', fontsize=12)
axes[1].set_ylabel(r'$\theta$ (deg)', fontsize=12)
axes[1].invert_yaxis()

title = fig.suptitle('', fontsize=14, y=0.98)
fig.colorbar(im_right, ax=axes[0], shrink=0.8, label=r'$B_r \times r^2$')
fig.colorbar(im_shell, ax=axes[1], shrink=0.8, label=r'$B_r$')
fig.tight_layout(rect=[0, 0, 1, 0.95])

# ---- Render ----
def update(frame_idx):
    with h5py.File(snap_files[frame_idx], "r") as f:
        Br = f["Br"][:].reshape(N_r, N_th, N_ph)
        t_val = f["time"][()]

    # Meridional slice: phi=0 (right half) and phi=pi (left half)
    Br_right = Br[:, :, 0].T * R**2
    Br_left = Br[:, :, N_ph//2].T * R**2
    im_right.set_array(Br_right.ravel())
    im_left.set_array(Br_left.ravel())

    # Shell map
    Br_shell = Br[k_shell, :, :]
    vmax_sh = max(np.percentile(np.abs(Br_shell), 97), 1e-10)
    im_shell.set_array(Br_shell.ravel())
    im_shell.set_clim(-vmax_sh, vmax_sh)

    title.set_text(f't = {t_val:.2f}   (r_shell = {radii[k_shell]:.1f})')
    print(f"  frame {frame_idx+1}/{len(snap_files)}", end='\r')
    return [im_right, im_left, im_shell, title]

print("Rendering movie...")
anim = animation.FuncAnimation(fig, update, frames=len(snap_files),
                                blit=False, interval=80)
out_path = os.path.join(data_dir, "dipole_movie.mp4")
anim.save(out_path, writer='ffmpeg', fps=15, dpi=120,
          extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
print(f"\nMovie saved to {out_path}")
