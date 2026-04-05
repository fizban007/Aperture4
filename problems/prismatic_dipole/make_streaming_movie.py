#!/usr/bin/env python3
"""Movie of particles streaming along oblique dipole field lines."""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os

data_dir = "Data_gca_streaming"
obliquity = 0.5236  # 30 degrees

with h5py.File(f'{data_dir}/sph_grid.h5') as f:
    theta = f['theta'][:]; phi = f['phi'][:]; radii = f['radii'][:]
N_th, N_ph, N_r = len(theta), len(phi), len(radii)

R, TH = np.meshgrid(radii, theta)
X = R * np.sin(TH); Z = R * np.cos(TH)

snap_files = sorted(glob.glob(f'{data_dir}/sph_0*.h5'))
snap_files = [f for f in snap_files if 'grid' not in f]
print(f"Found {len(snap_files)} snapshots")

# Color scale from mid-simulation
with h5py.File(snap_files[len(snap_files)//2]) as f:
    rho = f['rho'][:].reshape(N_r, N_th, N_ph)
rho_r = rho[:,:,0].T; rho_l = rho[:,:,N_ph//2].T
vmax = np.percentile(np.abs(np.concatenate([rho_r.ravel(), rho_l.ravel()])), 99)
print(f"Color scale: +/-{vmax:.4e}")

# Oblique dipole field lines in the phi=0 plane
# Magnetic axis: m_hat = (sin(alpha), 0, cos(alpha))
# In the phi=0 meridional plane, points are at (x, 0, z) = (r*sin(th), 0, r*cos(th))
# Magnetic colatitude: cos(th_m) = r_hat . m_hat = sin(th)*sin(alpha) + cos(th)*cos(alpha)
#                                                = cos(th - alpha)
# Field line: r_m = L * sin^2(th_m) where th_m is measured from magnetic axis
def dipole_field_lines(alpha, L_values, ax, color='green', lw=0.6):
    for L_val in L_values:
        # Parametrize by magnetic colatitude
        th_m = np.linspace(0.02, np.pi - 0.02, 500)
        r_fl = L_val * np.sin(th_m)**2
        # Convert magnetic coords to geographic
        # x_m = r*sin(th_m), z_m = r*cos(th_m) in magnetic frame
        # Rotate by alpha around y-axis to get geographic frame
        x_m = r_fl * np.sin(th_m)
        z_m = r_fl * np.cos(th_m)
        x_geo = x_m * np.cos(alpha) + z_m * np.sin(alpha)
        z_geo = -x_m * np.sin(alpha) + z_m * np.cos(alpha)
        mask = (r_fl > 1.0) & (r_fl < 9.5)
        # Split into segments to avoid connecting across origin
        segments = np.split(np.arange(len(th_m)), np.where(~mask)[0])
        for seg in segments:
            if len(seg) > 2:
                ax.plot(x_geo[seg], z_geo[seg], color=color, alpha=0.4, lw=lw)

# Set up figure
fig, ax = plt.subplots(figsize=(10, 10))
im_r = ax.pcolormesh(X, Z, np.zeros_like(X), cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, shading='gouraud', rasterized=True)
im_l = ax.pcolormesh(-X, Z, np.zeros_like(X), cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, shading='gouraud', rasterized=True)
tc = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(tc), np.sin(tc), 'k-', lw=1)
dipole_field_lines(obliquity, [1.5, 2, 3, 4, 5, 7, 9], ax)
ax.set_xlim(-9, 9); ax.set_ylim(-9, 9); ax.set_aspect('equal')
ax.set_xlabel('x', fontsize=12); ax.set_ylabel('z', fontsize=12)
title = ax.set_title('', fontsize=14)
fig.colorbar(im_r, ax=ax, shrink=0.7, label=r'$\rho$')
fig.tight_layout()

def update(frame_idx):
    with h5py.File(snap_files[frame_idx]) as f:
        rho = f['rho'][:].reshape(N_r, N_th, N_ph)
        t = f['time'][()]
    im_r.set_array((rho[:,:,0].T).ravel())
    im_l.set_array((rho[:,:,N_ph//2].T).ravel())
    title.set_text(f'Particles streaming along oblique dipole  t = {t:.2f}')
    print(f"  frame {frame_idx+1}/{len(snap_files)}", end='\r')
    return [im_r, im_l, title]

print("Rendering movie...")
anim = animation.FuncAnimation(fig, update, frames=len(snap_files),
                                blit=False, interval=80)
out_path = f'{data_dir}/streaming_movie.mp4'
anim.save(out_path, writer='ffmpeg', fps=15, dpi=120,
          extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
print(f"\nMovie saved to {out_path}")

# Also save a still frame
update(len(snap_files)//2)
fig.savefig(f'{data_dir}/streaming_snapshot.png', dpi=120)
print("Snapshot saved")
