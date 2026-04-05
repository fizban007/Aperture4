#!/usr/bin/env python3
"""Movie of Br*r^2 from vacuum rotating oblique dipole."""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob

data_dir = "Data_vacuum_dipole"

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
    Br = f['Br'][:].reshape(N_r, N_th, N_ph)
vmax = np.percentile(np.abs(Br[:,:,0].T * R**2), 97)

fig, ax = plt.subplots(figsize=(9, 9))
im_r = ax.pcolormesh(X, Z, np.zeros_like(X), cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, shading='gouraud', rasterized=True)
im_l = ax.pcolormesh(-X, Z, np.zeros_like(X), cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, shading='gouraud', rasterized=True)
tc = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(tc), np.sin(tc), 'k-', lw=1)
ax.set_xlim(-8, 8); ax.set_ylim(-8, 8); ax.set_aspect('equal')
ax.set_xlabel('x', fontsize=12); ax.set_ylabel('z', fontsize=12)
title = ax.set_title('', fontsize=14)
fig.colorbar(im_r, ax=ax, shrink=0.7, label=r'$B_r \times r^2$')
fig.tight_layout()

def update(frame_idx):
    with h5py.File(snap_files[frame_idx]) as f:
        Br = f['Br'][:].reshape(N_r, N_th, N_ph)
        t = f['time'][()]
    im_r.set_array((Br[:,:,0].T * R**2).ravel())
    im_l.set_array((Br[:,:,N_ph//2].T * R**2).ravel())
    title.set_text(f'$B_r \\times r^2$   t = {t:.2f}')
    print(f"  frame {frame_idx+1}/{len(snap_files)}", end='\r')
    return [im_r, im_l, title]

print("Rendering movie...")
anim = animation.FuncAnimation(fig, update, frames=len(snap_files),
                                blit=False, interval=80)
anim.save(f'{data_dir}/vacuum_dipole.mp4', writer='ffmpeg', fps=15, dpi=120,
          extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
print(f"\nMovie saved to {data_dir}/vacuum_dipole.mp4")
