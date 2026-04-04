#!/usr/bin/env python3
"""Inspect and visualize results from the prismatic mesh dipole simulation."""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import glob

DATA_DIR = Path("Data")

def load_mesh():
    """Load mesh data from mesh.h5"""
    with h5py.File(DATA_DIR / "mesh.h5", "r") as f:
        mesh = {
            "L": f["L"][()],
            "N_r": f["N_r"][()],
            "N_verts": f["N_verts"][()],
            "N_edges": f["N_edges"][()],
            "N_faces": f["N_faces"][()],
            "N_tri": f["N_tri"][()],
            "N_vert_s": f["N_vert_s"][()],
            "N_edge_s": f["N_edge_s"][()],
            "vert_x": f["vert_x"][:],
            "vert_y": f["vert_y"][:],
            "vert_z": f["vert_z"][:],
            "edge_v0": f["edge_v0"][:],
            "edge_v1": f["edge_v1"][:],
            "edge_length": f["edge_length"][:],
            "face_area": f["face_area"][:],
            "edge_boundary": f["edge_boundary"][:],
            "face_boundary": f["face_boundary"][:],
            "edge_radial_layer": f["edge_radial_layer"][:],
            "face_radial_layer": f["face_radial_layer"][:],
            "hodge1_inv": f["hodge1_inv"][:],
            "hodge2": f["hodge2"][:],
            "radii": f["radii"][:],
        }
    return mesh

def load_snapshot(step):
    """Load field data from a snapshot."""
    fname = DATA_DIR / f"step_{step:06d}.h5"
    with h5py.File(fname, "r") as f:
        snap = {
            "step": f["step"][()],
            "time": f["time"][()],
            "D_e": f["D_e"][:],
            "B_f": f["B_f"][:],
            "E_tilde": f["E_tilde"][:],
            "H_tilde": f["H_tilde"][:],
        }
    return snap

def get_available_steps():
    """Find all available snapshot steps."""
    files = sorted(glob.glob(str(DATA_DIR / "step_*.h5")))
    steps = [int(Path(f).stem.split("_")[1]) for f in files]
    return steps

def verify_mesh(mesh):
    """Run basic mesh consistency checks."""
    print("=" * 60)
    print("MESH VERIFICATION")
    print("=" * 60)
    print(f"  L = {mesh['L']}, N_r = {mesh['N_r']}")
    print(f"  Sphere: {mesh['N_vert_s']} vertices, {mesh['N_edge_s']} edges, "
          f"{mesh['N_tri']} triangles")

    # Euler formula on sphere
    V, E, F = mesh['N_vert_s'], mesh['N_edge_s'], mesh['N_tri']
    euler = V - E + F
    print(f"  Euler V-E+F = {V}-{E}+{F} = {euler} (should be 2)")

    print(f"  3D mesh: {mesh['N_verts']} vertices, {mesh['N_edges']} edges, "
          f"{mesh['N_faces']} faces")

    # Vertex radii
    r = np.sqrt(mesh["vert_x"]**2 + mesh["vert_y"]**2 + mesh["vert_z"]**2)
    print(f"  Vertex radii: min={r.min():.4f}, max={r.max():.4f}")
    print(f"  Radial shells: {mesh['radii'][:3]} ... {mesh['radii'][-3:]}")

    # Edge lengths
    el = mesh["edge_length"]
    print(f"  Edge lengths: min={el.min():.6f}, max={el.max():.6f}, "
          f"mean={el.mean():.6f}")

    # Face areas
    fa = mesh["face_area"]
    print(f"  Face areas: min={fa.min():.6f}, max={fa.max():.6f}, "
          f"mean={fa.mean():.6f}")

    # Hodge star
    h1 = mesh["hodge1_inv"]
    h2 = mesh["hodge2"]
    print(f"  Hodge1_inv: min={h1.min():.6f}, max={h1.max():.6f}, "
          f"zeros={np.sum(h1==0)}")
    print(f"  Hodge2: min={h2.min():.6f}, max={h2.max():.6f}, "
          f"zeros={np.sum(h2==0)}")

    # Boundary counts
    eb = mesh["edge_boundary"]
    fb = mesh["face_boundary"]
    print(f"  Edge boundaries: inner={np.sum(eb==1)}, outer={np.sum(eb==2)}, "
          f"interior={np.sum(eb==0)}")
    print(f"  Face boundaries: inner={np.sum(fb==1)}, outer={np.sum(fb==2)}, "
          f"interior={np.sum(fb==0)}")

    return True

def analyze_fields(mesh, snap):
    """Analyze field data from a snapshot."""
    print(f"\n--- Step {snap['step']}, time = {snap['time']:.4f} ---")
    D = snap["D_e"]
    B = snap["B_f"]
    E = snap["E_tilde"]
    H = snap["H_tilde"]

    print(f"  D_e: rms={np.sqrt(np.mean(D**2)):.6e}, "
          f"min={D.min():.6e}, max={D.max():.6e}")
    print(f"  B_f: rms={np.sqrt(np.mean(B**2)):.6e}, "
          f"min={B.min():.6e}, max={B.max():.6e}")
    print(f"  E_tilde: rms={np.sqrt(np.mean(E**2)):.6e}")
    print(f"  H_tilde: rms={np.sqrt(np.mean(H**2)):.6e}")
    print(f"  NaN check: D={np.any(np.isnan(D))}, B={np.any(np.isnan(B))}, "
          f"E={np.any(np.isnan(E))}, H={np.any(np.isnan(H))}")
    print(f"  Inf check: D={np.any(np.isinf(D))}, B={np.any(np.isinf(B))}")


def compute_edge_radial_profile(mesh, field_on_edges, label="field"):
    """Compute RMS of an edge field as a function of radial layer."""
    N_r = mesh["N_r"]
    layers = mesh["edge_radial_layer"]
    radii = mesh["radii"]

    rms_per_layer = np.zeros(N_r + 1)
    count_per_layer = np.zeros(N_r + 1)

    for e in range(mesh["N_edges"]):
        k = layers[e]
        rms_per_layer[k] += field_on_edges[e]**2
        count_per_layer[k] += 1

    mask = count_per_layer > 0
    rms_per_layer[mask] = np.sqrt(rms_per_layer[mask] / count_per_layer[mask])

    return radii[:N_r+1], rms_per_layer


def compute_face_radial_profile(mesh, field_on_faces, label="field"):
    """Compute RMS of a face field as a function of radial layer."""
    N_r = mesh["N_r"]
    layers = mesh["face_radial_layer"]
    radii = mesh["radii"]

    rms_per_layer = np.zeros(N_r + 1)
    count_per_layer = np.zeros(N_r + 1)

    for f in range(mesh["N_faces"]):
        k = layers[f]
        rms_per_layer[k] += field_on_faces[f]**2
        count_per_layer[k] += 1

    mask = count_per_layer > 0
    rms_per_layer[mask] = np.sqrt(rms_per_layer[mask] / count_per_layer[mask])

    return radii[:N_r+1], rms_per_layer


def plot_radial_profiles(mesh, steps_to_plot):
    """Plot radial profiles of E and B at different times."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for step in steps_to_plot:
        try:
            snap = load_snapshot(step)
        except FileNotFoundError:
            continue

        r_e, rms_E = compute_edge_radial_profile(mesh, snap["E_tilde"])
        r_b, rms_B = compute_face_radial_profile(mesh, snap["B_f"])

        axes[0].plot(r_e, rms_E, label=f"t={snap['time']:.1f}")
        axes[1].plot(r_b, rms_B, label=f"t={snap['time']:.1f}")

    # Add 1/r^3 dipole reference
    radii = mesh["radii"]
    r_ref = radii[radii > 0]
    dipole_ref = r_ref**(-3) * r_ref[0]**3
    axes[1].plot(r_ref, dipole_ref * rms_B[0] if len(steps_to_plot) > 0 else dipole_ref,
                 'k--', alpha=0.5, label=r"$\propto r^{-3}$")

    for ax in axes:
        ax.set_xlabel("r")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("RMS E_tilde (on edges)")
    axes[0].set_title("Electric field radial profile")
    axes[1].set_ylabel("RMS B_f (on faces)")
    axes[1].set_title("Magnetic flux radial profile")

    plt.tight_layout()
    plt.savefig("radial_profiles.png", dpi=150)
    print("\nSaved radial_profiles.png")


def plot_time_evolution(mesh, all_steps):
    """Plot time evolution of global field energy."""
    times = []
    E_energy = []
    B_energy = []

    for step in all_steps:
        try:
            snap = load_snapshot(step)
        except FileNotFoundError:
            continue

        times.append(snap["time"])
        # Approximate energy: sum of D*E and B*H
        # E_energy ~ sum(D_e * E_tilde) (dot product gives energy-like quantity)
        E_energy.append(np.sum(snap["D_e"] * snap["E_tilde"]))
        B_energy.append(np.sum(snap["B_f"] * snap["H_tilde"]))

    times = np.array(times)
    E_energy = np.array(E_energy)
    B_energy = np.array(B_energy)
    total = E_energy + B_energy

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, E_energy, label="Electric energy")
    ax.plot(times, B_energy, label="Magnetic energy")
    ax.plot(times, total, 'k-', label="Total EM energy", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy (arb. units)")
    ax.set_title("EM Energy Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("energy_evolution.png", dpi=150)
    print("Saved energy_evolution.png")


def plot_sphere_slice(mesh, snap, shell_index=0):
    """Plot B_f on a particular radial shell (triangular faces)."""
    N_tri = mesh["N_tri"]
    N_r = mesh["N_r"]

    # Get triangular face indices on this shell
    face_start = shell_index * N_tri
    face_end = face_start + N_tri
    B_on_shell = snap["B_f"][face_start:face_end]

    tri_start = shell_index * N_tri

    # Get vertex positions for this shell
    N_vert_s = mesh["N_vert_s"]
    vx = mesh["vert_x"][shell_index * N_vert_s : (shell_index + 1) * N_vert_s]
    vy = mesh["vert_y"][shell_index * N_vert_s : (shell_index + 1) * N_vert_s]
    vz = mesh["vert_z"][shell_index * N_vert_s : (shell_index + 1) * N_vert_s]

    # Project to Mollweide-like: use theta, phi
    r = np.sqrt(vx**2 + vy**2 + vz**2)
    theta = np.arccos(np.clip(vz / r, -1, 1))
    phi = np.arctan2(vy, vx)

    # Get triangle vertex indices (local to this shell)
    # tri_face_v0 etc. are global vertex indices
    from h5py import File
    with File(DATA_DIR / "mesh.h5", "r") as f:
        tf_v0 = f["tri_face_v0"][tri_start:tri_start + N_tri]
        tf_v1 = f["tri_face_v1"][tri_start:tri_start + N_tri]
        tf_v2 = f["tri_face_v2"][tri_start:tri_start + N_tri]

    # Convert to local vertex indices
    base = shell_index * N_vert_s
    tf_v0_local = tf_v0 - base
    tf_v1_local = tf_v1 - base
    tf_v2_local = tf_v2 - base

    # Plot using tripcolor
    import matplotlib.tri as mtri
    triangulation = mtri.Triangulation(phi, theta,
                                        np.column_stack([tf_v0_local,
                                                         tf_v1_local,
                                                         tf_v2_local]))

    fig, ax = plt.subplots(figsize=(12, 6))
    vmax = np.percentile(np.abs(B_on_shell), 95)
    tc = ax.tripcolor(triangulation, B_on_shell, cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax, shading='flat')
    plt.colorbar(tc, ax=ax, label="B_f")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\theta$")
    ax.set_title(f"B_f on shell {shell_index} (r={mesh['radii'][shell_index]:.2f}), "
                 f"t={snap['time']:.2f}")
    ax.invert_yaxis()

    plt.tight_layout()
    fname = f"B_shell_{shell_index}_step_{snap['step']:06d}.png"
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")


def main():
    mesh = load_mesh()
    verify_mesh(mesh)

    steps = get_available_steps()
    print(f"\nAvailable snapshots: {len(steps)} steps from {steps[0]} to {steps[-1]}")

    # Analyze initial and final snapshots
    snap0 = load_snapshot(steps[0])
    analyze_fields(mesh, snap0)

    snap_final = load_snapshot(steps[-1])
    analyze_fields(mesh, snap_final)

    # Check a mid-time snapshot
    mid_step = steps[len(steps) // 2]
    snap_mid = load_snapshot(mid_step)
    analyze_fields(mesh, snap_mid)

    # Plot radial profiles at several times
    profile_steps = [steps[0], steps[len(steps)//4], steps[len(steps)//2],
                     steps[3*len(steps)//4], steps[-1]]
    plot_radial_profiles(mesh, profile_steps)

    # Plot time evolution
    plot_time_evolution(mesh, steps)

    # Plot B on inner shell and a mid-radius shell
    plot_sphere_slice(mesh, snap_final, shell_index=0)
    plot_sphere_slice(mesh, snap_final, shell_index=mesh["N_r"] // 2)


if __name__ == "__main__":
    main()
