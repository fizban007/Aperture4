#!/usr/bin/env python3
"""Analyze the initial drift rates ‖dD/dt‖, ‖dB/dt‖ at the Kerr-Wald IC.

Loads ic_aux.h5 from a run directory, localizes each edge/face to its
shell radius, and reports:
  - global RMS and max(|dD/dt|/|D|_rms), etc. — scale of the drift
  - per-shell RMS — spatial localization (horizon vs. bulk)
"""
import argparse, os, sys
import numpy as np
import h5py


def load(path):
    with h5py.File(os.path.join(path, "ic_aux.h5")) as f:
        D     = np.array(f["D"])
        B     = np.array(f["B"])
        E_aux = np.array(f["E_aux"])
        H_aux = np.array(f["H_aux"])
        dD    = np.array(f["dD_dt"])
        dB    = np.array(f["dB_dt"])
    with h5py.File(os.path.join(path, "mesh.h5")) as f:
        N_r       = int(f["N_r"][()])
        N_edge_s  = int(f["N_edge_s"][()])
        N_vert_s  = int(f["N_vert_s"][()])
        N_tri     = int(f["N_tri"][()])
        radii     = np.array(f["radii"])
        edge_len  = np.array(f["edge_length"])
        face_area = np.array(f["face_area"])
    return dict(
        D=D, B=B, E_aux=E_aux, H_aux=H_aux, dD=dD, dB=dB,
        N_r=N_r, N_edge_s=N_edge_s, N_vert_s=N_vert_s, N_tri=N_tri,
        radii=radii, edge_len=edge_len, face_area=face_area,
    )


def shell_of_edge(e, N_h, N_edge_s, N_vert_s):
    """Return the shell k (or k_lower for vertical edges) for edge index e."""
    if e < N_h:
        k = e // N_edge_s                          # horizontal edge on shell k
        return k
    vi = e - N_h
    k = vi // N_vert_s                              # vertical edge from shell k
    return k  # call-site decides whether to use k or k+½


def shell_of_face(f, N_tri_faces, N_tri, N_edge_s):
    """Return the shell k (or k_lower for rect faces) for face index f."""
    if f < N_tri_faces:
        return f // N_tri                           # tri face on shell k
    ri = f - N_tri_faces
    return ri // N_edge_s                            # rect face between k and k+1


def per_shell_rms(values, shells, N_shells):
    """RMS of values binned by shell index."""
    out = np.zeros(N_shells)
    cnt = np.zeros(N_shells, dtype=int)
    for v, k in zip(values, shells):
        if 0 <= k < N_shells:
            out[k] += v * v
            cnt[k] += 1
    return np.sqrt(out / np.maximum(cnt, 1))


def analyze(path, label):
    d = load(path)
    N_r = d["N_r"]
    N_h = (N_r + 1) * d["N_edge_s"]
    N_v = N_r * d["N_vert_s"]
    N_tri_faces = (N_r + 1) * d["N_tri"]
    N_rect_faces = N_r * d["N_edge_s"]
    N_edges = N_h + N_v
    N_faces = N_tri_faces + N_rect_faces

    D, B, dD, dB = d["D"], d["B"], d["dD"], d["dB"]
    assert D.size == N_edges, (D.size, N_edges)
    assert B.size == N_faces, (B.size, N_faces)

    # Global scales.
    D_rms, B_rms = np.sqrt((D**2).mean()), np.sqrt((B**2).mean())
    dD_rms, dB_rms = np.sqrt((dD**2).mean()), np.sqrt((dB**2).mean())
    dD_max, dB_max = np.abs(dD).max(), np.abs(dB).max()

    # Split by edge/face type.
    dD_h, dD_v = dD[:N_h], dD[N_h:]
    dB_tri, dB_rect = dB[:N_tri_faces], dB[N_tri_faces:]
    D_h, D_v = D[:N_h], D[N_h:]
    B_tri, B_rect = B[:N_tri_faces], B[N_tri_faces:]

    print(f"\n=== {label}  (L subdivision implied by N_edge_s={d['N_edge_s']}, N_r={N_r}) ===")
    print(f"  Field scale       ‖D‖_rms = {D_rms:.4e}    ‖B‖_rms = {B_rms:.4e}")
    print(f"  Drift rate (all)  ‖dD/dt‖_rms = {dD_rms:.4e}   ‖dB/dt‖_rms = {dB_rms:.4e}")
    print(f"                    (normed)    = {dD_rms/D_rms:.4e}              = {dB_rms/B_rms:.4e}")
    print(f"  Drift rate (max)  |dD/dt|_max = {dD_max:.4e}   |dB/dt|_max = {dB_max:.4e}")
    print(f"                    (normed)    = {dD_max/D_rms:.4e}              = {dB_max/B_rms:.4e}")
    print(f"  E-fold time       τ_D = {D_rms/max(dD_rms,1e-30):.2f} M        τ_B = {B_rms/max(dB_rms,1e-30):.2f} M")

    # Split by element type.
    def _norm(x, ref): return np.sqrt((x**2).mean()) / ref
    print(f"  By element type:")
    print(f"    horiz edges ({N_h}):  ‖dD/dt‖/‖D‖ = {_norm(dD_h, D_rms):.4e}")
    print(f"    vert edges  ({N_v}):  ‖dD/dt‖/‖D‖ = {_norm(dD_v, D_rms):.4e}")
    print(f"    tri faces   ({N_tri_faces}):  ‖dB/dt‖/‖B‖ = {_norm(dB_tri, B_rms):.4e}")
    print(f"    rect faces  ({N_rect_faces}):  ‖dB/dt‖/‖B‖ = {_norm(dB_rect, B_rms):.4e}")

    # Per-shell RMS profile.
    radii = d["radii"]
    a = 0.998
    r_plus = 1.0 + np.sqrt(max(0.0, 1.0 - a * a))

    # Horizontal edges: shell = k, so indexing straightforward.
    dD_h_per_shell = np.zeros(N_r + 1)
    D_h_per_shell = np.zeros(N_r + 1)
    for k in range(N_r + 1):
        s, e = k * d["N_edge_s"], (k + 1) * d["N_edge_s"]
        dD_h_per_shell[k] = np.sqrt((dD[s:e]**2).mean())
        D_h_per_shell[k] = np.sqrt((D[s:e]**2).mean())

    # Tri faces per shell.
    dB_tri_per_shell = np.zeros(N_r + 1)
    B_tri_per_shell = np.zeros(N_r + 1)
    for k in range(N_r + 1):
        s, e = k * d["N_tri"], (k + 1) * d["N_tri"]
        dB_tri_per_shell[k] = np.sqrt((dB[s:e]**2).mean())
        B_tri_per_shell[k] = np.sqrt((B[s:e]**2).mean())

    # Rect faces: between shells k and k+1.  Label by k+½.
    dB_rect_per_slab = np.zeros(N_r)
    B_rect_per_slab = np.zeros(N_r)
    for k in range(N_r):
        s = N_tri_faces + k * d["N_edge_s"]
        e = N_tri_faces + (k + 1) * d["N_edge_s"]
        dB_rect_per_slab[k] = np.sqrt((dB[s:e]**2).mean())
        B_rect_per_slab[k] = np.sqrt((B[s:e]**2).mean())

    # Print a radial profile focused near the horizon + bulk samples.
    print(f"  Horizon r_+ ≈ {r_plus:.3f} M")
    print(f"  Per-shell profile (horizontal edges — ‖dD/dt‖ normalized by ‖D‖):")
    print(f"    {'k':>4} {'r':>8} {'‖dD‖_h':>12} {'‖D‖_h':>12} {'ratio':>10}")
    show_indices = list(range(0, min(10, N_r + 1))) \
                 + list(range(N_r + 1 - 5, N_r + 1)) \
                 + [N_r // 2]
    show_indices = sorted(set(show_indices))
    for k in show_indices:
        ratio = dD_h_per_shell[k] / max(D_h_per_shell[k], 1e-30)
        print(f"    {k:>4} {radii[k]:>8.3f} {dD_h_per_shell[k]:>12.4e} "
              f"{D_h_per_shell[k]:>12.4e} {ratio:>10.3e}")

    print(f"  Per-shell profile (tri faces — ‖dB/dt‖ normalized by ‖B‖):")
    print(f"    {'k':>4} {'r':>8} {'‖dB‖_tri':>12} {'‖B‖_tri':>12} {'ratio':>10}")
    for k in show_indices:
        ratio = dB_tri_per_shell[k] / max(B_tri_per_shell[k], 1e-30)
        print(f"    {k:>4} {radii[k]:>8.3f} {dB_tri_per_shell[k]:>12.4e} "
              f"{B_tri_per_shell[k]:>12.4e} {ratio:>10.3e}")

    return dict(dD_rms=dD_rms, dB_rms=dB_rms,
                D_rms=D_rms, B_rms=B_rms,
                dD_max=dD_max, dB_max=dB_max,
                dD_h_per_shell=dD_h_per_shell,
                dB_tri_per_shell=dB_tri_per_shell,
                D_h_per_shell=D_h_per_shell,
                B_tri_per_shell=B_tri_per_shell,
                radii=radii, N_r=N_r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Run directories containing ic_aux.h5, mesh.h5")
    ap.add_argument("--labels", nargs="+", default=None)
    args = ap.parse_args()
    labels = args.labels if args.labels else args.paths
    results = [analyze(p, l) for p, l in zip(args.paths, labels)]

    if len(results) == 2:
        r4, r5 = results
        print("\n=== Convergence between the two runs ===")
        print(f"  ‖dD/dt‖_rms  ratio (run1/run2) = {r4['dD_rms']/r5['dD_rms']:.3f}  (expect ~4 for O(h²))")
        print(f"  ‖dB/dt‖_rms  ratio (run1/run2) = {r4['dB_rms']/r5['dB_rms']:.3f}  (expect ~4 for O(h²))")
        print(f"  |dD/dt|_max  ratio (run1/run2) = {r4['dD_max']/r5['dD_max']:.3f}")
        print(f"  |dB/dt|_max  ratio (run1/run2) = {r4['dB_max']/r5['dB_max']:.3f}")


if __name__ == "__main__":
    main()
