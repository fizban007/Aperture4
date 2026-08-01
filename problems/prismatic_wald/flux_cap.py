#!/usr/bin/env python3
"""Polar-cap magnetic flux from the tri-face flux cochain.

The Meissner acceptance test for Kerr-Wald (README "Diagnostics"):

    Phi_cap(horizon) / Phi_cap(r=8)

B_f is the primal 2-cochain on tri faces, i.e. the flux integral over the
face, so the cap flux is just a signed sum over the faces of one shell whose
centroid sits in the northern hemisphere.  Face orientation is taken from the
(v0,v1,v2) winding and projected on the outward radial direction, so a face
wound inward contributes with a minus sign.

Usage:
    flux_cap.py RUNDIR [--file ic_aux.h5] [--r-ref 8.0] [--a 0.998]
    flux_cap.py RUNDIR --series          # ratio vs time over all snapshots
"""
import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np


def load_mesh(rundir):
    with h5py.File(os.path.join(rundir, "mesh.h5"), "r") as m:
        d = dict(
            N_r=int(m["N_r"][()]),
            N_tri=int(m["N_tri"][()]),
            N_edge_s=int(m["N_edge_s"][()]),
            N_vert_s=int(m["N_vert_s"][()]),
            L=int(m["L"][()]),
            radii=np.array(m["radii"], dtype=np.float64),
            vx=np.array(m["vert_x"], dtype=np.float64),
            vy=np.array(m["vert_y"], dtype=np.float64),
            vz=np.array(m["vert_z"], dtype=np.float64),
            tv0=np.array(m["tri_face_v0"]),
            tv1=np.array(m["tri_face_v1"]),
            tv2=np.array(m["tri_face_v2"]),
            face_area=np.array(m["face_area"], dtype=np.float64),
        )
        stride_r = int(m["output_radial_stride"][()])
        stride_a = int(m["output_angular_stride"][()])
    if stride_r > 1 or stride_a > 1:
        sys.exit(f"{rundir}: snapshots are downsampled "
                 f"(radial={stride_r}, angular={stride_a}); the cap sum needs "
                 f"every face.  Re-run with strides = 1.")
    return d


def cap_geometry(mesh):
    """Per-tri-face: orientation sign vs outward radial, and centroid z/r."""
    v0, v1, v2 = mesh["tv0"], mesh["tv1"], mesh["tv2"]
    p0 = np.stack([mesh["vx"][v0], mesh["vy"][v0], mesh["vz"][v0]], axis=1)
    p1 = np.stack([mesh["vx"][v1], mesh["vy"][v1], mesh["vz"][v1]], axis=1)
    p2 = np.stack([mesh["vx"][v2], mesh["vy"][v2], mesh["vz"][v2]], axis=1)
    nrm = np.cross(p1 - p0, p2 - p0)
    cen = (p0 + p1 + p2) / 3.0
    radial = np.einsum("ij,ij->i", nrm, cen)
    sign = np.where(radial >= 0.0, 1.0, -1.0)
    cen_r = np.linalg.norm(cen, axis=1)
    cos_theta = cen[:, 2] / np.maximum(cen_r, 1e-300)
    return sign, cos_theta


def cap_flux_profile(mesh, B):
    """Northern-cap flux and whole-shell flux for every shell k."""
    N_r, N_tri = mesh["N_r"], mesh["N_tri"]
    n_tri_faces = (N_r + 1) * N_tri
    if B.size < n_tri_faces:
        sys.exit(f"B has {B.size} entries, expected at least {n_tri_faces}")
    sign, cos_theta = cap_geometry(mesh)
    flux = sign * B[:n_tri_faces].astype(np.float64)
    north = cos_theta > 0.0

    phi_cap = np.zeros(N_r + 1)
    phi_tot = np.zeros(N_r + 1)
    for k in range(N_r + 1):
        s, e = k * N_tri, (k + 1) * N_tri
        phi_cap[k] = flux[s:e][north[s:e]].sum()
        phi_tot[k] = flux[s:e].sum()
    return phi_cap, phi_tot


def wald_cap_flux_analytic(a, r, B0=1.0):
    """Continuum Wald flux through the northern cap of a Boyer-Lindquist
    sphere of radius r: 2*pi*[A_phi(pi/2) - A_phi(0)] with
    A_phi = (B0/2)(g_phiphi + 2 a g_tphi).  Exact at any r, not just r_+."""
    a2 = a * a
    delta = r * r - 2.0 * r + a2
    # theta = pi/2:  rho^2 = r^2, sin^2 = 1
    rho2 = r * r
    g_pp = ((r * r + a2) ** 2 - delta * a2) / rho2
    g_tp = -2.0 * a * r / rho2
    return 2.0 * np.pi * (B0 / 2.0) * (g_pp + 2.0 * a * g_tp)


def shell_index(radii, r_target):
    return int(np.argmin(np.abs(radii - r_target)))


# Fixed evaluation radius for the horizon cap, shared by every level of the
# convergence family.  NOT "the first shell outside r_+": on the nested grid
# each refinement adds shells, so that rule slides inward with L (r = 1.1057
# at L3/L4 but 1.0644 at L5) and the measured ratio then changes for purely
# geometric reasons.  1.10569 is the coarsest common shell strictly outside
# r_+ = 1.06321, present at every level (physical shell 3 * 2^(L-3)).
R_HORIZON_DEFAULT = 1.10569


def report(rundir, fname, r_ref, a, B0, quiet=False, r_h=R_HORIZON_DEFAULT):
    mesh = load_mesh(rundir)
    path = os.path.join(rundir, fname)
    with h5py.File(path, "r") as f:
        key = "B" if "B" in f else "B_f"
        B = np.array(f[key])
        time = float(np.array(f["time"])) if "time" in f else 0.0
    phi_cap, phi_tot = cap_flux_profile(mesh, B)

    radii = mesh["radii"]
    r_plus = 1.0 + np.sqrt(max(0.0, 1.0 - a * a))
    k_h = shell_index(radii, r_h)
    k_ref = shell_index(radii, r_ref)
    if abs(radii[k_h] - r_h) / r_h > 1e-6 and not quiet:
        print(f"  WARNING: no shell at r={r_h}; nearest is {radii[k_h]:.5f}")

    ratio = phi_cap[k_h] / phi_cap[k_ref]
    if not quiet:
        print(f"=== {rundir}/{fname}   L={mesh['L']}  N_r={mesh['N_r']}  "
              f"t={time:.3f} M ===")
        print(f"  r_+ = {r_plus:.5f}")
        print(f"  horizon shell k={k_h:3d}  r={radii[k_h]:.5f}  "
              f"Phi_cap={phi_cap[k_h]:.6e}")
        print(f"  ref     shell k={k_ref:3d}  r={radii[k_ref]:.5f}  "
              f"Phi_cap={phi_cap[k_ref]:.6e}")
        print(f"  Phi_cap(horizon)/Phi_cap(r={radii[k_ref]:.3f}) = {ratio:.6e}")
        ana_h = wald_cap_flux_analytic(a, radii[k_h], B0)
        ana_r = wald_cap_flux_analytic(a, radii[k_ref], B0)
        print(f"  continuum Wald on the same two radii:  {ana_h/ana_r:.6e}")
        print(f"    (cap flux: numeric {phi_cap[k_h]:.6e} vs continuum "
              f"{ana_h:.6e} at the horizon shell)")
        # Monopole sanity: the whole-shell sum must vanish.
        scale = np.abs(phi_cap[k_ref])
        print(f"  monopole check  max_k |Phi_shell| / |Phi_cap(ref)| = "
              f"{np.abs(phi_tot).max()/scale:.3e}")
    return dict(ratio=ratio, k_h=k_h, k_ref=k_ref, r_h=radii[k_h],
                r_ref=radii[k_ref], phi_h=phi_cap[k_h],
                phi_ref=phi_cap[k_ref], time=time, L=mesh["L"],
                N_r=mesh["N_r"])


def series(rundir, r_ref, a, B0, r_h=R_HORIZON_DEFAULT):
    files = sorted(glob.glob(os.path.join(rundir, "step_*.h5")),
                   key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    if not files:
        sys.exit(f"no step_*.h5 snapshots in {rundir}")
    print(f"=== {rundir}: cap flux ratio vs time ===")
    print(f"  {'step':>8} {'t [M]':>10} {'ratio':>14}")
    out = []
    for p in files:
        r = report(rundir, os.path.basename(p), r_ref, a, B0, quiet=True,
                   r_h=r_h)
        step = int(re.search(r"step_(\d+)", p).group(1))
        print(f"  {step:>8} {r['time']:>10.3f} {r['ratio']:>14.6e}")
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--file", default="ic_aux.h5")
    ap.add_argument("--r-ref", type=float, default=8.0)
    ap.add_argument("--a", type=float, default=0.998)
    ap.add_argument("--B0", type=float, default=1.0)
    ap.add_argument("--r-h", type=float, default=R_HORIZON_DEFAULT,
                    help="radius of the horizon-cap shell (must exist on the "
                         "mesh; default is the coarsest common shell outside "
                         "r_+)")
    ap.add_argument("--series", action="store_true",
                    help="report the ratio for every step_*.h5 snapshot")
    args = ap.parse_args()
    if args.series:
        series(args.rundir, args.r_ref, args.a, args.B0, args.r_h)
    else:
        report(args.rundir, args.file, args.r_ref, args.a, args.B0,
               r_h=args.r_h)


if __name__ == "__main__":
    main()
