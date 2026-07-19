#!/usr/bin/env python3
"""Post-processing replacement for the in-code sph output (Phase 7D F9).

Reads the exporter's mesh-native dumps (step_XXXXXX.h5: E_e, B_f and,
for PIC runs, J_e / rho / rho_abs / gamma_wsum) plus mesh.h5 (full or
7D sphere-only), interpolates onto an (r_k, theta, phi) grid and writes
sph_XXXXXX.h5 with the same datasets and conventions as the retired
distributed prismatic_sph_output path:

  Er/Eth/Eph   D_i, coord-basis covariant      (E_or, E_ot*r, E_op*r*sin)
  Br/Bth/Bph   B^i, coord-basis contravariant  (flat metric: B_or,
               B_ot/r, B_op/(r sin); exact-pole B^phi nodes get the 0
               sentinel — genuine coordinate singularity)
  Jr/Jth/Jph   J_i, like D_i (J converted from the dual-2 cochain via
               the ANALYTIC hodge1_inv when the dump flags J_kind_dual2)
  rho/rho_abs  hat-interpolated vertex densities (cochain / lumped dual
               volume, computed analytically on sphere-only meshes)
  gamma_mean   gamma_wsum / rho_abs (raw cochain ratio)

Flat metric only (the distributed PIC stack is flat; the KS path stayed
single-rank).  Agreement with the in-code single-rank sph output is at
float-roundoff level (the C++ pipeline computes in float, this one in
float64) — validated to <= 1e-4 relative in the 7D parity check.

Usage:
  python sph_from_dump.py DATA_DIR [--steps 40 80 ...] [--ntheta 16]
         [--nphi 32] [--no-recovery] [--suffix ""]
With no --steps, every step_*.h5 in DATA_DIR is processed.
"""

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prismatic_recovery import (  # noqa: E402
    Mesh,
    build_recovery,
    locate,
    vertex_field,
)


# =========================================================================
# Analytic per-element geometry (mirrors prismatic_mesh_geom.h; needed
# when mesh.h5 is the 7D sphere-only flavor).
# =========================================================================

def sph_tri_omega(mesh):
    V = mesh.sphere_v[mesh.tri_verts]  # (T, 3, 3)
    a, b, c = V[:, 0], V[:, 1], V[:, 2]
    num = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = (
        1.0
        + np.einsum("ij,ij->i", a, b)
        + np.einsum("ij,ij->i", b, c)
        + np.einsum("ij,ij->i", c, a)
    )
    return 2.0 * np.arctan2(np.abs(num), den)


def vert_dual_vol(mesh):
    """Lumped vertex dual volumes, [(N_r+1) * N_vert_s]."""
    omega = sph_tri_omega(mesh)
    r = mesh.radii
    N_r, NVs = mesh.N_r, mesh.N_vert_s
    vol = np.zeros((N_r + 1, NVs))
    for k in range(N_r):
        a, b = r[k], r[k + 1]
        dr = b - a
        v_tot = omega * (b**3 - a**3) / 3.0
        v_top = (omega / dr) * ((b**4 - a**4) / 4.0 - a * (b**3 - a**3) / 3.0)
        v_bot = v_tot - v_top
        for j in range(3):
            np.add.at(vol[k], mesh.tri_verts[:, j], v_bot / 3.0)
            np.add.at(vol[k + 1], mesh.tri_verts[:, j], v_top / 3.0)
    return vol.reshape(-1)


def hodge1_inv(mesh):
    """Diagonal Hodge |e|/|e*| for the combined [h|v] edge cochain."""
    V = mesh.sphere_v
    r = mesh.radii
    N_r, NVs, NEs, NT = mesh.N_r, mesh.N_vert_s, mesh.N_edge_s, mesh.N_tri

    # circumcenter directions
    tv = mesh.tri_verts
    e1 = V[tv[:, 1]] - V[tv[:, 0]]
    e2 = V[tv[:, 2]] - V[tv[:, 0]]
    n = np.cross(e1, e2)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    cen = (V[tv[:, 0]] + V[tv[:, 1]] + V[tv[:, 2]]) / 3.0
    flip = np.einsum("ij,ij->i", n, cen) < 0
    n[flip] *= -1.0

    # sphere-edge adjacent tris
    etris = -np.ones((NEs, 2), dtype=int)
    for t in range(NT):
        for e in mesh.tri_edges_s[t]:
            if etris[e, 0] < 0:
                etris[e, 0] = t
            else:
                etris[e, 1] = t
    dot = np.clip(
        np.einsum("ij,ij->i", n[etris[:, 0]], n[etris[:, 1]]), -1.0, 1.0
    )
    beta = np.arccos(dot)
    ea, eb = mesh.sphere_edges[:, 0], mesh.sphere_edges[:, 1]
    alpha = np.arccos(
        np.clip(np.einsum("ij,ij->i", V[ea], V[eb]), -1.0, 1.0)
    )

    r_mid = 0.5 * (r[:-1] + r[1:])
    h1 = np.zeros((N_r + 1, NEs))
    for k in range(N_r + 1):
        if k == 0:
            r_lo, r_hi = r[0], r_mid[0]
        elif k == N_r:
            r_lo, r_hi = r_mid[N_r - 1], r[N_r]
        else:
            r_lo, r_hi = r_mid[k - 1], r_mid[k]
        area = (r_hi - r_lo) * 0.5 * (r_lo + r_hi) * beta
        h1[k] = np.where(area > 0, r[k] * alpha / area, 0.0)

    # vertex dual polygon solid angles (sorted circumcenter fan)
    Omega = np.zeros(NVs)
    for s in range(NVs):
        tris = mesh.vert_tris[s]
        anchor = V[s]
        if abs(anchor[0]) < 0.9:
            t_ax = np.array([0.0, -anchor[2], anchor[1]])
        else:
            t_ax = np.array([anchor[2], 0.0, -anchor[0]])
        t_ax /= np.linalg.norm(t_ax)
        b_ax = np.cross(anchor, t_ax)
        d = n[tris] - anchor
        ang = np.arctan2(d @ b_ax, d @ t_ax)
        order = np.argsort(ang)
        u = n[np.asarray(tris)[order]]
        om = 0.0
        for i in range(len(tris)):
            p, q = u[i], u[(i + 1) % len(tris)]
            num = anchor @ np.cross(p, q)
            den = 1.0 + anchor @ p + p @ q + q @ anchor
            om += 2.0 * np.arctan2(abs(num), den)
        Omega[s] = om
    v1 = np.zeros((N_r, NVs))
    for k in range(N_r):
        area = r_mid[k] ** 2 * Omega
        v1[k] = np.where(area > 0, (r[k + 1] - r[k]) / area, 0.0)

    return np.concatenate([h1.reshape(-1), v1.reshape(-1)])


# =========================================================================
# The sph grid computation.
# =========================================================================

def make_grid(mesh, n_theta, n_phi):
    theta = np.pi * np.arange(n_theta) / (n_theta - 1)
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    sx = (np.sin(th) * np.cos(ph)).ravel()
    sy = (np.sin(th) * np.sin(ph)).ravel()
    sz = np.cos(th).ravel()
    frame = dict(
        cos_phi=np.cos(ph).ravel(),
        sin_phi=np.sin(ph).ravel(),
        theta=theta,
        phi=phi,
    )
    return np.stack([sx, sy, sz], axis=1), frame


def project(vec, s_hat, frame):
    """Cartesian vectors -> orthonormal (r, theta, phi) projections with
    the per-column meridian frame (matches the C++ pole convention)."""
    sx, sy, sz = s_hat[:, 0], s_hat[:, 1], s_hat[:, 2]
    cos_th = sz
    sin_th = np.sqrt(sx * sx + sy * sy)
    cp, sp = frame["cos_phi"], frame["sin_phi"]
    v_r = vec[:, 0] * sx + vec[:, 1] * sy + vec[:, 2] * sz
    v_t = vec[:, 0] * cos_th * cp + vec[:, 1] * cos_th * sp - vec[:, 2] * sin_th
    v_p = -vec[:, 0] * sp + vec[:, 1] * cp
    return v_r, v_t, v_p, sin_th


def gather_at_shell(mesh, E_e, B_f, s_hat, tri, lam, k, zeta):
    """Whitney E and B at unit directions s_hat with EXPLICIT (layer k,
    zeta) — the in-code sph convention: shell points always interpolate
    in the layer STARTING at the shell (top shell: layer N_r-1, zeta 1),
    never in the layer the float radius happens to round into (the
    Whitney interpolant's normal components are discontinuous across
    shells, so this choice is part of the output definition)."""
    N = len(s_hat)
    E = np.zeros((N, 3))
    B = np.zeros((N, 3))
    phi = (1.0 - zeta, zeta)
    for i in range(N):
        t = tri[i]
        sv = mesh.tri_verts[t]
        r_mid = 0.5 * (mesh.radii[k] + mesh.radii[k + 1])
        p = r_mid * mesh.sphere_v[sv]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = n @ n
        gl = np.empty((3, 3))
        for a in range(3):
            bb, c = (a + 1) % 3, (a + 2) % 3
            gl[a] = np.cross(n, p[c] - p[bb]) / nn
        rh = lam[i] @ mesh.sphere_v[sv]
        rh = rh / np.linalg.norm(rh)
        dz = rh / (mesh.radii[k + 1] - mesh.radii[k])

        for j in range(3):
            fi, ti = j, (j + 1) % 3
            sign = mesh.tri_edge_signs[t, j]
            se = mesh.tri_edges_s[t, j]
            wvec = lam[i, fi] * gl[ti] - lam[i, ti] * gl[fi]
            for kk in range(2):
                E[i] += sign * E_e[mesh.h_edge_idx(k + kk, se)] * phi[kk] * wvec
            B[i] += sign * B_f[mesh.rect_face_idx(k, se)] * np.cross(wvec, dz)
        for a in range(3):
            E[i] += E_e[mesh.v_edge_idx(k, sv[a])] * lam[i, a] * dz
        dl12 = np.cross(gl[0], gl[1])
        for kk in range(2):
            B[i] += 2.0 * B_f[mesh.tri_face_idx(k + kk, t)] * phi[kk] * dl12
    return E, B


def recovery_at_shell(mesh, Bv, tri, lam, k, zeta):
    N = len(tri)
    B = np.zeros((N, 3))
    for i in range(N):
        sv = mesh.tri_verts[tri[i]]
        for a in range(3):
            B[i] += lam[i, a] * (
                (1.0 - zeta) * Bv[mesh.vert_idx(k, sv[a])]
                + zeta * Bv[mesh.vert_idx(k + 1, sv[a])]
            )
    return B


def process_step(mesh, step_path, out_path, n_theta, n_phi, use_recovery,
                 rec, dual_vol, h1inv):
    with h5py.File(step_path, "r") as f:
        E_e = f["E_e"][()].astype(np.float64)
        B_f = f["B_f"][()].astype(np.float64)
        J_e = f["J_e"][()].astype(np.float64) if "J_e" in f else None
        rho = f["rho"][()].astype(np.float64) if "rho" in f else None
        rho_abs = f["rho_abs"][()].astype(np.float64) if "rho_abs" in f else None
        gw = f["gamma_wsum"][()].astype(np.float64) if "gamma_wsum" in f else None
        step = int(f["step"][()])
        time = float(f["time"][()])
        j_dual = bool(f["J_kind_dual2"][()]) if "J_kind_dual2" in f else False

    if J_e is not None and j_dual:
        J_e = h1inv * J_e

    s_hat, frame = make_grid(mesh, n_theta, n_phi)
    N_ang = n_theta * n_phi
    N_shells = mesh.N_r + 1

    Bv = None
    if use_recovery:
        Bv = vertex_field(mesh, rec, B_f)

    out = {
        name: np.zeros(N_shells * N_ang)
        for name in ["Br", "Bth", "Bph", "Er", "Eth", "Eph"]
    }
    if J_e is not None:
        for name in ["Jr", "Jth", "Jph"]:
            out[name] = np.zeros(N_shells * N_ang)
    if rho is not None:
        out["rho"] = np.zeros(N_shells * N_ang)
    if rho_abs is not None:
        out["rho_abs"] = np.zeros(N_shells * N_ang)
        if gw is not None:
            out["gamma_mean"] = np.zeros(N_shells * N_ang)

    # ONE angular location pass (triangle assignment is radius-
    # independent); explicit (layer, zeta) per shell below.
    r_loc = 0.5 * (mesh.radii[0] + mesh.radii[1])
    tri, _, lam, _ = locate(mesh, r_loc * s_hat)

    for k in range(N_shells):
        r = mesh.radii[k]
        layer = k if k < mesh.N_r else mesh.N_r - 1
        zeta = 0.0 if k < mesh.N_r else 1.0
        E, B = gather_at_shell(mesh, E_e, B_f, s_hat, tri, lam, layer, zeta)
        if Bv is not None:
            B = recovery_at_shell(mesh, Bv, tri, lam, layer, zeta)

        sl = slice(k * N_ang, (k + 1) * N_ang)
        E_or, E_ot, E_op, sin_th = project(E, s_hat, frame)
        B_or, B_ot, B_op, _ = project(B, s_hat, frame)
        out["Er"][sl] = E_or
        out["Eth"][sl] = E_ot * r
        out["Eph"][sl] = E_op * r * sin_th
        out["Br"][sl] = B_or
        out["Bth"][sl] = B_ot / r
        # Structural pole sentinel: theta rows 0 and n_theta-1 ARE the
        # poles; the float sin(pi) residual must not leak garbage (the
        # in-code path retains a residual at the south pole — this tool
        # writes the intended 0 sentinel at both).
        pole = np.zeros(N_ang, dtype=bool).reshape(n_theta, n_phi)
        pole[0] = pole[-1] = True
        pole = pole.ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            bph = np.where((sin_th > 0) & ~pole, B_op / (r * sin_th), 0.0)
        out["Bph"][sl] = bph

        if J_e is not None:
            J, _ = gather_at_shell(mesh, J_e, B_f, s_hat, tri, lam, layer,
                                   zeta)
            J_or, J_ot, J_op, _ = project(J, s_hat, frame)
            out["Jr"][sl] = J_or
            out["Jth"][sl] = J_ot * r
            out["Jph"][sl] = J_op * r * sin_th

        if rho is not None or rho_abs is not None:
            shells = (layer, layer + 1)
            phi_hat = (1.0 - zeta, zeta)
            rho_val = np.zeros(N_ang)
            ra_val = np.zeros(N_ang)
            ra_raw = np.zeros(N_ang)
            gw_raw = np.zeros(N_ang)
            for lev in range(2):
                for vi in range(3):
                    sv = mesh.tri_verts[tri, vi]
                    v_idx = shells[lev] * mesh.N_vert_s + sv
                    w = lam[:, vi] * phi_hat[lev]
                    inv_vol = 1.0 / dual_vol[v_idx]
                    if rho is not None:
                        rho_val += rho[v_idx] * inv_vol * w
                    if rho_abs is not None:
                        ra_val += rho_abs[v_idx] * inv_vol * w
                        ra_raw += rho_abs[v_idx] * w
                    if gw is not None:
                        gw_raw += gw[v_idx] * w
            if rho is not None:
                out["rho"][sl] = rho_val
            if rho_abs is not None:
                out["rho_abs"][sl] = ra_val
                if gw is not None:
                    out["gamma_mean"][sl] = np.where(
                        ra_raw > 1e-30, gw_raw / np.maximum(ra_raw, 1e-30), 0.0
                    )

    with h5py.File(out_path, "w") as f:
        for name, arr in out.items():
            f.create_dataset(name, data=arr.astype(np.float32))
        f.create_dataset("step", data=np.int32(step))
        f.create_dataset("time", data=np.float64(time))
    print(f"  {os.path.basename(out_path)}: step={step}, time={time:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("--steps", type=int, nargs="*", default=None)
    ap.add_argument("--ntheta", type=int, default=None,
                    help="default: sph_grid.h5's N_theta, else 64")
    ap.add_argument("--nphi", type=int, default=None)
    ap.add_argument("--no-recovery", action="store_true",
                    help="primal Whitney B gather instead of the C0 "
                         "vertex recovery")
    ap.add_argument("--suffix", default="",
                    help="output name suffix: sph<suffix>_XXXXXX.h5")
    args = ap.parse_args()

    mesh = Mesh(os.path.join(args.data_dir, "mesh.h5"))

    n_theta, n_phi = args.ntheta, args.nphi
    grid_path = os.path.join(args.data_dir, "sph_grid.h5")
    if (n_theta is None or n_phi is None) and os.path.exists(grid_path):
        with h5py.File(grid_path, "r") as f:
            n_theta = n_theta or int(f["N_theta"][()])
            n_phi = n_phi or int(f["N_phi"][()])
    n_theta = n_theta or 64
    n_phi = n_phi or 128

    # Per-element geometry: prefer the full mesh file's arrays, compute
    # analytically for 7D sphere-only meshes.
    if hasattr(mesh, "vert_dual_vol"):
        dual_vol = mesh.vert_dual_vol.astype(np.float64)
    else:
        dual_vol = vert_dual_vol(mesh)
    if hasattr(mesh, "hodge1_inv"):
        h1inv = mesh.hodge1_inv.astype(np.float64)
    else:
        h1inv = hodge1_inv(mesh)

    rec = None
    if not args.no_recovery:
        print("Building vertex recovery weights ...")
        rec = build_recovery(mesh)

    if args.steps is not None:
        paths = [
            os.path.join(args.data_dir, f"step_{s:06d}.h5") for s in args.steps
        ]
    else:
        paths = sorted(glob.glob(os.path.join(args.data_dir, "step_*.h5")))

    # Write the grid description alongside (matches sph_grid.h5).
    with h5py.File(
        os.path.join(args.data_dir, f"sph{args.suffix}_grid.h5"), "w"
    ) as f:
        f.create_dataset("N_theta", data=np.int32(n_theta))
        f.create_dataset("N_phi", data=np.int32(n_phi))
        f.create_dataset("N_r", data=np.int32(mesh.N_r))
        f.create_dataset(
            "theta", data=(np.pi * np.arange(n_theta) / (n_theta - 1)))
        f.create_dataset("phi", data=(2 * np.pi * np.arange(n_phi) / n_phi))
        f.create_dataset("radii", data=mesh.radii)

    for p in paths:
        m = re.search(r"step_(\d+)\.h5$", p)
        if m is None:
            continue
        out = os.path.join(
            args.data_dir, f"sph{args.suffix}_{int(m.group(1)):06d}.h5"
        )
        process_step(mesh, p, out, n_theta, n_phi, not args.no_recovery,
                     rec, dual_vol, h1inv)


if __name__ == "__main__":
    main()
