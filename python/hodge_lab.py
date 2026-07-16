"""Hodge lab: test candidate B->H constitutive maps for 2nd-order
consistency using the spurious-curl probe, entirely offline.

Probe: for the EXACT static dipole (curl-free), the discrete Ampere
chain  h1inv * d1t * circ(B)  is pure constitutive-map truncation.
Candidate maps produce circ_f = estimate of int_{f*} H . dl per face;
the baseline diagonal uses h2[f]*B_f (one-point quadrature at an
off-center crossing -> O(h)); the reconstruction candidate fits the
local linear div-free field from ~30 nearby fluxes (validated
vertex-recovery machinery) and integrates it exactly along the dual
segment -> exact for linear fields -> expected O(h^2).

Usage: python hodge_lab.py  (runs L4 and L5; meshes generated via the
GPU IC of vacuum_dipole with Omega=0, kept on disk)
"""
import math
import os
import subprocess
import sys

import h5py
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

sys.path.insert(0, "/home/alex/Projects/Aperture4/python")
import prismatic_recovery as pr

BIN = "/home/alex/Projects/Aperture4/problems/prismatic_dipole/bin/vacuum_dipole"
R_IN, R_OUT = 2.5, 4.0     # probe annulus (interior, away from boundaries)


def log(*a):
    print(*a)
    sys.stdout.flush()


def ensure_mesh(L, NR):
    d = f"Data_lab_L{L}"
    if not os.path.exists(f"{d}/step_000000.h5"):
        cfg = f"""dt = 0.001
max_steps = 1
subdivision_level = {L}
N_r = {NR}
r_min = 1.0
r_max = 45.0
Bp = 1.0
Omega = 0.0
obliquity = 0.0
use_flat_metric = true
damping_length = 0
damping_coef = 0.0
fld_output_interval = 1
output_dir = "{d}"
sph_N_theta = 8
sph_N_phi = 16
"""
        open(f"config_lab_L{L}.toml", "w").write(cfg)
        r = subprocess.run([BIN, "-c", f"config_lab_L{L}.toml"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-300:]
    return d


def probe(L, NR):
    d = ensure_mesh(L, NR)
    mesh = pr.Mesh(f"{d}/mesh.h5")
    with h5py.File(f"{d}/mesh.h5") as f:
        el = f["edge_radial_layer"][()]
        d1t = csr_matrix((f["d1t_val"][()].astype(np.float64),
                          f["d1t_col_idx"][()], f["d1t_row_ptr"][()]),
                         shape=(mesh.N_edges, mesh.N_faces))
        h2 = f["hodge2"][()].astype(np.float64)
        h1i = f["hodge1_inv"][()].astype(np.float64)
    with h5py.File(f"{d}/step_000000.h5") as f0:
        B = f0["B_f"][()].astype(np.float64)

    re = mesh.radii[el]
    probe_edges = np.where((re >= R_IN) & (re < R_OUT))[0]

    # faces adjacent to probe edges
    ind = d1t.indptr
    faces_needed = np.unique(np.concatenate(
        [d1t.indices[ind[e]:ind[e + 1]] for e in probe_edges]))

    # ---- geometry: prism circumcenter directions and radii ----
    v = mesh.sphere_v[mesh.tri_verts]                    # (T,3,3)
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    flip = np.einsum("tx,tx->t", n, v.mean(axis=1)) < 0
    n[flip] *= -1                                        # outward u_circ
    r_mid = 0.5 * (mesh.radii[:-1] + mesh.radii[1:])     # (N_r,)

    # sphere-edge -> adjacent triangles
    edge_tris = -np.ones((mesh.N_edge_s, 2), dtype=int)
    for t in range(mesh.N_tri):
        for e in mesh.tri_edges_s[t]:
            if edge_tris[e, 0] < 0:
                edge_tris[e, 0] = t
            else:
                edge_tris[e, 1] = t

    # ---- recovery fits (per 3D vertex) ----
    rec = pr.build_recovery(mesh, n=4)

    n_tf = (mesh.N_r + 1) * mesh.N_tri

    # segment endpoints + designated vertex per needed face
    GQ = np.array([0.06943184420297371, 0.33000947820757187,
                   0.6699905217924281, 0.9305681557970262])
    GW = np.array([0.1739274225687269, 0.3260725774312731,
                   0.3260725774312731, 0.1739274225687269])

    rows, cols, vals = [], [], []
    for f in faces_needed:
        if f < n_tf:                      # tri face on shell k
            k = f // mesh.N_tri
            t = f - k * mesh.N_tri
            if k == 0 or k == mesh.N_r:
                continue                  # boundary duals: keep diagonal
            u = n[t]
            p0, p1 = r_mid[k - 1] * u, r_mid[k] * u
            svert = mesh.tri_verts[t][0]
            kv = k
            # straight radial segment; Gauss points on the line
            X = p0[None, :] + (p1 - p0)[None, :] * GQ[:, None]
            dX = np.repeat((p1 - p0)[None, :], 4, axis=0)
        else:                             # rect face in layer k
            fi = f - n_tf
            k = fi // mesh.N_edge_s
            e = fi - k * mesh.N_edge_s
            t0, t1 = edge_tris[e]
            u0, u1 = n[t0], n[t1]
            # arc at r_mid[k]; Gauss points along normalized lerp
            P = u0[None, :] * (1 - GQ)[:, None] + u1[None, :] * GQ[:, None]
            Pn = np.linalg.norm(P, axis=1, keepdims=True)
            X = r_mid[k] * P / Pn
            dP = (u1 - u0)[None, :]
            dX = r_mid[k] * (dP - (P / Pn) *
                             np.einsum("qx,qx->q", P / Pn, np.repeat(dP, 4, 0)
                                       )[:, None]) / Pn
            svert = mesh.sphere_edges[e][0]
            kv = k
        vi = kv * mesh.N_vert_s + svert
        x_v = mesh.radii[kv] * mesh.sphere_v[svert]

        # circulation of fitted linear field: chord . B0  +  K : G
        chord = np.einsum("qx,q->x", dX, GW)
        K = np.einsum("qx,qj,q->xj", dX, X - x_v[None, :], GW)  # int dl_i (x-xv)_j
        wB, wG = rec["w_B"][vi], rec["w_G"][vi]                 # (3,nf),(9,nf)
        row = chord @ wB + K.reshape(9) @ wG                    # (nf,)

        # orientation: the dual segment must run along the face's +normal
        # (the diagonal h2 (>0) implicitly assumes this).  GEOMETRIC sign:
        # compare the segment chord with the face normal computed from the
        # stored vertex order (same convention as the validated flux
        # quadrature) — never involve field values here, or sign flips
        # would cancel real error and fake consistency.
        pf = rec["faces"][vi]
        if f < n_tf:
            s = 1.0  # radial chord, outward normal: aligned by construction
        else:
            # face vertex ORDER (not the canonical edge order) fixes the
            # normal: v0=(r_lo, u_a), v1=(r_lo, u_b), normal ~ (u_b-u_a) x r
            sa = int(mesh.rect_face_v0[fi]) % mesh.N_vert_s
            sb = int(mesh.rect_face_v1[fi]) % mesh.N_vert_s
            ua, ub = mesh.sphere_v[sa], mesh.sphere_v[sb]
            um = ua + ub
            um /= np.linalg.norm(um)
            nf = np.cross(ub - ua, um)   # dXu x dXz direction at midpoint
            chord_vec = X[-1] - X[0]
            s = 1.0 if float(chord_vec @ nf) >= 0 else -1.0
        rows.extend([f] * len(pf))
        cols.extend(pf)
        vals.extend((s * row).tolist())

    W = coo_matrix((vals, (rows, cols)),
                   shape=(mesh.N_faces, mesh.N_faces)).tocsr()
    covered = np.zeros(mesh.N_faces, dtype=bool)
    covered[np.unique(rows)] = True

    circ_diag = h2 * B
    circ_corr = np.where(covered, W @ B, circ_diag)

    def spurious(circ):
        curl = h1i * (d1t @ circ)
        absref = h1i * (np.abs(d1t) @ np.abs(circ_diag))
        return (np.linalg.norm(curl[probe_edges]) /
                np.linalg.norm(absref[probe_edges]))

    return spurious(circ_diag), spurious(circ_corr)


if __name__ == "__main__":
    out = {}
    for L, NR in [(4, 51), (5, 102)]:
        d0, d1 = probe(L, NR)
        out[L] = (d0, d1)
        log(f"L{L}: diagonal {d0:.4e}   reconstruction {d1:.4e}")
    log(f"order ratios L4->L5: diagonal {out[4][0]/out[5][0]:.2f}, "
        f"reconstruction {out[4][1]/out[5][1]:.2f}  (2=1st order, 4=2nd)")
