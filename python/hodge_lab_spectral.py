"""Spectral lab for the reconstruction Hodge: assemble the full leapfrog
step operator at small L as a scipy LinearOperator and compute its
dominant eigenvalues, for the diagonal baseline, the single-anchor
reconstruction (the form measured unstable in the solver), and the
anchor-AVERAGED reconstruction:

  tri face  : rows averaged over its 3 vertices (same shell)
  rect face : averaged over 2 sphere-edge endpoints x 2 shells (k, k+1)
  h-edge    : averaged over its 2 endpoints
  v-edge    : averaged over its 2 end shells (k, k+1)

The radial anchor pairs center the stencils of the radially-spanning
elements — the single-anchor rows are radially lopsided, and the solver
blowup was localized at small r where the relative radial gradient is
largest.

Usage: python hodge_lab_spectral.py [L]   (default 2; 3 for confirmation)
"""
import math
import sys

import h5py
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.linalg import LinearOperator, eigs

sys.path.insert(0, "/home/alex/Projects/Aperture4/python")
import prismatic_recovery as pr
import hodge_lab
from hodge_lab_chain import Lab


def log(*a):
    print(*a)
    sys.stdout.flush()


def assemble(lab, average):
    """Full sparse W2 (faces x faces) and W1 (edges x edges); diagonal
    fallback outside the safe interior ranges."""
    mesh = lab.mesh
    n_tf = lab.n_tf
    n_h = lab.n_h
    N_r = mesh.N_r

    rows, cols, vals = [], [], []
    diag_faces = []

    def add_row(f, contribs):
        # contribs: list of (gidx array, coef array, weight)
        for gidx, coef, w in contribs:
            rows.extend([f] * len(gidx))
            cols.extend(gidx.tolist())
            vals.extend((w * coef).tolist())

    # ---- W2 ----
    for f in range(mesh.N_faces):
        if f < n_tf:
            k = f // mesh.N_tri
            t = f - k * mesh.N_tri
            if k < 1 or k > N_r - 1:
                diag_faces.append(f)
                continue
            anchors = ([(int(mesh.tri_verts[t][0]), k)] if not average else
                       [(int(v), k) for v in mesh.tri_verts[t]])
        else:
            fi = f - n_tf
            k = fi // mesh.N_edge_s
            e = fi - k * mesh.N_edge_s
            kmax = (N_r - 2) if not average else (N_r - 2)
            if k < 1 or k > kmax:
                diag_faces.append(f)
                continue
            s0, s1 = (int(x) for x in mesh.sphere_edges[e])
            anchors = ([(s0, k)] if not average else
                       [(s0, k), (s1, k), (s0, min(k + 1, N_r - 1)),
                        (s1, min(k + 1, N_r - 1))])
        contribs = []
        w = 1.0 / len(anchors)
        for s, kv in anchors:
            gidx, coef = row_w2(lab, f, s, kv)
            contribs.append((gidx, coef, w))
        add_row(f, contribs)

    W2 = coo_matrix((vals, (rows, cols)),
                    shape=(mesh.N_faces, mesh.N_faces)).tocsr()
    dvec = np.zeros(mesh.N_faces)
    dvec[diag_faces] = lab.h2[diag_faces]
    W2 = W2 + diags(dvec)

    # ---- W1 ----
    rows, cols, vals = [], [], []
    diag_edges = []
    for ei in range(mesh.N_edges):
        if ei < n_h:
            k = ei // mesh.N_edge_s
            e = ei - k * mesh.N_edge_s
            if k < 2 or k > N_r - 2:
                diag_edges.append(ei)
                continue
            s0, s1 = (int(x) for x in mesh.sphere_edges[e])
            anchors = [(s0, k)] if not average else [(s0, k), (s1, k)]
        else:
            li = ei - n_h
            k = li // mesh.N_vert_s
            s = li - k * mesh.N_vert_s
            kmax = (N_r - 2) if not average else (N_r - 3)
            if k < 2 or k > kmax:
                diag_edges.append(ei)
                continue
            anchors = [(s, k)] if not average else [(s, k), (s, k + 1)]
        contribs = []
        w = 1.0 / len(anchors)
        for s2, kv in anchors:
            gidx, coef = row_w1(lab, ei, s2, kv)
            contribs.append((gidx, coef, w))
        for gidx, coef, ww in contribs:
            rows.extend([ei] * len(gidx))
            cols.extend(gidx.tolist())
            vals.extend((ww * coef).tolist())
    W1 = coo_matrix((vals, (rows, cols)),
                    shape=(mesh.N_edges, mesh.N_edges)).tocsr()
    dvec = np.zeros(mesh.N_edges)
    dvec[diag_edges] = lab.h1i[diag_edges]
    W1 = W1 + diags(dvec)
    return W2, W1


def row_w2(lab, f, s, kv):
    """W2 row of face f anchored at (s, kv): columns + coefficients."""
    mesh = lab.mesh
    kv = min(max(kv, 1), mesh.N_r - 1)
    gidx, W = lab.bfit(kv, s)
    x_v = mesh.radii[kv] * mesh.sphere_v[s]
    GQ, GW = lab_GQ, lab_GW
    if f < lab.n_tf:
        k = f // mesh.N_tri
        t = f - k * mesh.N_tri
        u = lab.circ_dir[t]
        p0, p1 = lab.r_mid[k - 1] * u, lab.r_mid[k] * u
        X = p0[None, :] + (p1 - p0)[None, :] * GQ[:, None]
        dX = np.repeat((p1 - p0)[None, :], len(GQ), axis=0)
        sgn = 1.0
    else:
        fi = f - lab.n_tf
        k = fi // mesh.N_edge_s
        e = fi - k * mesh.N_edge_s
        t0, t1 = lab.edge_tris[e]
        u0, u1 = lab.circ_dir[t0], lab.circ_dir[t1]
        P = u0[None, :] * (1 - GQ)[:, None] + u1[None, :] * GQ[:, None]
        Pn = np.linalg.norm(P, axis=1, keepdims=True)
        X = lab.r_mid[k] * P / Pn
        dX = lab.r_mid[k] * ((u1 - u0)[None, :]
                             - (P / Pn) * ((P / Pn) @ (u1 - u0))[:, None]) / Pn
        sa = int(mesh.rect_face_v0[fi]) % mesh.N_vert_s
        sb = int(mesh.rect_face_v1[fi]) % mesh.N_vert_s
        ua, ub = mesh.sphere_v[sa], mesh.sphere_v[sb]
        um = ua + ub
        um /= np.linalg.norm(um)
        nf = np.cross(ub - ua, um)
        sgn = 1.0 if float((X[-1] - X[0]) @ nf) >= 0 else -1.0
    chord = np.einsum("qx,q->x", dX, GW)
    K = np.einsum("qx,qj,q->xj", dX, X - x_v[None, :], GW)
    coef = sgn * (np.concatenate([chord, K.reshape(9)]) @ W)
    return gidx, coef


def row_w1(lab, ei, s, kv):
    mesh = lab.mesh
    kv = min(max(kv, 2), mesh.N_r - 2)
    eids, W = lab.pfit(kv, s)
    x_v = mesh.radii[kv] * mesh.sphere_v[s]
    GQ, GW = lab_GQ, lab_GW
    if ei < lab.n_h:
        k = ei // mesh.N_edge_s
        e = ei - k * mesh.N_edge_s
        s0, s1 = mesh.sphere_edges[e]
        u0, u1 = mesh.sphere_v[s0], mesh.sphere_v[s1]
        r = mesh.radii[k]
        P = u0[None, :] * (1 - GQ)[:, None] + u1[None, :] * GQ[:, None]
        Pn = np.linalg.norm(P, axis=1, keepdims=True)
        X = r * P / Pn
        dX = r * ((u1 - u0)[None, :]
                  - (P / Pn) * ((P / Pn) @ (u1 - u0))[:, None]) / Pn
    else:
        li = ei - lab.n_h
        k = li // mesh.N_vert_s
        sv = li - k * mesh.N_vert_s
        u = mesh.sphere_v[sv]
        ra, rb = mesh.radii[k], mesh.radii[k + 1]
        rq = ra + (rb - ra) * GQ
        X = rq[:, None] * u[None, :]
        dX = np.repeat((rb - ra) * u[None, :], len(GQ), axis=0)
    chord = np.einsum("qx,q->x", dX, GW)
    K = np.einsum("qx,qj,q->xj", dX, X - x_v[None, :], GW)
    coef = np.concatenate([chord, K.reshape(9)]) @ W
    return eids, coef


def spectral_radius(mesh, d1, W2, W1, d1t, dt, label):
    Ne, Nf = mesh.N_edges, mesh.N_faces

    def step(x):
        E = x[:Ne]
        B = x[Ne:]
        Bn = B - dt * (d1 @ E)
        En = E + dt * (W1 @ (d1t @ (W2 @ Bn)))
        return np.concatenate([En, Bn])

    G = LinearOperator((Ne + Nf, Ne + Nf), matvec=step)
    vals = eigs(G, k=12, which="LM", return_eigenvectors=False,
                maxiter=5000, tol=1e-8)
    rho = np.abs(vals).max()
    growth = math.log(rho) / dt if rho > 1 else 0.0
    log(f"  {label}: max|mu| = {rho:.8f}  -> growth {growth:.3f} /time")
    return rho


def main(L=2):
    NR = {2: 13, 3: 26}[L]
    d = hodge_lab.ensure_mesh(L, NR)
    lab = Lab(L, NR)
    mesh = lab.mesh
    with h5py.File(f"{d}/mesh.h5") as f:
        d1 = csr_matrix((f["d1_val"][()].astype(np.float64),
                         f["d1_col_idx"][()], f["d1_row_ptr"][()]),
                        shape=(mesh.N_faces, mesh.N_edges))
    P = 2 * math.pi / 0.2
    dt = P / (400 * 2 ** (L - 2))   # same CFL fraction as the ladder

    log(f"L{L}: assembling operators...")
    log("  [diagonal]")
    W2d = diags(lab.h2)
    W1d = diags(lab.h1i)
    spectral_radius(mesh, d1, W2d, W1d, lab.d1t, dt, "diagonal      ")
    for avg in (False, True):
        W2, W1 = assemble(lab, avg)
        name = "anchor-avg    " if avg else "single-anchor "
        spectral_radius(mesh, d1, W2, W1, lab.d1t, dt, name)


lab_GQ = np.array([0.1127016653792583, 0.5, 0.8872983346207417])
lab_GW = np.array([5.0 / 18, 8.0 / 18, 5.0 / 18])

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
