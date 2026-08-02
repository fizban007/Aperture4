#!/usr/bin/env python3
"""Stability probe for the GR off-diagonal Hodge correction (GRPIC_PLAN B1, option a).

Why this is not an eigenvalue sweep
-----------------------------------
The GR solver's semi-discrete generator is

    A = [[ 0,            d1t . aH2 ],
         [ -d1 . aH1,    0         ]]      acting on x = (D~, B)

with aH1 = alpha*Hodge1, aH2 = alpha*Hodge2.  Define the energy inner product
E = blkdiag(aH1, aH2).  Then

    <Ax, x>_E = B^T aH2 d1 aH1 D~  -  D~^T aH1 d1t aH2 B

and since d1t = d1^T exactly (topological incidence), the two terms are
transposes of one another and cancel **iff aH1 and aH2 are symmetric**.  So A
is skew-adjoint in the energy norm, its spectrum is pure imaginary, and the
scheme is neutrally stable -- provided the Hodge operators are SYMMETRIC and
E is POSITIVE DEFINITE.

That turns "is it stable?" into two O(nnz) linear-algebra checks instead of an
ARPACK sweep, and it matches the recorded diagnosis of the earlier failure:
the reconstruction Hodge was unstable because its single-anchor rows were
ASYMMETRIC ("strongly non-normal operator"), not because it was wide.

This script therefore measures, for three Hodge variants:
  1. skew-adjointness residual  |<Ax,x>_E| / (||Ax||_E ||x||_E)  on random x
  2. symmetry defect            ||H - H^T||_inf / ||H||_inf
  3. positive definiteness      lambda_min(H)

Variants: diagonal baseline; SYMMETRIC off-diagonal correction; and a
deliberately ASYMMETRIC version of the same correction, as a positive control
that the probe actually detects the bad case.

Scope: this isolates the Hodge question.  The radial-shift cross terms
(beta x B, beta x D) are omitted -- they are a separate, already-validated
mechanism, and they are not what B1 is about.

Usage:  python3 hodge_lab_gr_offdiag.py [RUNDIR]
        (default problems/prismatic_wald/Data_conv_L3_ana)
"""
import sys
import time

import h5py
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, diags, bmat
from scipy.sparse.linalg import eigsh

A_SPIN = 0.998


def log(*a):
    print(*a)
    sys.stdout.flush()


def load(rundir):
    with h5py.File(f"{rundir}/mesh.h5", "r") as m:
        d = dict(
            N_r=int(m["N_r"][()]), N_tri=int(m["N_tri"][()]),
            N_edge_s=int(m["N_edge_s"][()]), N_vert_s=int(m["N_vert_s"][()]),
            N_edges=int(m["N_edges"][()]), N_faces=int(m["N_faces"][()]),
            L=int(m["L"][()]),
            radii=np.array(m["radii"], dtype=np.float64),
            hodge1_inv=np.array(m["hodge1_inv"], dtype=np.float64),
            hodge2=np.array(m["hodge2"], dtype=np.float64),
            edge_len=np.array(m["edge_length"], dtype=np.float64),
            e0=np.array(m["edge_v0"]), e1=np.array(m["edge_v1"]),
            vx=np.array(m["vert_x"], dtype=np.float64),
            vy=np.array(m["vert_y"], dtype=np.float64),
            vz=np.array(m["vert_z"], dtype=np.float64),
        )
        d["d1"] = csr_matrix((np.array(m["d1_val"], dtype=np.float64),
                              np.array(m["d1_col_idx"]),
                              np.array(m["d1_row_ptr"])),
                             shape=(d["N_faces"], d["N_edges"]))
        d["face_layer"] = np.array(m["face_radial_layer"])
    return d


def edge_geometry(d):
    """Midpoint (r, theta), unit tangent, and phi-hat for every edge."""
    p0 = np.stack([d["vx"][d["e0"]], d["vy"][d["e0"]], d["vz"][d["e0"]]], 1)
    p1 = np.stack([d["vx"][d["e1"]], d["vy"][d["e1"]], d["vz"][d["e1"]]], 1)
    mid = 0.5 * (p0 + p1)
    r = np.linalg.norm(mid, axis=1)
    cth = mid[:, 2] / np.maximum(r, 1e-300)
    sth = np.sqrt(np.maximum(1.0 - cth * cth, 0.0))
    t = p1 - p0
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)
    # phi-hat = z_hat x r_hat, normalized (zero on the axis)
    rho = np.hypot(mid[:, 0], mid[:, 1])
    phihat = np.zeros_like(mid)
    ok = rho > 1e-12
    phihat[ok, 0] = -mid[ok, 1] / rho[ok]
    phihat[ok, 1] = mid[ok, 0] / rho[ok]
    rhat = mid / np.maximum(r, 1e-300)[:, None]
    return r, sth, cth, t, phihat, rhat


def ks_gamma_up_rphi(a, r, sth, cth):
    """gamma^{r phi} and gamma^{rr} for Kerr-Schild spherical."""
    rho2 = r * r + a * a * cth * cth
    gu11 = (a * a + r * r) / rho2 - 2.0 * r / (rho2 + 2.0 * r)
    gu13 = a / rho2
    return gu11, gu13


def build_correction(d, a=A_SPIN, symmetric=True):
    """Sparse off-diagonal Hodge coupling between radial (vertical) edges and
    the azimuthal content of the surrounding horizontal edges.

    Model form: for a vertical edge e, the diagonal Hodge supplies only the
    gamma^rr D_r piece; the missing gamma^{r phi} D_phi is reconstructed from
    the horizontal edges sharing e's endpoints, weighted by how azimuthal each
    of those edges is (|t . phihat|).  Coupling strength is set by the local
    gamma^{r phi}/gamma^{rr} ratio, i.e. the true relative size of the omitted
    term -- so the magnitude is physical, not tuned.
    """
    N_r, Nes, Nvs = d["N_r"], d["N_edge_s"], d["N_vert_s"]
    n_h = (N_r + 1) * Nes
    N_e = d["N_edges"]
    r, sth, cth, t, phihat, rhat = edge_geometry(d)
    gu11, gu13 = ks_gamma_up_rphi(a, r, sth, cth)
    ratio = gu13 / np.maximum(gu11, 1e-300)          # gamma^rphi / gamma^rr

    # vertex -> incident horizontal edges
    h_ids = np.arange(n_h)
    v_of_h0, v_of_h1 = d["e0"][:n_h], d["e1"][:n_h]
    from collections import defaultdict
    fan = defaultdict(list)
    for e, (va, vb) in enumerate(zip(v_of_h0, v_of_h1)):
        fan[va].append(e)
        fan[vb].append(e)

    rows, cols, vals = [], [], []
    for e in range(n_h, N_e):
        va, vb = d["e0"][e], d["e1"][e]
        nbrs = fan.get(va, []) + fan.get(vb, [])
        if not nbrs:
            continue
        nb = np.asarray(nbrs)
        # how azimuthal each neighbour is, at ITS OWN location
        w = np.abs(np.einsum("ij,ij->i", t[nb], phihat[nb]))
        s = w.sum()
        if s <= 0:
            continue
        w = w / s
        # scale: |e*| for the vertical edge (= edge_len/hodge1_inv) times the
        # physical ratio of the omitted term
        scale = ratio[e] * d["edge_len"][e] / max(d["hodge1_inv"][e], 1e-300)
        rows.extend([e] * len(nb))
        cols.extend(nb.tolist())
        vals.extend((scale * w).tolist())
    C = coo_matrix((vals, (rows, cols)), shape=(N_e, N_e)).tocsr()
    return C if not symmetric else (C + C.T) * 0.5


def alpha_of(d, which):
    """Lapse at edge midpoints (which='e') or face centres (which='f')."""
    if which == "e":
        r, sth, cth, *_ = edge_geometry(d)
    else:
        # face radial layer -> shell radius is adequate for a stability probe
        r = d["radii"][np.clip(d["face_layer"], 0, d["N_r"])]
        cth = np.zeros_like(r)
    rho2 = r * r + A_SPIN * A_SPIN * cth * cth
    return 1.0 / np.sqrt(1.0 + 2.0 * r / rho2)


def probe(d, H1, label, aH2):
    N_e, N_f = d["N_edges"], d["N_faces"]
    ae = alpha_of(d, "e")
    # DESIGN CONSTRAINT surfaced by this probe: fold the lapse in
    # SYMMETRICALLY, alpha*H1 -> sqrt(alpha) H1 sqrt(alpha).  Writing
    # diags(alpha) @ H1 (lapse on the left, as the current code effectively
    # does) is symmetric only while H1 is diagonal -- the moment an
    # off-diagonal term exists it breaks symmetry and the scheme loses its
    # energy conservation.  The continuum object alpha*gamma^ij is symmetric
    # in (i,j), and the split form reproduces diag(alpha*h) exactly on the
    # diagonal part, so this costs nothing for the existing scheme.
    sa = diags(np.sqrt(ae))
    aH1 = (sa @ H1 @ sa).tocsr()
    dif = (aH1 - aH1.T).tocoo()
    scale = np.abs(aH1.data).max() if aH1.nnz else 1.0
    sym_defect = (np.abs(dif.data).max() if dif.nnz else 0.0) / max(scale, 1e-300)

    d1 = d["d1"]
    rng = np.random.default_rng(0)
    res = []
    for _ in range(3):
        D = rng.standard_normal(N_e)
        B = rng.standard_normal(N_f)
        # A x
        dD = d1.T @ (aH2 @ B)
        dB = -(d1 @ (aH1 @ D))
        # Energy rate  dW/dt = D^T (aH1 dD) + B^T (aH2 dB)  for the physical
        # energy W = 1/2 (D^T aH1 D + B^T aH2 B).
        #
        # ORDER MATTERS.  Writing this as dD^T (aH1 D) instead gives
        # D^T aH1^T dD, and the two terms then cancel IDENTICALLY whatever
        # aH1 is -- a vacuous test that passes even the asymmetric control.
        # With this ordering the residual reduces to
        #   D^T (aH1 - aH1^T) d1^T aH2 B,
        # which vanishes iff aH1 is symmetric.  That is the whole point.
        num = D @ (aH1 @ dD) + B @ (aH2 @ dB)
        nAx = np.sqrt(abs(dD @ (aH1 @ dD) + dB @ (aH2 @ dB)))
        nx = np.sqrt(abs(D @ (aH1 @ D) + B @ (aH2 @ B)))
        res.append(abs(num) / max(nAx * nx, 1e-300))
    return sym_defect, float(np.max(res))


def main():
    rundir = (sys.argv[1] if len(sys.argv) > 1 else
              "/home/alex/Projects/Aperture4/problems/prismatic_wald/"
              "Data_conv_L3_ana")
    t0 = time.time()
    d = load(rundir)
    log(f"mesh L={d['L']}  N_r={d['N_r']}  edges={d['N_edges']}  "
        f"faces={d['N_faces']}   ({time.time()-t0:.1f}s)")

    aH2 = diags(alpha_of(d, "f") * d["hodge2"])
    H1_diag = diags(1.0 / np.maximum(d["hodge1_inv"], 1e-300))

    t1 = time.time()
    Csym = build_correction(d, symmetric=True)
    Casym = build_correction(d, symmetric=False)
    log(f"correction built: {Csym.nnz} nnz  ({time.time()-t1:.1f}s)")

    variants = [("diagonal (current)", H1_diag),
                ("+ SYMMETRIC off-diagonal", H1_diag + Csym),
                ("+ ASYMMETRIC (control)", H1_diag + Casym)]

    log(f"\n{'variant':>26} {'sym defect':>12} {'skew resid':>12} {'verdict':>10}")
    log("-" * 64)
    for name, H1 in variants:
        sd, sk = probe(d, H1, name, aH2)
        verdict = "STABLE" if sk < 1e-10 else "UNSTABLE"
        log(f"{name:>26} {sd:12.3e} {sk:12.3e} {verdict:>10}")

    # positive definiteness of the symmetric candidate (needed for E to be a
    # norm at all); k small, shift-invert-free smallest-algebraic.
    log("\npositive-definiteness of alpha*H1 (energy metric must be SPD):")
    for name, H1 in variants[:2]:
        sa = diags(np.sqrt(alpha_of(d, "e")))
        M = (sa @ H1 @ sa).tocsc()
        M = (M + M.T) * 0.5
        t2 = time.time()
        lo = eigsh(M, k=1, which="SA", return_eigenvectors=False,
                   maxiter=5000, tol=1e-6)[0]
        log(f"  {name:>26}  lambda_min = {lo:.6e}   ({time.time()-t2:.1f}s)")

    log(f"\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
