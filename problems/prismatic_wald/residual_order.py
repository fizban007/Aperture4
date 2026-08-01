#!/usr/bin/env python3
"""Normalization-free convergence order of the DEC operators at the on-shell
Kerr-Wald state.

The README warns that analyze_drift.py's dD/dt over |D| misreads the order by
one: D~ is a dual 2-cochain (~h^2) while dD is a signed sum of ~h^1 terms, so
a *constant* dD/D already means first order.  The fix it prescribes is a
cancellation ratio -- the signed sum over the sum of magnitudes:

    C = || A x || / || |A| |x| ||

which is dimensionless, normalization-free, and O(h^p) when the operator is
p-th order accurate at the exact solution.

Faraday:  A = d1,  x = E_aux    (should vanish: dB/dt = -d1 E_aux = 0)
Ampere:   A = d1t, x = H_aux    (NOT hodge2 * H_aux -- the GR solver already
          folds hodge2 into H_aux: dec_field_solver_gr_ks_impl.hpp:246 sets
          H_aux[f] = face_alpha[f] * hodge2[f] * B[f] + shift, and the update
          at :300 is a bare d1t_val * H_aux.  The class comment at
          dec_field_solver_gr_ks.h:87 still shows the pre-refactor form with
          an explicit hodge2 and is stale.  The FLAT solver is different --
          dec_solver_dist.h:489 does d1t_val * face_hodge2[f] * B_f[f] -- so
          the flat analogue of x really is hodge2 * B.)

Run on the _ana runs of the convergence family (field_spin = bh_spin), which
sample the analytic stationary solution.

Usage:  residual_order.py Data_conv_L3_ana Data_conv_L4_ana Data_conv_L5_ana
"""
import argparse
import os
import sys

import h5py
import numpy as np
from scipy.sparse import csr_matrix


def load(rundir):
    with h5py.File(os.path.join(rundir, "mesh.h5"), "r") as m:
        N_r = int(m["N_r"][()])
        N_tri = int(m["N_tri"][()])
        N_edge_s = int(m["N_edge_s"][()])
        L = int(m["L"][()])
        N_faces = int(m["N_faces"][()])
        N_edges = int(m["N_edges"][()])
        d1 = csr_matrix((np.array(m["d1_val"], dtype=np.float64),
                         np.array(m["d1_col_idx"]),
                         np.array(m["d1_row_ptr"])),
                        shape=(N_faces, N_edges))
        d1t = csr_matrix((np.array(m["d1t_val"], dtype=np.float64),
                          np.array(m["d1t_col_idx"]),
                          np.array(m["d1t_row_ptr"])),
                         shape=(N_edges, N_faces))
        hodge2 = np.array(m["hodge2"], dtype=np.float64)
        face_layer = np.array(m["face_radial_layer"])
        edge_layer = np.array(m["edge_radial_layer"])
    with h5py.File(os.path.join(rundir, "ic_aux.h5"), "r") as f:
        E_aux = np.array(f["E_aux"], dtype=np.float64)
        H_aux = np.array(f["H_aux"], dtype=np.float64)
    return dict(L=L, N_r=N_r, N_tri=N_tri, N_edge_s=N_edge_s, d1=d1, d1t=d1t,
                hodge2=hodge2, E_aux=E_aux, H_aux=H_aux,
                face_layer=face_layer, edge_layer=edge_layer,
                N_faces=N_faces, N_edges=N_edges)


def cancellation(A, x, row_mask=None):
    """|| A x || / || |A| |x| || over the selected rows (RMS norms)."""
    num = A @ x
    den = abs(A) @ np.abs(x)
    if row_mask is not None:
        num, den = num[row_mask], den[row_mask]
    den_rms = np.sqrt((den ** 2).mean())
    if den_rms == 0.0:
        return np.nan
    return np.sqrt((num ** 2).mean()) / den_rms


def measure(rundir, drop_outer):
    d = load(rundir)
    N_r = d["N_r"]
    fmask = None
    emask = None
    if drop_outer:
        # Mask BOTH radial ends, not just the outer one.  Each end carries a
        # ghost layer whose half-open dual loops make the residual identity
        # legitimately fail (roadmap: "masked by BCs in every current use").
        # Those rows are O(1) and, being a ~1/N_r fraction, contribute a
        # slowly-decaying term that drags the global order well below the
        # bulk value -- Ampere reads 0.56 unmasked vs 0.98 in the bulk.
        # Vertical edges/rect faces at layer k span k..k+1, so they need
        # k+1 <= N_r - 1 as well.
        nh = (N_r + 1) * d["N_edge_s"]
        n_tri_all = d["N_faces"] - N_r * d["N_edge_s"]
        is_h = np.arange(d["N_edges"]) < nh
        is_tri = np.arange(d["N_faces"]) < n_tri_all
        el, fl = d["edge_layer"], d["face_layer"]
        emask = np.where(is_h, (el >= 1) & (el <= N_r - 1),
                         (el >= 1) & (el + 1 <= N_r - 1))
        fmask = np.where(is_tri, (fl >= 1) & (fl <= N_r - 1),
                         (fl >= 1) & (fl + 1 <= N_r - 1))
    c_far = cancellation(d["d1"], d["E_aux"], fmask)
    c_amp = cancellation(d["d1t"], d["H_aux"], emask)
    return d["L"], c_far, c_amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundirs", nargs="+")
    args = ap.parse_args()

    for drop in (False, True):
        tag = ("BULK only -- ghost shell masked at both radial ends" if drop
               else "all shells, including both ghost layers")
        print(f"\n=== Cancellation ratio C = ||A x|| / || |A| |x| ||  — {tag} ===")
        print(f"  {'L':>3} {'Faraday d1.E_aux':>18} {'order':>7} "
              f"{'Ampere d1t.H_aux':>21} {'order':>7}")
        prev = None
        for rd in args.rundirs:
            L, cf, ca = measure(rd, drop)
            if prev is None:
                print(f"  {L:>3} {cf:18.6e} {'--':>7} {ca:21.6e} {'--':>7}")
            else:
                of = np.log2(prev[1] / cf)
                oa = np.log2(prev[2] / ca)
                print(f"  {L:>3} {cf:18.6e} {of:+7.2f} {ca:21.6e} {oa:+7.2f}")
            prev = (L, cf, ca)


if __name__ == "__main__":
    main()
