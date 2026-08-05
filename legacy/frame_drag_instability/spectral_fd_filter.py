#!/usr/bin/env python
"""Growth bound with and without a face-space filter inside the W coupling.

Filter: F = I - s * Dg^-1 Lg on the face-adjacency graph (faces sharing an
edge), s = 0.5 -> zero gain at the graph Nyquist. Applied as W @ F.
Also reports the most NEGATIVE eigenvalue of the unfiltered form (spectrum
symmetry = sign-independence of the instability).
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

exec(open(sys.argv[0].replace("spectral_fd_filter", "spectral_load")).read())
meta, d = load(sys.argv[1])
n_he, n_ve = meta["n_he"], meta["n_ve"]
n_tri, n_rect = meta["n_tri"], meta["n_rect"]
ne = n_he + n_ve
nf = n_tri + n_rect
es, bs = meta["e_split"], meta["b_split"]

def csr(rows, cols, vals, shape, col_off=0):
    return sp.csr_matrix(
        (vals, cols.astype(int) + col_off, rows.astype(int)), shape=shape)

d1_tri = csr(d["d1_tri_h_row"], d["d1_tri_h_col"], d["d1_tri_h_val"],
             (n_tri, ne))
d1_rect_h = csr(d["d1_rect_h_row"], d["d1_rect_h_col"], d["d1_rect_h_val"],
                (n_rect, ne))
d1_rect_v = csr(d["d1_rect_v_row"], d["d1_rect_v_col"], d["d1_rect_v_val"],
                (n_rect, ne), col_off=es)
D1 = sp.vstack([d1_tri, d1_rect_h + d1_rect_v]).tocsr()
W_h_tri = csr(d["d1t_h_tri_row"], d["d1t_h_tri_col"], d["fd_h_tri_val"],
              (n_he, nf))
W_h_rect = csr(d["d1t_h_rect_row"], d["d1t_h_rect_col"], d["fd_h_rect_val"],
               (n_he, nf), col_off=bs)
W_v_rect = csr(d["d1t_v_rect_row"], d["d1t_v_rect_col"], d["fd_v_rect_val"],
               (n_ve, nf), col_off=bs)
W = sp.vstack([W_h_tri + W_h_rect, W_v_rect]).tocsr()
h2 = np.concatenate([d["tri_face_hodge2"], d["rect_face_hodge2"]])
e_bnd = np.concatenate([d["h_edge_boundary"], d["v_edge_boundary"]]) != 0
f_bnd = np.concatenate([d["tri_face_boundary"], d["rect_face_boundary"]]) != 0

# face-adjacency graph via shared edges (pattern of |D1| |D1|^T)
P = sp.csr_matrix((np.abs(D1.data), D1.indices, D1.indptr), shape=D1.shape)
Ag = (P @ P.T).tolil()
Ag.setdiag(0)
Ag = Ag.tocsr()
Ag.data[:] = 1.0
deg = np.asarray(Ag.sum(axis=1)).ravel()
Fg = sp.eye(nf) - 0.5 * sp.diags(1.0 / deg) @ (sp.diags(deg) - Ag)

keep_e = ~e_bnd
keep_f = ~f_bnd
h2k = h2[keep_f]
inv_sqrt = sp.diags(1.0 / np.sqrt(h2k))
D1k = D1[keep_f][:, keep_e]

for tag, Wuse in (("unfiltered", W), ("filtered", (W @ Fg).tocsr())):
    Wk = Wuse[keep_e][:, keep_f]
    A = sp.csr_matrix(sp.diags(h2k) @ D1k @ Wk)
    S = (-0.5) * (A + A.T)
    Ssym = inv_sqrt @ S @ inv_sqrt
    hi = eigsh(Ssym, k=3, which="LA", return_eigenvectors=False)
    lo = eigsh(Ssym, k=3, which="SA", return_eigenvectors=False)
    print(f"{tag:>11}: max growth bound {np.sort(hi)[::-1].round(4)}, "
          f"min {np.sort(lo).round(4)}")
