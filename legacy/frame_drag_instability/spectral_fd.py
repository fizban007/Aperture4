#!/usr/bin/env python
"""Energy-norm growth bound of the frame-drag Faraday coupling.

Semi-discrete system: dE/dt = h1inv d1t h2 B ; dB/dt = -d1(E + W B).
With U = 1/2 E' h1inv^-1 E + 1/2 B' h2 B:  dU/dt = -B' (h2 d1 W) B.
Growth rate bound gamma = max eig of  -sym(h2 d1 W) x = lam h2 x.
Boundary-flagged edges/faces (prescribed by BCs) are removed.
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def load(fn):
    d = {}
    with open(fn) as f:
        first = f.readline().split()
        meta = dict(zip(
            ["L", "N_r", "r_max", "n_he", "n_ve", "n_tri", "n_rect",
             "e_split", "b_split"],
            [float(x) if "." in x else int(x) for x in first[1:]]))
        while True:
            hdr = f.readline().split()
            if not hdr:
                break
            name, n = hdr[0], int(hdr[1])
            a = np.fromfile(f, count=n, sep="\n")
            d[name] = a
    return meta, d

meta, d = load(sys.argv[1])
n_he, n_ve = meta["n_he"], meta["n_ve"]
n_tri, n_rect = meta["n_tri"], meta["n_rect"]
ne = n_he + n_ve
nf = n_tri + n_rect
es, bs = meta["e_split"], meta["b_split"]
N_r = meta["N_r"]
NEs = n_he // (N_r + 1)
NTs = n_tri // (N_r + 1)
NVs = n_ve // N_r

def csr(rows, cols, vals, shape, col_off=0):
    return sp.csr_matrix(
        (vals, cols.astype(int) + col_off, rows.astype(int)), shape=shape)

# d1: nf x ne
d1_tri = csr(d["d1_tri_h_row"], d["d1_tri_h_col"], d["d1_tri_h_val"],
             (n_tri, ne))
d1_rect_h = csr(d["d1_rect_h_row"], d["d1_rect_h_col"], d["d1_rect_h_val"],
                (n_rect, ne))
d1_rect_v = csr(d["d1_rect_v_row"], d["d1_rect_v_col"], d["d1_rect_v_val"],
                (n_rect, ne), col_off=es)
D1 = sp.vstack([d1_tri, d1_rect_h + d1_rect_v]).tocsr()

# W: ne x nf   (h-edge rows: tri cols + rect cols(offset bs); v rows: rect)
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
print(f"# L={meta['L']} N_r={N_r}: {ne} edges ({e_bnd.sum()} bnd), "
      f"{nf} faces ({f_bnd.sum()} bnd)")

keep_e = ~e_bnd
keep_f = ~f_bnd
D1k = D1[keep_f][:, keep_e]
Wk = W[keep_e][:, keep_f]
h2k = h2[keep_f]

A = sp.csr_matrix(sp.diags(h2k) @ D1k @ Wk)
S = (-0.5) * (A + A.T)
# generalized symmetric eig: S x = lam h2 x  ->  h2^-1/2 S h2^-1/2
inv_sqrt = sp.diags(1.0 / np.sqrt(h2k))
Ssym = inv_sqrt @ S @ inv_sqrt
vals, vecs = eigsh(Ssym, k=6, which="LA")
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]
print("top growth-bound eigenvalues (1/time):", np.round(vals, 4))

# localization of the top eigenvector: energy by radial shell
kept_idx = np.where(keep_f)[0]
tri_shell = np.arange(n_tri) // NTs
rect_shell = bs + 0 * np.arange(n_rect)  # placeholder
# rect face shell: rect f -> shell f // (NEs_sphere) where rects per shell
NRs = n_rect // N_r
rect_shell = np.arange(n_rect) // NRs
f_shell = np.concatenate([tri_shell, rect_shell])[kept_idx]
v = vecs[:, 0] ** 2
prof = np.zeros(N_r + 1)
for k in range(N_r + 1):
    prof[k] = v[f_shell == k].sum()
print("top-mode energy by shell (k: frac):")
print("  " + " ".join(f"{k}:{prof[k]:.3f}" for k in range(min(N_r + 1, 14))
                      if prof[k] > 0.005))
r_of_k = 1.0 * (meta["r_max"]) ** (np.arange(N_r + 1) / N_r)
kpk = int(np.argmax(prof))
print(f"peak shell k={kpk} (r ~ {r_of_k[kpk]:.3f} if log-spaced)")
