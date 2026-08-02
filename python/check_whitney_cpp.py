#!/usr/bin/env python3
"""Cross-check the C++ Whitney Galerkin Hodge port against the validated lab.

Three checks, in order of strictness:

  1. MATRIX EQUALITY.  Re-assemble M1, M1a (alpha inside quadrature) and
     M2a globally with the exact loops of hodge_lab_whitney.py (the
     validated reference) and compare entrywise against the CSR operators
     the C++ code dumped to whitney.h5.  This catches every porting error
     directly: signs, orientation, quadrature, metric forms.
     Expected: relative max diff at rounding level (~1e-12; the only
     difference is theta from acos(vz) vs the stored vert_theta).

  2. GAUSS FLUX through r = const (continuum truth EXACTLY 0).  With the
     Whitney IC the stored D~ already is M1 . D_primal, so the projection
     is a plain sum of D~ over the vertical edges of a shell.
     Expected at L3 (lab): 4.579e-2 / -6.443e-2 / -6.961e-2 at
     r = 1.106 / 3.093 / 8.016; orders +1.90 / +1.98 / +2.00 across L.

  3. NEAR-HORIZON MAGNETIC ENERGY vs 2-D quadrature ground truth,
     accumulated per prism with the local M2a (alpha inside), using the
     C++ B cochain.  Expected: +0.631% / +0.153% / +0.039% at L3/4/5.

Usage: check_whitney_cpp.py [L ...]     (default: 3)
Reads problems/prismatic_wald/Data_conv_L{L}_ana_whitney.
"""
import sys

import h5py
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

sys.path.insert(0, "/home/alex/Projects/Aperture4/python")
import hodge_lab_whitney as lab

BASE = ("/home/alex/Projects/Aperture4/problems/prismatic_wald/"
        "Data_conv_L{}_ana_whitney")
RLO, RHI = 1.1, 1.5


def log(*a):
    print(*a)
    sys.stdout.flush()


def load(L):
    d = {}
    with h5py.File(BASE.format(L) + "/mesh.h5", "r") as m:
        for k in ("N_r", "N_tri", "N_edge_s", "N_vert_s", "N_edges",
                  "N_faces"):
            d[k] = int(m[k][()])
        d["radii"] = np.array(m["radii"], dtype=np.float64)
        d["tvs"] = np.array(m["tri_verts"]).reshape(-1, 3)
        d["te"] = np.array(m["tri_edges_s"]).reshape(-1, 3)
        d["tes"] = np.array(m["tri_edge_signs"]).reshape(-1, 3)
        # theta exactly as the C++ assembly derives it: acos(clamped vz).
        d["vth"] = np.arccos(np.clip(
            np.array(m["sphere_vz"], dtype=np.float64), -1.0, 1.0))
        # phi is shell-independent; vert_phi[:N_vert_s] = sphere_phi.
        d["vph"] = np.array(m["vert_phi"], dtype=np.float64)[:d["N_vert_s"]]
    with h5py.File(BASE.format(L) + "/ic_aux.h5", "r") as f:
        d["D"] = np.array(f["D"], dtype=np.float64)
        d["B"] = np.array(f["B"], dtype=np.float64)
        d["D_primal"] = (np.array(f["D_primal"], dtype=np.float64)
                         if "D_primal" in f else None)
        d["whitney"] = int(f["whitney_hodge"][()]) if "whitney_hodge" in f \
            else 0
    return d


def load_cpp_csr(L):
    path = BASE.format(L) + "/whitney.h5"
    out = {}
    with h5py.File(path, "r") as f:
        n_e = int(f["N_edges"][()])
        n_f = int(f["N_faces"][()])
        out["M1"] = csr_matrix(
            (np.array(f["m1_val"]), np.array(f["m1_col_idx"]),
             np.array(f["m1_row_ptr"])), shape=(n_e, n_e))
        out["M1a"] = csr_matrix(
            (np.array(f["m1a_val"]), np.array(f["m1_col_idx"]),
             np.array(f["m1_row_ptr"])), shape=(n_e, n_e))
        out["M2a"] = csr_matrix(
            (np.array(f["m2a_val"]), np.array(f["m2_col_idx"]),
             np.array(f["m2_row_ptr"])), shape=(n_f, n_f))
        out["jac"] = np.array(f["m1_jacobi"])
        if "c1_val" in f:
            out["C1"] = csr_matrix(
                (np.array(f["c1_val"]), np.array(f["c1_col_idx"]),
                 np.array(f["c1_row_ptr"])), shape=(n_e, n_f))
            out["C1t"] = csr_matrix(
                (np.array(f["c1t_val"]), np.array(f["c1t_col_idx"]),
                 np.array(f["c1t_row_ptr"])), shape=(n_f, n_e))
    return out


def assemble_reference(L, d):
    """Global sparse M1 / M1a / M2a with hodge_lab_whitney's exact loops.

    Differences from lab.build(): assembles global COO instead of applying
    per prism; adds the alpha-inside-quadrature variants; folds the
    rect-face orientation (-1) into the 2-form basis the way the C++ port
    does (equivalent to the lab's sgn=1 energy variant).
    """
    Nr, Ntri, Nes, Nvs = d["N_r"], d["N_tri"], d["N_edge_s"], d["N_vert_s"]
    Ne, Nf = d["N_edges"], d["N_faces"]
    Nh = (Nr + 1) * Nes
    ntf = (Nr + 1) * Ntri
    tvs, te, tes = d["tvs"], d["te"], d["tes"]
    th_v = d["vth"][:Nvs]
    ph_v = d["vph"][:Nvs]

    U = np.zeros((Ntri, 3, 2))
    U[:, :, 0] = th_v[tvs]
    p0 = ph_v[tvs[:, 0]]
    U[:, :, 1] = p0[:, None] + lab.wrap(ph_v[tvs] - p0[:, None])
    M = np.ones((Ntri, 3, 3))
    M[:, :, 1:] = U
    C = np.linalg.inv(M)
    grad = np.transpose(C[:, 1:, :], (0, 2, 1))
    cross = np.cross(U[:, 1] - U[:, 0], U[:, 2] - U[:, 0])
    Ac = 0.5 * np.abs(cross)
    sgnA = np.sign(cross)
    loc = [(0, 1), (1, 2), (2, 0)]
    es = tes.astype(np.float64)

    r1_, c1_, x1_, x1a_ = [], [], [], []
    r2_, c2_, x2_ = [], [], []
    rc_, cc_, xc_ = [], [], []      # C1 (edge x face) shift coupling
    rc2_, cc2_, xc2_ = [], [], []   # C2 (face x edge), independent formula
    en_acc = 0.0
    rng = [1e9, -1e9]
    Bf = d["B"]
    A = lab.A

    def beta_r(r, th):
        c, s = np.cos(th), np.sin(th)
        rho2 = r * r + A * A * c * c
        Z = 2.0 * r / rho2
        return Z / (1.0 + Z)

    for k in range(Nr):
        r0, r1 = d["radii"][k], d["radii"][k + 1]
        dr = r1 - r0
        Eidx = np.concatenate([k * Nes + te, (k + 1) * Nes + te,
                               Nh + k * Nvs + tvs], 1)          # (Ntri, 9)
        Fidx = np.concatenate(
            [(k * Ntri + np.arange(Ntri))[:, None],
             ((k + 1) * Ntri + np.arange(Ntri))[:, None],
             ntf + k * Nes + te], 1)                            # (Ntri, 5)
        M1 = np.zeros((Ntri, 9, 9))
        M1a = np.zeros((Ntri, 9, 9))
        M2a = np.zeros((Ntri, 5, 5))
        C1 = np.zeros((Ntri, 9, 5))
        C2 = np.zeros((Ntri, 5, 9))
        for z, wz in zip(lab.GZ, lab.GW):
            r = r0 + z * dr
            for lam, wt in zip(lab.TRI_L, lab.TRI_W):
                th = lam @ U[:, :, 0].T
                g = lab.metric(r, th)
                W = np.zeros((Ntri, 9, 3))
                Bd = np.zeros((Ntri, 5, 3))
                for j, (a_, b_) in enumerate(loc):
                    w2 = lam[a_] * grad[:, b_, :] - lam[b_] * grad[:, a_, :]
                    W[:, j, 1:] = w2 * (1 - z) * es[:, j][:, None]
                    W[:, 3 + j, 1:] = w2 * z * es[:, j][:, None]
                    # C++ port: rect orientation -1 folded into the basis.
                    rot = np.stack([-w2[:, 1], w2[:, 0]], 1) * sgnA[:, None]
                    Bd[:, 2 + j, 1:] = -rot / dr * es[:, j][:, None]
                for i in range(3):
                    W[:, 6 + i, 0] = lam[i] / dr
                Bd[:, 0, 0] = (1 - z) / Ac
                Bd[:, 1, 0] = z / Ac
                gu = np.zeros((Ntri, 3, 3))
                gu[:, 0, 0] = g["gu_rr"]
                gu[:, 1, 1] = g["gu_tt"]
                gu[:, 2, 2] = g["gu_pp"]
                gu[:, 0, 2] = gu[:, 2, 0] = g["gu_rph"]
                gl = np.zeros((Ntri, 3, 3))
                gl[:, 0, 0] = g["g_rr"]
                gl[:, 1, 1] = g["g_tt"]
                gl[:, 2, 2] = g["g_pp"]
                gl[:, 0, 2] = gl[:, 2, 0] = g["g_rph"]
                jac = wz * wt * dr * Ac
                M1 += np.einsum('p,pai,pij,pbj->pab', jac * g["sqg"], W, gu, W)
                M1a += np.einsum('p,pai,pij,pbj->pab',
                                 jac * g["sqg"] * g["al"], W, gu, W)
                M2a += np.einsum('p,pai,pij,pbj->pab',
                                 jac * g["al"] / g["sqg"], Bd, gl, Bd)
                # C1[e,f] = <beta x B_f, W_e>_1:
                #   sqg br [ (g^rp W_r + g^pp W_p) Bd^th - g^tt W_th Bd^ph ]
                br = beta_r(r, th)
                w_c = jac * g["sqg"] * br
                Wc = (g["gu_rph"][:, None] * W[:, :, 0] +
                      g["gu_pp"][:, None] * W[:, :, 2])
                Wt = g["gu_tt"][:, None] * W[:, :, 1]
                C1 += w_c[:, None, None] * (
                    Wc[:, :, None] * Bd[:, None, :, 1] -
                    Wt[:, :, None] * Bd[:, None, :, 2])
                # C2[f,e] = <beta x D_e, B_f>_2 built INDEPENDENTLY from the
                # Ampere-side formula; analytically C2 = -C1^T:
                #   integrand = eps^{kjm} (B_f)_k beta_j (W_e)_m sqg
                # with beta_j = (g_rr b, 0, g_rp b) and (B_f)_k = g_kl Bd^l/sqg:
                #   = b [ (B_f)_th ((g_rp W_th... ] -- expand via eps~:
                #   eps~^{r th ph}=+1; (beta x D)^k = eps~^{kjm} beta_j D_m / sqg
                b_r_low = g["g_rr"] * br          # beta_r (lower)
                b_p_low = g["g_rph"] * br         # beta_phi (lower)
                # (beta x D)^r  = (b_th D_ph - b_ph D_th)/sqg = -b_p_low D_th/sqg
                # (beta x D)^th = (b_ph D_r - b_r D_ph)/sqg
                # (beta x D)^ph = (b_r D_th - b_th D_r)/sqg = b_r_low D_th/sqg
                cross_r = -b_p_low[:, None] * W[:, :, 1] / g["sqg"][:, None]
                cross_t = (b_p_low[:, None] * W[:, :, 0] -
                           b_r_low[:, None] * W[:, :, 2]) / g["sqg"][:, None]
                cross_p = b_r_low[:, None] * W[:, :, 1] / g["sqg"][:, None]
                # C2[f,e] = int g_kl (beta x D_e)^k Bd_f^l d3x  (no alpha)
                gBd_r = (g["g_rr"][:, None] * Bd[:, :, 0] +
                         g["g_rph"][:, None] * Bd[:, :, 2])
                gBd_t = g["g_tt"][:, None] * Bd[:, :, 1]
                gBd_p = (g["g_rph"][:, None] * Bd[:, :, 0] +
                         g["g_pp"][:, None] * Bd[:, :, 2])
                C2 += jac[:, None, None] * (
                    np.einsum('pf,pe->pfe', gBd_r, cross_r) +
                    np.einsum('pf,pe->pfe', gBd_t, cross_t) +
                    np.einsum('pf,pe->pfe', gBd_p, cross_p))
        ii = np.repeat(Eidx, 9, axis=1).ravel()
        jj = np.tile(Eidx, (1, 9)).ravel()
        r1_.append(ii)
        c1_.append(jj)
        x1_.append(M1.ravel())
        x1a_.append(M1a.ravel())
        ii = np.repeat(Fidx, 5, axis=1).ravel()
        jj = np.tile(Fidx, (1, 5)).ravel()
        r2_.append(ii)
        c2_.append(jj)
        x2_.append(M2a.ravel())
        rc_.append(np.repeat(Eidx, 5, axis=1).ravel())
        cc_.append(np.tile(Fidx, (1, 9)).ravel())
        xc_.append(C1.ravel())
        rc2_.append(np.repeat(Fidx, 9, axis=1).ravel())
        cc2_.append(np.tile(Eidx, (1, 5)).ravel())
        xc2_.append(C2.ravel())
        # per-prism energy over the radial band (C++ B cochain, no extra
        # sign flip: the -1 is already inside this Bd basis)
        if r0 >= RLO and r1 <= RHI:
            rng[0] = min(rng[0], r0)
            rng[1] = max(rng[1], r1)
            Bl = Bf[Fidx]
            en_acc += float(np.einsum('pa,pab,pb->', Bl, M2a, Bl))

    def build_csr(rr, cc, xx, nr, nc):
        return coo_matrix(
            (np.concatenate(xx), (np.concatenate(rr), np.concatenate(cc))),
            shape=(nr, nc)).tocsr()

    return (build_csr(r1_, c1_, x1_, Ne, Ne),
            build_csr(r1_, c1_, x1a_, Ne, Ne),
            build_csr(r2_, c2_, x2_, Nf, Nf),
            build_csr(rc_, cc_, xc_, Ne, Nf),
            build_csr(rc2_, cc2_, xc2_, Nf, Ne),
            en_acc, rng)


def rel_diff(A, B):
    scale = max(np.abs(A.data).max(), 1e-300)
    return abs(A - B).max() / scale


def main():
    Ls = [int(x) for x in sys.argv[1:]] or [3]
    prev = None
    for L in Ls:
        d = load(L)
        assert d["whitney"] == 1, "run was not made with use_whitney_hodge"
        log(f"== L{L}: N_edges={d['N_edges']}, N_faces={d['N_faces']}")

        ref_M1, ref_M1a, ref_M2a, ref_C1, ref_C2, en, rng = \
            assemble_reference(L, d)
        # The analytic adjointness identity the Ampere pairing rests on:
        # C2 (independent formula) must equal -C1^T.
        adj = rel_diff(ref_C2, -ref_C1.T.tocsr())
        log(f"   shift adjointness |C2 + C1^T| rel: {adj:.3e}")
        try:
            cpp = load_cpp_csr(L)
            log(f"   matrix rel diff:  M1  {rel_diff(ref_M1, cpp['M1']):.3e}"
                f"   M1a {rel_diff(ref_M1a, cpp['M1a']):.3e}"
                f"   M2a {rel_diff(ref_M2a, cpp['M2a']):.3e}")
            jac_ref = 1.0 / ref_M1.diagonal()
            jd = np.abs(jac_ref - cpp["jac"]).max() / np.abs(jac_ref).max()
            log(f"   jacobi rel diff:  {jd:.3e}")
            if "C1" in cpp:
                log(f"   shift rel diff:   C1  "
                    f"{rel_diff(ref_C1, cpp['C1']):.3e}"
                    f"   C1t {rel_diff(ref_C1.T.tocsr(), cpp['C1t']):.3e}")
        except FileNotFoundError:
            log("   (no whitney.h5 — matrix comparison skipped)")

        # Gauss flux: with the Whitney IC, D~ = M1 . D_primal already, so
        # the flux through shell k is the plain v-edge sum.
        Nh = (d["N_r"] + 1) * d["N_edge_s"]
        Nvs = d["N_vert_s"]
        out = []
        for rt in (1.10569, 3.0927, 8.0160):
            k = int(np.argmin(np.abs(d["radii"] - rt)))
            out.append(d["D"][Nh + k * Nvs + np.arange(Nvs)].sum())
        line = ("   flux(r=1.106, 3.093, 8.016) = "
                + "  ".join(f"{v:12.4e}" for v in out))
        if prev is not None:
            line += "   ord " + " ".join(
                f"{np.log2(abs(p / c)):+5.2f}" for p, c in zip(prev, out))
        log(line)
        prev = out

        # Energy over the actual covered band vs quadrature ground truth.
        Et = lab.energy_truth(rng[0], rng[1])
        log(f"   energy[{rng[0]:.4f},{rng[1]:.4f}] = {en:.6e}"
            f"   truth {Et:.6e}   err {en / Et - 1:+.3%}")

        if d["D_primal"] is not None:
            # PCG sanity: M1 . D_primal must reproduce the stored D~.
            res = ref_M1 @ d["D_primal"] - d["D"]
            log(f"   |M1 D_primal - D~| / |D~| = "
                f"{np.abs(res).max() / max(np.abs(d['D']).max(), 1e-300):.3e}")
    log("\nlab reference (hodge_lab_whitney, diagonal-IC data):")
    log("  L3 flux  4.579e-2  -6.443e-2  -6.961e-2   energy +0.631%")
    log("  L4 flux  1.335e-2  -1.495e-2  -1.733e-2   energy +0.153%")
    log("  L5 flux  3.583e-3  -3.779e-3  -4.330e-3   energy +0.039%")


if __name__ == "__main__":
    main()
