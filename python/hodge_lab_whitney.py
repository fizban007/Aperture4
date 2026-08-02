#!/usr/bin/env python3
"""Proper Whitney-form Galerkin Hodge stars on the prismatic mesh, in the
Kerr-Schild (r, theta, phi) chart.  Reference implementation for GRPIC_PLAN B1.

Element: lowest-order triangular prism (wedge), triangle treated as affine in
(theta, phi) and linear in zeta = (r - r_k)/dr.

  1-forms (9 DOF, covariant components in (r,th,ph)):
    horizontal, bottom : (lam_i grad lam_j - lam_j grad lam_i) * (1-zeta)
    horizontal, top    : same * zeta
    vertical at vert i : W_r = lam_i / dr
  2-forms (5 DOF, DENSITIZED proxy Bd^i = sqrt(g) B^i, which is what has
  polynomial components and unit face flux):
    bottom / top tri   : Bd^r = (1-zeta)/A_c ,  zeta/A_c
    rect on edge (i,j) : Bd^(th,ph) = rot90(w_ij) / dr        (triangle RT0)

  M1[a,b] = \\int g^{ij} W_a,i W_b,j sqrt(g)  dr dth dph
  M2[a,b] = \\int g_ij Bd_a^i Bd_b^j / sqrt(g) dr dth dph

Both are Gram matrices of a real inner product, hence SYMMETRIC POSITIVE
DEFINITE by construction -- which is exactly the stability condition
established in hodge_lab_gr_offdiag.py (skew-adjointness of the generator in
the energy norm).  Note the lapse must still be folded in symmetrically,
sqrt(alpha) M sqrt(alpha).

Scored with the two complementary projections that are NOT blind (see
GRPIC_PLAN B1a):
  hodge1 -> flux through r=const, continuum truth EXACTLY 0 (vacuum);
  hodge2 -> near-horizon magnetic energy vs a 2-D quadrature ground truth.

Usage: python3 hodge_lab_whitney.py [L ...]        (default 3 4)
"""
import sys
import time

import h5py
import numpy as np
from scipy.integrate import quad

A = 0.998
BASE = "/home/alex/Projects/Aperture4/problems/prismatic_wald/Data_conv_L{}_ana"
RLO, RHI = 1.1, 1.5


def log(*a):
    print(*a); sys.stdout.flush()


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def metric(r, th):
    c, s = np.cos(th), np.sin(th)
    rho2 = r * r + A * A * c * c
    Z = 2.0 * r / rho2
    g_rr = 1.0 + Z
    g_rph = -A * s * s * (1.0 + Z)
    g_tt = rho2
    P = r * r + A * A + Z * A * A * s * s
    g_pp = P * s * s
    sqg = rho2 * s * np.sqrt(1.0 + Z)
    gu_rr = P / (g_rr * rho2)          # axis-regular forms (sin^2 cancelled)
    gu_rph = A / rho2
    gu_tt = 1.0 / rho2
    # g^pp = g_rr/det2 with det2 = g_rr*g_pp - g_rph^2 = s^2 (1+Z) rho2,
    # so the (1+Z) cancels: g^pp = 1/(rho2 s^2).  An earlier version kept
    # a spurious g_rr factor here; NEITHER validation projection can see
    # that (flux reads only vertical M1 rows, whose basis has no phi
    # component; energy reads only M2) -- it surfaced when the C++ port
    # (via Metric_KS::gu33) disagreed on the horizontal M1 rows and a
    # direct numerical inversion of gamma_ij sided with the C++.
    gu_pp = 1.0 / (rho2 * np.maximum(s * s, 1e-300))
    al = 1.0 / np.sqrt(1.0 + Z)
    return dict(g_rr=g_rr, g_rph=g_rph, g_tt=g_tt, g_pp=g_pp, sqg=sqg, al=al,
                gu_rr=gu_rr, gu_rph=gu_rph, gu_tt=gu_tt, gu_pp=gu_pp)


def waldB(r, th):
    c, s = np.cos(th), np.sin(th)
    rho2 = r * r + A * A * c * c
    sqg = rho2 * s * np.sqrt(1.0 + 2.0 * r / rho2)
    dAphdth = (A * A + r * r - 2 * r) * s * c + 2 * r * (r ** 4 - A ** 4) * s * c / rho2 ** 2
    dAphdr = (r + A * A * (1 + c * c) * (2 * r * r / rho2 - 1) / rho2) * s * s
    dArdth = -A * s * c + 2 * A * r * (A * A - r * r) * s * c / rho2 ** 2
    return dAphdth / sqg, -dAphdr / sqg, -dArdth / sqg


def energy_truth(rlo=None, rhi=None):
    RL = RLO if rlo is None else rlo
    RH = RHI if rhi is None else rhi
    def f(r, th):
        g = metric(r, th); Br, Bth, Bph = waldB(r, th)
        return g["al"] * (g["g_rr"] * Br * Br + g["g_tt"] * Bth * Bth
                          + g["g_pp"] * Bph * Bph + 2 * g["g_rph"] * Br * Bph) * g["sqg"]
    return 2 * np.pi * quad(lambda r: quad(lambda t: f(r, t), 1e-9, np.pi - 1e-9,
                                           limit=200)[0], RL, RH, limit=200)[0]


def load(L):
    with h5py.File(BASE.format(L) + "/mesh.h5", "r") as m:
        d = {k: int(m[k][()]) for k in ("N_r", "N_tri", "N_edge_s", "N_vert_s",
                                        "N_edges", "N_faces")}
        d["radii"] = np.array(m["radii"], dtype=np.float64)
        d["tvs"] = np.array(m["tri_verts"]).reshape(-1, 3)
        d["te"] = np.array(m["tri_edges_s"]).reshape(-1, 3)
        d["tes"] = np.array(m["tri_edge_signs"]).reshape(-1, 3)
        d["se0"] = np.array(m["sphere_edge_v0"]) if "sphere_edge_v0" in m else None
        d["vth"] = np.array(m["vert_theta"], dtype=np.float64)
        d["vph"] = np.array(m["vert_phi"], dtype=np.float64)
        d["h1inv"] = np.array(m["hodge1_inv"], dtype=np.float64)
        d["e0"] = np.array(m["edge_v0"]); d["e1"] = np.array(m["edge_v1"])
    with h5py.File(BASE.format(L) + "/ic_aux.h5", "r") as f:
        d["D"] = np.array(f["D"], dtype=np.float64)
        d["B"] = np.array(f["B"], dtype=np.float64)
    return d


# 3-point (edge-midpoint) triangle rule, degree 2; 2-point Gauss in zeta.
TRI_L = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
TRI_W = np.full(3, 1.0 / 3.0)
GZ = 0.5 + np.array([-1, 1]) / (2 * np.sqrt(3.0))
GW = np.full(2, 0.5)


def build(L, d):
    Nr, Ntri, Nes, Nvs = d["N_r"], d["N_tri"], d["N_edge_s"], d["N_vert_s"]
    Ne, Nf = d["N_edges"], d["N_faces"]
    Nh = (Nr + 1) * Nes
    ntf = (Nr + 1) * Ntri
    tvs, te, tes = d["tvs"], d["te"], d["tes"]
    th_v = d["vth"][:Nvs]; ph_v = d["vph"][:Nvs]

    # triangle-local (theta, phi) with phi unwrapped about vertex 0
    U = np.zeros((Ntri, 3, 2))
    U[:, :, 0] = th_v[tvs]
    p0 = ph_v[tvs[:, 0]]
    U[:, :, 1] = p0[:, None] + wrap(ph_v[tvs] - p0[:, None])
    M = np.ones((Ntri, 3, 3)); M[:, :, 1:] = U
    C = np.linalg.inv(M)                       # lam_i(u) = C[0,i] + C[1,i]th + C[2,i]ph
    grad = np.transpose(C[:, 1:, :], (0, 2, 1))   # (Ntri,3,2)  grad lam_i
    Ac = 0.5 * np.abs(np.cross(U[:, 1] - U[:, 0], U[:, 2] - U[:, 0]))
    sgnA = np.sign(np.cross(U[:, 1] - U[:, 0], U[:, 2] - U[:, 0]))

    # local edge j connects local verts (j, (j+1)%3); verify against the mesh
    loc = [(0, 1), (1, 2), (2, 0)]
    Dg = np.zeros(Ne); Hg = np.zeros(Nf)
    Dprim = d["h1inv"] * d["D"]
    Bf = d["B"]
    Eacc = [0.0, 0.0]; rng = [1e9, -1e9]

    for k in range(Nr):
        r0, r1 = d["radii"][k], d["radii"][k + 1]; dr = r1 - r0
        eh0 = k * Nes + te; eh1 = (k + 1) * Nes + te
        ev = Nh + k * Nvs + tvs
        Eidx = np.concatenate([eh0, eh1, ev], 1)                 # (Ntri,9)
        Fidx = np.concatenate([(k * Ntri + np.arange(Ntri))[:, None],
                               ((k + 1) * Ntri + np.arange(Ntri))[:, None],
                               ntf + k * Nes + te], 1)           # (Ntri,5)
        # edge sign: local i->j vs global v0->v1
        es = tes.astype(np.float64)
        M1 = np.zeros((Ntri, 9, 9)); M2 = np.zeros((Ntri, 5, 5))
        for iz, (z, wz) in enumerate(zip(GZ, GW)):
            r = r0 + z * dr
            for lam, wt in zip(TRI_L, TRI_W):
                th = lam @ U[:, :, 0].T
                g = metric(r, th)
                # ---- 1-form basis, covariant (r,th,ph) ----
                W = np.zeros((Ntri, 9, 3))
                for j, (a_, b_) in enumerate(loc):
                    w2 = (lam[a_] * grad[:, b_, :] - lam[b_] * grad[:, a_, :])
                    W[:, j, 1:] = w2 * (1 - z) * es[:, j][:, None]
                    W[:, 3 + j, 1:] = w2 * z * es[:, j][:, None]
                for i in range(3):
                    W[:, 6 + i, 0] = lam[i] / dr
                # ---- 2-form densitized proxy ----
                Bd = np.zeros((Ntri, 5, 3))
                Bd[:, 0, 0] = (1 - z) / Ac
                Bd[:, 1, 0] = z / Ac
                for j, (a_, b_) in enumerate(loc):
                    w2 = (lam[a_] * grad[:, b_, :] - lam[b_] * grad[:, a_, :])
                    rot = np.stack([-w2[:, 1], w2[:, 0]], 1) * sgnA[:, None]
                    Bd[:, 2 + j, 1:] = rot / dr * es[:, j][:, None]
                # ---- contract ----
                gu = np.zeros((Ntri, 3, 3))
                gu[:, 0, 0] = g["gu_rr"]; gu[:, 1, 1] = g["gu_tt"]
                gu[:, 2, 2] = g["gu_pp"]; gu[:, 0, 2] = g["gu_rph"]; gu[:, 2, 0] = g["gu_rph"]
                gl = np.zeros((Ntri, 3, 3))
                gl[:, 0, 0] = g["g_rr"]; gl[:, 1, 1] = g["g_tt"]
                gl[:, 2, 2] = g["g_pp"]; gl[:, 0, 2] = g["g_rph"]; gl[:, 2, 0] = g["g_rph"]
                jac = wz * wt * dr * Ac
                M1 += np.einsum('p,pai,pij,pbj->pab', jac * g["sqg"], W, gu, W)
                M2 += np.einsum('p,pai,pij,pbj->pab', jac * g["al"] / g["sqg"], Bd, gl, Bd)
        np.add.at(Dg, Eidx.ravel(), np.einsum('pab,pb->pa', M1, Dprim[Eidx]).ravel())
        # energy accumulated PER PRISM over the radial band -- summing a
        # face-indexed array over a face-radius band double counts and
        # straddles, which is not the same integral.
        if r0 >= RLO and r1 <= RHI:
            rng[0] = min(rng[0], r0); rng[1] = max(rng[1], r1)
            for sgn in (0, 1):
                Bl = Bf[Fidx].copy()
                if sgn: Bl[:, 2:] *= -1.0        # rect-face orientation flip
                Eacc[sgn] += float(np.einsum('pa,pab,pb->', Bl, M2, Bl))
    return Dg, Eacc, rng


def main():
    Ls = [int(x) for x in sys.argv[1:]] or [3, 4]
    Et = energy_truth()
    log(f"Whitney-form Galerkin Hodge.  E_true(r in [{RLO},{RHI}]) = {Et:.8e}\n")
    log(f"{'L':>2} {'flux r=1.106':>14} {'r=3.093':>13} {'r=8.016':>13} "
        f"{'actual band':>17} {'E_whitney':>12} {'err':>9}")
    prev = None
    for L in Ls:
        t0 = time.time()
        d = load(L)
        Dg, Eacc, rng = build(L, d)
        Nr, Ntri, Nes, Nvs = d["N_r"], d["N_tri"], d["N_edge_s"], d["N_vert_s"]
        Nh = (Nr + 1) * Nes; ntf = (Nr + 1) * Ntri
        out = []
        for rt in (1.10569, 3.0927, 8.0160):
            k = int(np.argmin(np.abs(d["radii"] - rt)))
            out.append(Dg[Nh + k * Nvs + np.arange(Nvs)].sum())
        # ground truth over the ACTUAL radial extent the prisms cover, which
        # shifts with L -- comparing every level to E_true(1.1,1.5) charges
        # the scheme for a band-selection difference it did not make.
        Eref = energy_truth(rng[0], rng[1])
        line = (f"{L:>2} " + " ".join(f"{v:13.4e}" for v in out)
                + f"  [{rng[0]:.4f},{rng[1]:.4f}] {Eacc[1]:12.5e} "
                + f"{Eacc[1]/Eref-1:+8.3%}")
        if prev is not None:
            line += "   ord " + " ".join(f"{np.log2(abs(p/c)):+5.2f}"
                                         for p, c in zip(prev, out))
        log(line + f"   ({time.time()-t0:.0f}s)")
        prev = out


if __name__ == "__main__":
    main()
