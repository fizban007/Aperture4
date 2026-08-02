#!/usr/bin/env python3
"""End-to-end accuracy test for the GR off-diagonal Hodge correction
(GRPIC_PLAN blocker B1).  Companion to hodge_lab_gr_offdiag.py, which covers
stability; this one covers CONSISTENCY -- does the correction actually remove
the O(1) Gauss-law violation, and does the residual then converge?

The defect
----------
For a radial (vertical) edge the code forms

    D~_diag[e] = (D_primal[e] / |e|) * |e*|  =  (D_r / sqrt(g_rr)) * |e*|

where |e*| is the INDUCED metric area of the dual polygon in the r=const
surface (verified: prismatic_mesh_metric_impl.hpp builds it from tri
circumcenters at r_mid).  The exact dual-face flux is

    D~[e] = \\int D^r sqrt(g) dtheta dphi = (g^rr D_r + g^rphi D_phi) * S_e,
    S_e   = |e*| * sqrt(g) / sqrt(g_thth g_phph).

Two separate errors follow, both O(1) at a != 0:
  (1) the omitted g^rphi D_phi term  -- needs D_phi, which is not a mesh
      direction and must be reconstructed from horizontal edge circulations;
  (2) a scalar prefactor, since  g^rr sqrt(g_rr) sqrt(g)/sqrt(g_thth g_phph)
      = 1/sqrt(1 - k^2)  with  k^2 = g_rphi^2/(g_rr g_phph),  which is 1 only
      when g_rphi = 0.

Two-stage validation
--------------------
Stage A uses ANALYTIC D_r, D_phi: validates the flux formula itself.
Stage B uses the DISCRETE state -- D_r from the stored circulations and
D_phi least-squares-reconstructed from horizontal edges: validates the
scheme as it would actually be implemented.

Ground truth: the continuum flux through any r=const surface is EXACTLY zero
for vacuum Wald (verified independently by quadrature).

Usage: python3 hodge_lab_gr_gauss.py [L ...]     (default 3 4 5)
"""
import sys
import time
from collections import defaultdict

import h5py
import numpy as np

A = 0.998
BP = 1.0
BASE = "/home/alex/Projects/Aperture4/problems/prismatic_wald/Data_conv_L{}_ana"


def log(*a):
    print(*a)
    sys.stdout.flush()


# ---------------------------------------------------------------- metric ---
def ks(r, th):
    c, s = np.cos(th), np.sin(th)
    rho2 = r * r + A * A * c * c
    Z = 2.0 * r / rho2
    g_rr = 1.0 + Z
    g_rph = -A * s * s * (1.0 + Z)
    g_thth = rho2
    g_phph = (r * r + A * A + Z * A * A * s * s) * s * s
    sqrtg = rho2 * s * np.sqrt(1.0 + Z)
    # AXIS REGULARIZATION.  g_phph, g_rph ~ sin^2, sqrt(g) ~ sin, so the
    # ratios below are 0/0 at theta = 0, pi even though every one of them has
    # a finite limit.  Cancel sin analytically (same trick as
    # prismatic_sph_output.cpp).  With P = r^2+a^2+Z a^2 sin^2 one gets
    #   det2 = sin^2 * g_rr * rho2      (the sin^2 factors out exactly)
    #   g^rr = P/(g_rr rho2),  g^rphi = a/rho2,
    #   S_e/|e*| = sqrt(rho2 g_rr / P)
    P = r * r + A * A + Z * A * A * s * s
    gu_rr = P / (g_rr * rho2)
    gu_rph = A / rho2
    S_over_area = np.sqrt(rho2 * g_rr / P)
    return dict(g_rr=g_rr, g_rph=g_rph, g_thth=g_thth, g_phph=g_phph,
                sqrtg=sqrtg, gu_rr=gu_rr, gu_rph=gu_rph, Z=Z, rho2=rho2,
                P=P, S_over_area=S_over_area, s2=s * s)


def wald_D_up(r, th):
    """Analytic contravariant D^r, D^phi (matches wald_solution.hpp)."""
    c, s = np.cos(th), np.sin(th)
    rho2 = r * r + A * A * c * c
    Z = 2.0 * r / rho2
    alpha = 1.0 / np.sqrt(1.0 + Z)
    beta1 = 2.0 * r / (r * (2.0 + r) + (A * c) ** 2)
    gu11 = (A * A + r * r) / rho2 - 2.0 * r / (rho2 + 2.0 * r)
    gu13 = A / rho2
    gu33 = 1.0 / rho2 / np.maximum(s * s, 1e-300)
    dA0dr = A * (1.0 + c * c) * ((A * c) ** 2 - r * r) / rho2 ** 2
    dAphdr = (r + A * A * (1.0 + c * c) * (2.0 * r * r / rho2 - 1.0) / rho2) * s * s
    D_r = BP * (gu11 * dA0dr + gu13 * beta1 * dAphdr) / alpha
    D_ph = BP * (gu33 * beta1 * dAphdr + gu13 * dA0dr) / alpha
    return D_r, D_ph


def wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


# ------------------------------------------------------------------ mesh ---
def load(L):
    with h5py.File(BASE.format(L) + "/mesh.h5", "r") as m:
        d = dict(
            N_r=int(m["N_r"][()]), N_edge_s=int(m["N_edge_s"][()]),
            N_vert_s=int(m["N_vert_s"][()]), N_edges=int(m["N_edges"][()]),
            radii=np.array(m["radii"], dtype=np.float64),
            h1inv=np.array(m["hodge1_inv"], dtype=np.float64),
            elen=np.array(m["edge_length"], dtype=np.float64),
            e0=np.array(m["edge_v0"]), e1=np.array(m["edge_v1"]),
            vth=np.array(m["vert_theta"], dtype=np.float64),
            vph=np.array(m["vert_phi"], dtype=np.float64),
            svz=np.array(m["sphere_vz"], dtype=np.float64),
        )
    with h5py.File(BASE.format(L) + "/ic_aux.h5", "r") as f:
        d["Dt"] = np.array(f["D"], dtype=np.float64)
    return d


def reconstruct_Dphi(d):
    """Least-squares (D_theta, D_phi) at every (sphere vertex, shell) from the
    circulations of the incident horizontal edges.  Returns array [k, s]."""
    N_r, Nes, Nvs = d["N_r"], d["N_edge_s"], d["N_vert_s"]
    n_h = (N_r + 1) * Nes
    Dprim = d["h1inv"] * d["Dt"]                    # primal circulations
    th, ph = d["vth"], d["vph"]

    # incident horizontal edges per sphere-vertex (shell 0 pattern repeats)
    fan = defaultdict(list)
    for e in range(Nes):
        fan[d["e0"][e] % Nvs].append(e)
        fan[d["e1"][e] % Nvs].append(e)

    Dphi = np.zeros((N_r + 1, Nvs))
    for k in range(N_r + 1):
        off = k * Nes
        voff = k * Nvs
        for s in range(Nvs):
            es = fan.get(s, [])
            if len(es) < 2:
                continue
            eg = np.asarray(es) + off
            a0, a1 = d["e0"][eg], d["e1"][eg]
            M = np.stack([th[a1] - th[a0], wrap(ph[a1] - ph[a0])], 1)
            rhs = Dprim[eg]
            # normal equations; skip degenerate (polar) fits
            G = M.T @ M
            if abs(np.linalg.det(G)) < 1e-14:
                continue
            Dphi[k, s] = np.linalg.solve(G, M.T @ rhs)[1]
    return Dphi


def shell_flux(d, Dphi_disc, k_target):
    """Total flux through the r=const surface in slab k, four ways."""
    N_r, Nes, Nvs = d["N_r"], d["N_edge_s"], d["N_vert_s"]
    n_h = (N_r + 1) * Nes
    k = k_target
    ei = n_h + k * Nvs + np.arange(Nvs)

    r0, r1 = d["radii"][k], d["radii"][k + 1]
    rm = 0.5 * (r0 + r1)
    cth = np.clip(d["svz"], -1.0, 1.0)
    thv = np.arccos(cth)

    g = ks(rm, thv)
    area = d["elen"][ei] / np.maximum(d["h1inv"][ei], 1e-300)     # |e*|
    S_e = area * g["S_over_area"]

    # discrete D_r from the stored circulation
    Dprim = d["h1inv"][ei] * d["Dt"][ei]
    D_r_disc = Dprim / (r1 - r0)

    # analytic covariant components at (rm, thv)
    Dup_r, Dup_ph = wald_D_up(rm, thv)
    D_r_ana = g["g_rr"] * Dup_r + g["g_rph"] * Dup_ph
    D_ph_ana = g["s2"] * (g["P"] * Dup_ph - A * g["g_rr"] * Dup_r)

    Dphi_d = 0.5 * (Dphi_disc[k] + Dphi_disc[k + 1])

    out = {}
    out["code (diagonal)"] = d["Dt"][ei].sum()
    out["A: analytic, corrected"] = ((g["gu_rr"] * D_r_ana
                                      + g["gu_rph"] * D_ph_ana) * S_e).sum()
    out["A: analytic, no offdiag"] = ((g["gu_rr"] * D_r_ana) * S_e).sum()
    out["B: discrete, corrected"] = ((g["gu_rr"] * D_r_disc
                                      + g["gu_rph"] * Dphi_d) * S_e).sum()
    out["B: discrete, no offdiag"] = ((g["gu_rr"] * D_r_disc) * S_e).sum()
    return out


def main():
    Ls = [int(x) for x in sys.argv[1:]] or [3, 4, 5]
    targets = [1.10569, 3.0927, 8.0160]
    log("Total flux through r=const surfaces.  Continuum truth = 0 EXACTLY\n"
        "(vacuum Wald), so every number below is pure error.\n")
    for L in Ls:
        t0 = time.time()
        d = load(L)
        Dphi = reconstruct_Dphi(d)
        rows = {}
        for rt in targets:
            k = int(np.argmin(np.abs(d["radii"] - rt)))
            rows[rt] = shell_flux(d, Dphi, k)
        keys = list(next(iter(rows.values())).keys())
        log(f"--- L{L}  (N_r={d['N_r']}, {time.time()-t0:.1f}s) ---")
        log(f"  {'variant':>26} " + " ".join(f"{'r='+str(t):>13}" for t in targets))
        for kk in keys:
            log(f"  {kk:>26} " + " ".join(f"{rows[t][kk]:13.4e}" for t in targets))
        log("")


if __name__ == "__main__":
    main()
