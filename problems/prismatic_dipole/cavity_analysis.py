#!/usr/bin/env python3
"""Verification and convergence analysis for the spherical-cavity TE/TM mode test.

Reads field snapshots produced by `cavity_resonator`, evaluates the analytic
TE or TM eigenmode at the snapshot times, and reports L2 errors. Can be run
on a single output directory (single-resolution check) or on a list of
output directories at different mesh resolutions (convergence study).

Single-run usage:
    python cavity_analysis.py Data_cavity \\
        --l 1 --m 0 --n_root 1 --pol E --start_with_e false

Convergence study usage:
    python cavity_analysis.py Data_cavity_L2 Data_cavity_L3 Data_cavity_L4 \\
        --l 1 --m 0 --n_root 1 --pol E --convergence
"""

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn
from scipy.optimize import brentq


# =========================================================================
# Data loading (mirrors validate_vacuum_dipole.py)
# =========================================================================

def load_mesh(data_dir):
    mesh = {}
    with h5py.File(os.path.join(data_dir, "mesh.h5")) as f:
        for key in ["L", "N_r", "N_tri", "N_vert_s", "N_edge_s",
                    "N_verts", "N_edges", "N_faces"]:
            mesh[key] = int(f[key][()])
        for key in ["radii", "vert_x", "vert_y", "vert_z",
                    "face_area", "edge_length",
                    "tri_face_v0", "tri_face_v1", "tri_face_v2",
                    "rect_face_v0", "rect_face_v1", "rect_face_v2", "rect_face_v3",
                    "edge_v0", "edge_v1",
                    "edge_boundary", "face_boundary",
                    "edge_radial_layer", "face_radial_layer",
                    "hodge1_inv", "hodge2"]:
            mesh[key] = f[key][:]
    return mesh


def load_fields(data_dir, step):
    with h5py.File(os.path.join(data_dir, f"step_{step:06d}.h5")) as f:
        B_f = f["B_f"][:]
        E_e = f["E_e"][:]
        t = float(f["time"][()])
    return B_f, E_e, t


def list_steps(data_dir):
    """Return sorted list of available step indices."""
    files = [f for f in os.listdir(data_dir)
             if f.startswith("step_") and f.endswith(".h5")]
    return sorted(int(f[5:11]) for f in files)


# =========================================================================
# Eigenvalue solver (matches src/systems/prismatic/cavity_modes.hpp)
# =========================================================================

def te_determinant(l, k, a, b):
    return spherical_jn(l, k * a) * spherical_yn(l, k * b) \
         - spherical_jn(l, k * b) * spherical_yn(l, k * a)


def tm_radial_u(l, k, r):
    x = k * r
    return spherical_jn(l, x) + x * spherical_jn(l, x, derivative=True)


def tm_radial_v(l, k, r):
    x = k * r
    return spherical_yn(l, x) + x * spherical_yn(l, x, derivative=True)


def tm_determinant(l, k, a, b):
    return tm_radial_u(l, k, a) * tm_radial_v(l, k, b) \
         - tm_radial_u(l, k, b) * tm_radial_v(l, k, a)


def find_nth_root(D, n_root, k_min, k_max, n_grid=4096):
    """Find n-th positive root of D(k) = 0 by bracketing + brentq."""
    ks = np.linspace(k_min, k_max, n_grid)
    fs = np.array([D(k) for k in ks])
    # Identify sign changes that aren't at singularities
    found = 0
    for i in range(len(ks) - 1):
        if fs[i] * fs[i + 1] < 0 and abs(fs[i]) < 1e10 and abs(fs[i + 1]) < 1e10:
            found += 1
            if found == n_root:
                return brentq(D, ks[i], ks[i + 1], xtol=1e-12)
    raise RuntimeError(f"Root #{n_root} not found in [{k_min}, {k_max}]")


def cavity_eigenvalue(l, a, b, n_root, polarization):
    dr = b - a
    k_max = (n_root + 5) * np.pi / dr
    if polarization == 'E':  # TE
        return find_nth_root(lambda k: te_determinant(l, k, a, b),
                              n_root, 1e-6, k_max)
    else:  # TM
        return find_nth_root(lambda k: tm_determinant(l, k, a, b),
                              n_root, 1e-6, k_max)


def inner_bc_alpha(l, k, a, polarization):
    if polarization == 'E':
        return -spherical_jn(l, k * a) / spherical_yn(l, k * a)
    else:
        u = tm_radial_u(l, k, a)
        v = tm_radial_v(l, k, a)
        return -u / v


# =========================================================================
# Real spherical harmonics (matches the C++ convention)
# =========================================================================

def _assoc_legendre_pair(l, m, x):
    """Return (P_l^m(x), P_{l-1}^m(x)) for arrays x. Same recursion and
    Condon-Shortley convention as cavity_modes.hpp::assoc_legendre. m >= 0,
    l >= m. Returns 0 for the second value when l == m.
    """
    # P_m^m
    Pmm = np.ones_like(x)
    if m > 0:
        somx2 = np.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for _ in range(m):
            Pmm = Pmm * (-fact * somx2)
            fact += 2.0
    if l == m:
        return Pmm, np.zeros_like(x)
    # P_{m+1}^m
    Pmmp1 = x * (2.0 * m + 1.0) * Pmm
    if l == m + 1:
        return Pmmp1, Pmm
    # Upward recursion
    P_lm2 = Pmm
    P_lm1 = Pmmp1
    P_l = None
    for ll in range(m + 2, l + 1):
        P_l = (x * (2.0 * ll - 1.0) * P_lm1 - (ll + m - 1.0) * P_lm2) / (ll - m)
        P_lm2 = P_lm1
        P_lm1 = P_l
    return P_l, P_lm2  # (P_l^m, P_{l-1}^m)


def real_sph_harm(l, m, theta, phi):
    """Return Y_lm, ∂Y/∂θ, ∂Y/∂φ in real form (vectorized in θ, φ).

    Convention matches src/systems/prismatic/cavity_modes.hpp:
        m > 0:  sqrt(2) * N_l|m| * P_l^|m|(cos θ) * cos(|m| φ)
        m = 0:  N_l0  * P_l(cos θ)
        m < 0:  sqrt(2) * N_l|m| * P_l^|m|(cos θ) * sin(|m| φ)

    Uses analytical derivatives:
        dP_l^m/dθ = (l cos θ P_l^m - (l+m) P_{l-1}^m) / sin θ
    """
    am = abs(m)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    Plm, Plm_lo = _assoc_legendre_pair(l, am, cos_t)

    # dP/dθ — handle the (rare) sin θ ≈ 0 case
    safe_sin = np.where(np.abs(sin_t) > 1e-14, sin_t, 1.0)
    dPlm_dt = (l * cos_t * Plm - (l + am) * Plm_lo) / safe_sin
    dPlm_dt = np.where(np.abs(sin_t) > 1e-14, dPlm_dt, 0.0)

    # Normalization N_lm  =  sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)
    norm = np.sqrt((2.0 * l + 1.0) / (4.0 * np.pi))
    for k in range(l - am + 1, l + am + 1):
        norm /= np.sqrt(k)

    if m == 0:
        Y = norm * Plm
        dY_dtheta = norm * dPlm_dt
        dY_dphi = np.zeros_like(theta)
    else:
        sqrt2 = np.sqrt(2.0)
        if m > 0:
            cos_mp = np.cos(m * phi)
            sin_mp = np.sin(m * phi)
            Y = sqrt2 * norm * Plm * cos_mp
            dY_dtheta = sqrt2 * norm * dPlm_dt * cos_mp
            dY_dphi = -sqrt2 * norm * Plm * m * sin_mp
        else:
            cos_mp = np.cos(am * phi)
            sin_mp = np.sin(am * phi)
            Y = sqrt2 * norm * Plm * sin_mp
            dY_dtheta = sqrt2 * norm * dPlm_dt * sin_mp
            dY_dphi = sqrt2 * norm * Plm * am * cos_mp
    return Y, dY_dtheta, dY_dphi


# =========================================================================
# Mode field evaluator (matches cavity_modes.hpp::evaluate_mode_at_time)
# =========================================================================

def evaluate_mode(l, m, k, alpha, amp, polarization, x, y, z, t,
                  start_with_e):
    """Vectorized analytic mode field at (x, y, z, t).
    Returns (Ex, Ey, Ez, Bx, By, Bz) arrays of shape x.shape.
    """
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)

    er = np.stack([sin_t * cos_p, sin_t * sin_p, cos_t], axis=-1)
    et = np.stack([cos_t * cos_p, cos_t * sin_p, -sin_t], axis=-1)
    ep = np.stack([-sin_p, cos_p, np.zeros_like(sin_p)], axis=-1)

    Y, dY_dtheta, dY_dphi = real_sph_harm(l, m, theta, phi)

    kr = k * r
    f = spherical_jn(l, kr) + alpha * spherical_yn(l, kr)
    fp_x = spherical_jn(l, kr, derivative=True) \
         + alpha * spherical_yn(l, kr, derivative=True)
    rf_prime = f + r * k * fp_x

    safe_inv_sin = np.where(np.abs(sin_t) > 1e-14, 1.0 / sin_t, 0.0)
    omega = k
    l_lp1 = l * (l + 1)

    Er = np.zeros_like(r)
    Et = np.zeros_like(r)
    Ep = np.zeros_like(r)
    Br = np.zeros_like(r)
    Bt = np.zeros_like(r)
    Bp = np.zeros_like(r)

    if polarization == 'E':  # TE
        Et = f * dY_dphi * safe_inv_sin
        Ep = -f * dY_dtheta
        Br = -(l_lp1 / (omega * r)) * f * Y
        Bt = -(rf_prime / (omega * r)) * dY_dtheta
        Bp = -(rf_prime / (omega * r)) * dY_dphi * safe_inv_sin
    else:  # TM
        Bt = f * dY_dphi * safe_inv_sin
        Bp = -f * dY_dtheta
        Er = -(l_lp1 / (omega * r)) * f * Y
        Et = -(rf_prime / (omega * r)) * dY_dtheta
        Ep = -(rf_prime / (omega * r)) * dY_dphi * safe_inv_sin

    Er *= amp
    Et *= amp
    Ep *= amp
    Br *= amp
    Bt *= amp
    Bp *= amp

    Ex_p = Er * er[..., 0] + Et * et[..., 0] + Ep * ep[..., 0]
    Ey_p = Er * er[..., 1] + Et * et[..., 1] + Ep * ep[..., 1]
    Ez_p = Er * er[..., 2] + Et * et[..., 2] + Ep * ep[..., 2]
    Bx_p = Br * er[..., 0] + Bt * et[..., 0] + Bp * ep[..., 0]
    By_p = Br * er[..., 1] + Bt * et[..., 1] + Bp * ep[..., 1]
    Bz_p = Br * er[..., 2] + Bt * et[..., 2] + Bp * ep[..., 2]

    cphase = np.cos(omega * t)
    sphase = np.sin(omega * t)
    if start_with_e:
        Ex = Ex_p * cphase
        Ey = Ey_p * cphase
        Ez = Ez_p * cphase
        Bx = Bx_p * sphase
        By = By_p * sphase
        Bz = Bz_p * sphase
    else:
        Ex = -Ex_p * sphase
        Ey = -Ey_p * sphase
        Ez = -Ez_p * sphase
        Bx = Bx_p * cphase
        By = By_p * cphase
        Bz = Bz_p * cphase
    return Ex, Ey, Ez, Bx, By, Bz


# =========================================================================
# Gauss quadrature (10-point on [0,1], same as validate_vacuum_dipole.py)
# =========================================================================

_gauss_xs = np.array([0.1488743389816312, 0.4333953941292472,
                      0.6794095682990244, 0.8650633666889845,
                      0.9739065285171717])
_gauss_ws = np.array([0.2955242247147529, 0.2692667193099963,
                      0.2190863625159821, 0.1494513491505806,
                      0.0666713443086881])
_u01 = np.concatenate([0.5 + 0.5 * _gauss_xs, 0.5 - 0.5 * _gauss_xs])
_w01 = np.concatenate([_gauss_ws, _gauss_ws]) * 0.5


def analytic_face_flux_tri(p0, p1, p2, field_fn):
    e1 = p1 - p0
    e2 = p2 - p0
    n = np.cross(e1, e2)
    flux = np.zeros(len(p0))
    for u, wu in zip(_u01, _w01):
        for s, ws in zip(_u01, _w01):
            v = (1.0 - u) * s
            pt = p0 + u * e1 + v * e2
            Fx, Fy, Fz = field_fn(pt[:, 0], pt[:, 1], pt[:, 2])
            flux += wu * ws * (1.0 - u) * (Fx*n[:, 0] + Fy*n[:, 1] + Fz*n[:, 2])
    return flux


def analytic_face_flux_quad(p0, p1, p2, p3, field_fn):
    flux = np.zeros(len(p0))
    for u, wu in zip(_u01, _w01):
        for v, wv in zip(_u01, _w01):
            pt = (1-u)*(1-v)*p0 + u*(1-v)*p1 + u*v*p2 + (1-u)*v*p3
            dxdu = -(1-v)*p0 + (1-v)*p1 + v*p2 - v*p3
            dxdv = -(1-u)*p0 - u*p1 + u*p2 + (1-u)*p3
            nn = np.cross(dxdu, dxdv)
            Fx, Fy, Fz = field_fn(pt[:, 0], pt[:, 1], pt[:, 2])
            flux += wu * wv * (Fx*nn[:, 0] + Fy*nn[:, 1] + Fz*nn[:, 2])
    return flux


def analytic_edge_circ(x0, x1, field_fn):
    dl = x1 - x0
    circ = np.zeros(len(x0))
    for s, ws in zip(_u01, _w01):
        pt = x0 + s * dl
        Fx, Fy, Fz = field_fn(pt[:, 0], pt[:, 1], pt[:, 2])
        circ += ws * (Fx*dl[:, 0] + Fy*dl[:, 1] + Fz*dl[:, 2])
    return circ


def _get_verts(mesh, indices):
    return np.column_stack([mesh["vert_x"][indices],
                             mesh["vert_y"][indices],
                             mesh["vert_z"][indices]])


# =========================================================================
# Cochain construction from analytic mode
# =========================================================================

def analytic_cochains(mesh, mode_args, t):
    """Compute (E_e_analytic, B_f_analytic) by Gauss quadrature of the
    analytic eigenmode at time t."""
    l, m, k, alpha, amp, polarization, start_with_e = mode_args

    def E_field(x, y, z):
        Ex, Ey, Ez, _, _, _ = evaluate_mode(
            l, m, k, alpha, amp, polarization, x, y, z, t, start_with_e)
        return Ex, Ey, Ez

    def B_field(x, y, z):
        _, _, _, Bx, By, Bz = evaluate_mode(
            l, m, k, alpha, amp, polarization, x, y, z, t, start_with_e)
        return Bx, By, Bz

    n_tri_faces = mesh["N_tri"] * (mesh["N_r"] + 1)
    n_rect_faces = mesh["N_edge_s"] * mesh["N_r"]

    # Triangular faces
    p0 = _get_verts(mesh, mesh["tri_face_v0"])
    p1 = _get_verts(mesh, mesh["tri_face_v1"])
    p2 = _get_verts(mesh, mesh["tri_face_v2"])
    flux_tri = analytic_face_flux_tri(p0, p1, p2, B_field)

    # Rectangular faces
    p0 = _get_verts(mesh, mesh["rect_face_v0"])
    p1 = _get_verts(mesh, mesh["rect_face_v1"])
    p2 = _get_verts(mesh, mesh["rect_face_v2"])
    p3 = _get_verts(mesh, mesh["rect_face_v3"])
    flux_rect = analytic_face_flux_quad(p0, p1, p2, p3, B_field)

    B_f = np.concatenate([flux_tri, flux_rect])

    # All edges
    x0 = _get_verts(mesh, mesh["edge_v0"])
    x1 = _get_verts(mesh, mesh["edge_v1"])
    E_e = analytic_edge_circ(x0, x1, E_field)

    return E_e, B_f


# =========================================================================
# Error metrics
# =========================================================================

def l2_norms(mesh, A_num, A_ana, mask=None):
    """Discrete L2 norms of (A_num - A_ana) and A_ana, induced by the
    diagonal DEC Hodge star.

    For face cochains:
        ||B||^2 = Σ_f hodge2[f] * B_f^2     ≈ ∫|B|^2 dV
    For edge cochains:
        ||E||^2 = Σ_e (1/hodge1_inv[e]) * E_e^2     ≈ ∫|E|^2 dV
    (The 1/hodge1_inv form avoids needing to invert the diagonal.)

    Returns (||diff||, ||A_ana||) — caller decides how to normalize.
    """
    if A_num.shape == mesh["face_area"].shape:
        weights = mesh["hodge2"]
    else:
        # hodge1 = 1 / hodge1_inv; protect zeros from boundary tagging
        h1inv = mesh["hodge1_inv"]
        weights = np.where(h1inv > 0, 1.0 / np.where(h1inv > 0, h1inv, 1.0), 0.0)

    d2 = weights * (A_num - A_ana) ** 2
    a2 = weights * A_ana ** 2

    if mask is not None:
        d2 = d2[mask]
        a2 = a2[mask]
    return float(np.sqrt(d2.sum())), float(np.sqrt(a2.sum()))


# =========================================================================
# Main entry points
# =========================================================================

def analyze_run(data_dir, l, m, n_root, polarization, start_with_e,
                amp=1.0, verbose=True):
    """Compare every snapshot in data_dir against the analytic mode.
    Returns (times, E_errors, B_errors)."""
    mesh = load_mesh(data_dir)
    a = float(mesh["radii"][0])
    b = float(mesh["radii"][-1])
    k = cavity_eigenvalue(l, a, b, n_root, polarization)
    alpha = inner_bc_alpha(l, k, a, polarization)
    omega = k
    period = 2 * np.pi / omega
    if verbose:
        print(f"Cavity: r_min={a}, r_max={b}")
        print(f"Mode: {polarization} l={l} m={m} n_root={n_root}")
        print(f"  k = {k:.8f}, omega = {omega:.8f}, period T = {period:.6f}")
        print(f"  alpha = {alpha:.6e}")

    mode_args = (l, m, k, alpha, amp, polarization, start_with_e)

    steps = list_steps(data_dir)
    if not steps:
        raise RuntimeError(f"No step files in {data_dir}")

    # Reference amplitude: ||analytic|| at the phase where each field is at
    # its temporal max (so we have a non-vanishing denominator regardless of
    # which snapshot we score). For start_with_e=False this is t=0 for B and
    # t=T/4 for E; flipped for start_with_e=True.
    face_mask = mesh["face_boundary"] == 0
    edge_mask = mesh["edge_boundary"] == 0
    t_Bmax = 0.0 if not start_with_e else period / 4.0
    t_Emax = period / 4.0 if not start_with_e else 0.0
    _, B_amp_norm = l2_norms(mesh,
                              np.zeros(mesh["N_faces"]),
                              analytic_cochains(mesh, mode_args, t_Bmax)[1],
                              mask=face_mask)
    _, E_amp_norm = l2_norms(mesh,
                              np.zeros(mesh["N_edges"]),
                              analytic_cochains(mesh, mode_args, t_Emax)[0],
                              mask=edge_mask)
    if verbose:
        print(f"  reference ||B||@max = {B_amp_norm:.4e}, "
              f"||E||@max = {E_amp_norm:.4e}")

    times, E_errs, B_errs = [], [], []
    for step in steps:
        B_f, E_e, t = load_fields(data_dir, step)
        E_ana, B_ana = analytic_cochains(mesh, mode_args, t)

        E_diff_norm, _ = l2_norms(mesh, E_e, E_ana, mask=edge_mask)
        B_diff_norm, _ = l2_norms(mesh, B_f, B_ana, mask=face_mask)

        # Normalize by the temporal-max amplitude — meaningful for all phases
        E_err = E_diff_norm / E_amp_norm if E_amp_norm > 0 else E_diff_norm
        B_err = B_diff_norm / B_amp_norm if B_amp_norm > 0 else B_diff_norm
        times.append(t)
        E_errs.append(E_err)
        B_errs.append(B_err)
        if verbose:
            print(f"  step {step:6d}  t = {t:8.4f}  "
                  f"t/T = {t/period:6.3f}  "
                  f"L2(E)/|E_max| = {E_err:.4e}  "
                  f"L2(B)/|B_max| = {B_err:.4e}")
    return np.array(times), np.array(E_errs), np.array(B_errs), period


def convergence_study(data_dirs, l, m, n_root, polarization, start_with_e,
                       amp=1.0, plot_path=None):
    """Run analyze_run on each directory and plot error vs h."""
    Ls, hs, E_finals, B_finals = [], [], [], []
    for d in data_dirs:
        print(f"\n=== {d} ===")
        mesh = load_mesh(d)
        L_val = mesh["L"]
        # Approximate h: mean angular edge length on the inner shell
        N_edge_s = mesh["N_edge_s"]
        h = float(np.mean(mesh["edge_length"][:N_edge_s])
                  / mesh["radii"][0])  # angular size in radians
        times, Ee, Be, T = analyze_run(d, l, m, n_root, polarization,
                                        start_with_e, amp=amp, verbose=True)
        Ls.append(L_val)
        hs.append(h)
        E_finals.append(Ee[-1])
        B_finals.append(Be[-1])

    Ls = np.array(Ls)
    hs = np.array(hs)
    E_finals = np.array(E_finals)
    B_finals = np.array(B_finals)

    print("\n=== Convergence summary ===")
    print(f"{'L':>4} {'h(rad)':>10} {'L2(E)':>14} {'L2(B)':>14}")
    for L, h, eE, eB in zip(Ls, hs, E_finals, B_finals):
        print(f"{L:>4} {h:>10.4f} {eE:>14.4e} {eB:>14.4e}")

    # Slopes (least squares fit log-log)
    if len(Ls) >= 2:
        slope_E = np.polyfit(np.log(hs), np.log(E_finals), 1)[0]
        slope_B = np.polyfit(np.log(hs), np.log(B_finals), 1)[0]
        print(f"\nConvergence rates: E -> {slope_E:.3f}, B -> {slope_B:.3f}")

    if plot_path:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(hs, E_finals, 'o-', label=f'E (slope ~ {slope_E:.2f})')
        ax.loglog(hs, B_finals, 's-', label=f'B (slope ~ {slope_B:.2f})')
        # Reference O(h) and O(h^2)
        ref_h = np.array([hs.min(), hs.max()])
        ax.loglog(ref_h, B_finals[-1] * (ref_h / hs[-1])**1, 'k--',
                   alpha=0.4, label='O(h)')
        ax.loglog(ref_h, B_finals[-1] * (ref_h / hs[-1])**2, 'k:',
                   alpha=0.4, label='O(h²)')
        ax.set_xlabel("h (angular edge length, rad)")
        ax.set_ylabel("Relative L2 error")
        ax.set_title(f"Cavity {polarization}_{l}_{m}_{n_root} convergence")
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        print(f"\nPlot saved to {plot_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dirs", nargs='+', help="One or more output directories")
    p.add_argument("--l", type=int, default=1)
    p.add_argument("--m", type=int, default=0)
    p.add_argument("--n_root", type=int, default=1)
    p.add_argument("--pol", choices=['E', 'M'], default='E',
                    help="E for TE, M for TM")
    p.add_argument("--start_with_e", action='store_true',
                    help="If set, t=0 has E max (default: B max)")
    p.add_argument("--amp", type=float, default=1.0)
    p.add_argument("--convergence", action='store_true',
                    help="Treat all data_dirs as a convergence series")
    p.add_argument("--plot", default=None,
                    help="Save plot to this path (convergence mode)")
    args = p.parse_args()

    if args.convergence or len(args.data_dirs) > 1:
        convergence_study(args.data_dirs, args.l, args.m, args.n_root,
                           args.pol, args.start_with_e, amp=args.amp,
                           plot_path=args.plot)
    else:
        analyze_run(args.data_dirs[0], args.l, args.m, args.n_root,
                     args.pol, args.start_with_e, amp=args.amp)


if __name__ == "__main__":
    main()
