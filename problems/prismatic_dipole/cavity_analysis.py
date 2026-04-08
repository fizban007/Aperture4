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
        # Optional structured output downsampling. The writer applies
        # independent radial / angular strides over (k, sub-element);
        # when either is > 1, snapshots store sparse cochain arrays in
        # 1-to-1 order with output_edge_idx / output_face_idx, and the
        # downsampled mesh's unique vertex set is also recorded.
        mesh["output_radial_stride"]  = (int(f["output_radial_stride"][()])
                                         if "output_radial_stride"  in f else 1)
        mesh["output_angular_stride"] = (int(f["output_angular_stride"][()])
                                         if "output_angular_stride" in f else 1)
        # Combined factor used by analyze_run as a "downsampling is on"
        # flag (1 ⇒ full output).
        mesh["output_subsample"] = (mesh["output_radial_stride"] *
                                    mesh["output_angular_stride"])
        if "output_edge_idx" in f:
            mesh["output_edge_idx"] = f["output_edge_idx"][:]
        if "output_face_idx" in f:
            mesh["output_face_idx"] = f["output_face_idx"][:]
        if "output_vert_idx" in f:
            mesh["output_vert_idx"] = f["output_vert_idx"][:]
    return mesh


def load_fields(data_dir, step, mesh=None):
    """Load (B_f, E_e, t) for a given step.

    If `mesh` is provided AND `mesh['output_subsample'] > 1`, the snapshot
    arrays on disk are sparse (1-to-1 with `output_face_idx` /
    `output_edge_idx`) — they are scattered back into full-length arrays
    so the rest of the analysis pipeline (which addresses by global index)
    works unchanged. Positions not retained on disk remain zero, so any
    analysis subset must be a subset of the saved indices.
    """
    with h5py.File(os.path.join(data_dir, f"step_{step:06d}.h5")) as f:
        B_f = f["B_f"][:]
        E_e = f["E_e"][:]
        t = float(f["time"][()])
    if mesh is not None and mesh.get("output_subsample", 1) > 1:
        B_full = np.zeros(mesh["N_faces"], dtype=B_f.dtype)
        E_full = np.zeros(mesh["N_edges"], dtype=E_e.dtype)
        B_full[mesh["output_face_idx"]] = B_f
        E_full[mesh["output_edge_idx"]] = E_e
        return B_full, E_full, t
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

def _gauss_legendre_01(n):
    """n-point Gauss-Legendre nodes and weights on [0, 1]."""
    xs, ws = np.polynomial.legendre.leggauss(n)
    return 0.5 * (xs + 1.0), 0.5 * ws


# Default: 5-point Gauss is exact for degree-9 polynomials. Far more than
# needed for the smooth analytic eigenmodes — but cheap enough that lowering
# this further only saves a few seconds. Override via --quad_order.
_u01, _w01 = _gauss_legendre_01(5)


def set_quad_order(n):
    """Reset the global Gauss quadrature order (n nodes per direction)."""
    global _u01, _w01
    _u01, _w01 = _gauss_legendre_01(n)


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

def analytic_cochains(mesh, mode_args, t,
                       face_idx=None, edge_idx=None):
    """Compute (E_e_analytic, B_f_analytic) by Gauss quadrature of the
    analytic eigenmode at time t.

    If face_idx / edge_idx are given, only those (global) face / edge
    indices are evaluated; the returned arrays are indexed by the same
    subset rather than by full mesh index. This is the subsample fast
    path used by analyze_run when --subsample > 1."""
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

    if face_idx is None:
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
    else:
        # Subsample path: evaluate only the requested face indices.
        face_idx = np.asarray(face_idx)
        is_tri = face_idx < n_tri_faces
        tri_local = face_idx[is_tri]
        rect_local = face_idx[~is_tri] - n_tri_faces

        flux = np.zeros(face_idx.shape, dtype=float)
        if tri_local.size > 0:
            p0 = _get_verts(mesh, mesh["tri_face_v0"][tri_local])
            p1 = _get_verts(mesh, mesh["tri_face_v1"][tri_local])
            p2 = _get_verts(mesh, mesh["tri_face_v2"][tri_local])
            flux[is_tri] = analytic_face_flux_tri(p0, p1, p2, B_field)
        if rect_local.size > 0:
            p0 = _get_verts(mesh, mesh["rect_face_v0"][rect_local])
            p1 = _get_verts(mesh, mesh["rect_face_v1"][rect_local])
            p2 = _get_verts(mesh, mesh["rect_face_v2"][rect_local])
            p3 = _get_verts(mesh, mesh["rect_face_v3"][rect_local])
            flux[~is_tri] = analytic_face_flux_quad(p0, p1, p2, p3, B_field)
        B_f = flux

    if edge_idx is None:
        x0 = _get_verts(mesh, mesh["edge_v0"])
        x1 = _get_verts(mesh, mesh["edge_v1"])
        E_e = analytic_edge_circ(x0, x1, E_field)
    else:
        edge_idx = np.asarray(edge_idx)
        x0 = _get_verts(mesh, mesh["edge_v0"][edge_idx])
        x1 = _get_verts(mesh, mesh["edge_v1"][edge_idx])
        E_e = analytic_edge_circ(x0, x1, E_field)

    return E_e, B_f


# =========================================================================
# Error metrics
# =========================================================================

def _hodge_face_weight(mesh, face_idx=None):
    w = mesh["hodge2"]
    return w if face_idx is None else w[face_idx]


def _hodge_edge_weight(mesh, edge_idx=None):
    h1inv = mesh["hodge1_inv"]
    w = np.where(h1inv > 0, 1.0 / np.where(h1inv > 0, h1inv, 1.0), 0.0)
    return w if edge_idx is None else w[edge_idx]


def l2_norms(mesh, A_num, A_ana, mask=None):
    """Discrete L2 norms of (A_num - A_ana) and A_ana, induced by the
    diagonal DEC Hodge star.

    For face cochains:
        ||B||^2 = Σ_f hodge2[f] * B_f^2     ≈ ∫|B|^2 dV
    For edge cochains:
        ||E||^2 = Σ_e (1/hodge1_inv[e]) * E_e^2     ≈ ∫|E|^2 dV

    Returns (||diff||, ||A_ana||) — caller decides how to normalize.
    """
    if A_num.shape == mesh["face_area"].shape:
        weights = _hodge_face_weight(mesh)
    else:
        weights = _hodge_edge_weight(mesh)

    d2 = weights * (A_num - A_ana) ** 2
    a2 = weights * A_ana ** 2

    if mask is not None:
        d2 = d2[mask]
        a2 = a2[mask]
    return float(np.sqrt(d2.sum())), float(np.sqrt(a2.sum()))


def l2_norms_subset(weights_sub, A_num_sub, A_ana_sub):
    """L2 norms on a pre-selected subset (weights and values aligned)."""
    d2 = weights_sub * (A_num_sub - A_ana_sub) ** 2
    a2 = weights_sub * A_ana_sub ** 2
    return float(np.sqrt(d2.sum())), float(np.sqrt(a2.sum()))


# =========================================================================
# Main entry points
# =========================================================================

def _infer_dt(data_dir, steps):
    """Infer the simulation dt from the first non-zero snapshot.
    Returns dt or None if not deducible."""
    for step in steps:
        if step == 0:
            continue
        # Don't pass mesh — we only need the timestamp, not the cochains.
        with h5py.File(os.path.join(data_dir, f"step_{step:06d}.h5")) as f:
            t = float(f["time"][()])
        if step > 0 and t > 0:
            return float(t) / float(step)
    return None


def analyze_run(data_dir, l, m, n_root, polarization, start_with_e,
                amp=1.0, verbose=True, b_half_shift=False, subsample=1):
    """Compare every snapshot in data_dir against the analytic mode.

    If b_half_shift is True, evaluates the analytic B at (t - dt/2) instead
    of t. In the Yee/leapfrog interpretation E lives at integer times t^n
    and B lives at half-integer times t^{n-1/2}, so the snapshot's B is
    naturally half a step behind its timestamp. Shifting removes the
    O(dt) bias from the comparison and exposes the underlying
    spatial+leapfrog 2nd-order error.

    If subsample > 1, the L2 norm is computed only on every-Nth interior
    face / edge — by far the dominant cost. For smooth fields the relative
    error is essentially unchanged because both numerator and denominator
    use the same subset.

    Returns (times, E_errors, B_errors, period)."""
    mesh = load_mesh(data_dir)
    a = float(mesh["radii"][0])
    b = float(mesh["radii"][-1])
    k = cavity_eigenvalue(l, a, b, n_root, polarization)
    alpha = inner_bc_alpha(l, k, a, polarization)
    omega = k
    period = 2 * np.pi / omega

    steps = list_steps(data_dir)
    if not steps:
        raise RuntimeError(f"No step files in {data_dir}")

    dt = _infer_dt(data_dir, steps) if b_half_shift else None
    b_shift = -0.5 * dt if (b_half_shift and dt is not None) else 0.0

    # Pre-compute the index subset (interior elements, every-N-th).
    # If the writer subsampled the output (mesh["output_subsample"] > 1),
    # the indices we score on must be a subset of what was actually saved
    # to disk — anything outside `output_*_idx` is zero in the loaded
    # snapshots and would silently bias the L2 error.
    out_sub = mesh.get("output_subsample", 1)
    if out_sub > 1:
        saved_face_mask = np.zeros(mesh["N_faces"], dtype=bool)
        saved_face_mask[mesh["output_face_idx"]] = True
        saved_edge_mask = np.zeros(mesh["N_edges"], dtype=bool)
        saved_edge_mask[mesh["output_edge_idx"]] = True
        interior_face_idx = np.where((mesh["face_boundary"] == 0) & saved_face_mask)[0]
        interior_edge_idx = np.where((mesh["edge_boundary"] == 0) & saved_edge_mask)[0]
    else:
        interior_face_idx = np.where(mesh["face_boundary"] == 0)[0]
        interior_edge_idx = np.where(mesh["edge_boundary"] == 0)[0]
    if subsample > 1:
        face_idx = interior_face_idx[::subsample]
        edge_idx = interior_edge_idx[::subsample]
    else:
        face_idx = interior_face_idx
        edge_idx = interior_edge_idx

    face_w = _hodge_face_weight(mesh, face_idx)
    edge_w = _hodge_edge_weight(mesh, edge_idx)

    if verbose:
        print(f"Cavity: r_min={a}, r_max={b}")
        print(f"Mode: {polarization} l={l} m={m} n_root={n_root}")
        print(f"  k = {k:.8f}, omega = {omega:.8f}, period T = {period:.6f}")
        print(f"  alpha = {alpha:.6e}")
        if b_half_shift:
            print(f"  inferred dt = {dt:.6e}, B compared at t {b_shift:+.4e}")
        if out_sub > 1:
            print(f"  output downsample = {out_sub} (writer kept "
                  f"{len(mesh['output_face_idx'])}/{mesh['N_faces']} faces, "
                  f"{len(mesh['output_edge_idx'])}/{mesh['N_edges']} edges)")
        print(f"  scoring on {len(face_idx)}/{len(interior_face_idx)} interior "
              f"faces, {len(edge_idx)}/{len(interior_edge_idx)} interior edges "
              f"(analysis subsample={subsample})")

    mode_args = (l, m, k, alpha, amp, polarization, start_with_e)

    # Reference amplitude (uses the same subset).
    t_Bmax = 0.0 if not start_with_e else period / 4.0
    t_Emax = period / 4.0 if not start_with_e else 0.0
    _, B_ana_max = analytic_cochains(mesh, mode_args, t_Bmax,
                                       face_idx=face_idx, edge_idx=edge_idx)
    E_ana_max, _ = analytic_cochains(mesh, mode_args, t_Emax,
                                       face_idx=face_idx, edge_idx=edge_idx)
    _, B_amp_norm = l2_norms_subset(face_w, np.zeros_like(B_ana_max), B_ana_max)
    _, E_amp_norm = l2_norms_subset(edge_w, np.zeros_like(E_ana_max), E_ana_max)
    if verbose:
        print(f"  reference ||B||@max = {B_amp_norm:.4e}, "
              f"||E||@max = {E_amp_norm:.4e}")

    times, E_errs, B_errs = [], [], []
    for step in steps:
        B_f, E_e, t = load_fields(data_dir, step, mesh=mesh)
        # Evaluate analytic E at t, analytic B at (t + b_shift)
        E_ana, _ = analytic_cochains(mesh, mode_args, t,
                                      face_idx=face_idx, edge_idx=edge_idx)
        _, B_ana = analytic_cochains(mesh, mode_args, t + b_shift,
                                      face_idx=face_idx, edge_idx=edge_idx)

        E_diff_norm, _ = l2_norms_subset(edge_w, E_e[edge_idx], E_ana)
        B_diff_norm, _ = l2_norms_subset(face_w, B_f[face_idx], B_ana)

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
                       amp=1.0, plot_path=None, b_half_shift=False,
                       subsample=1, target_samples=None, metric='peak'):
    """Run analyze_run on each directory and plot error vs h.

    metric:
      'peak'  - max error over the period (most robust)
      'mean'  - time-averaged error over the period
      'final' - error at the last snapshot (was the original; least robust
                because the last snapshot may not land at the same t/T
                across resolutions)
    """
    if metric not in ('peak', 'mean', 'final'):
        raise ValueError(f"metric must be peak/mean/final, got {metric}")

    Ls, hs, E_metrics, B_metrics = [], [], [], []
    for d in data_dirs:
        print(f"\n=== {d} ===")
        mesh = load_mesh(d)
        L_val = mesh["L"]
        # Approximate h: mean angular edge length on the inner shell
        N_edge_s = mesh["N_edge_s"]
        h = float(np.mean(mesh["edge_length"][:N_edge_s])
                  / mesh["radii"][0])  # angular size in radians
        # Auto-pick subsample to keep ~target_samples interior faces in the
        # L2 sum (analysis cost is then roughly L-independent).
        if target_samples is not None and target_samples > 0:
            n_interior = int(np.sum(mesh["face_boundary"] == 0))
            sub = max(1, n_interior // int(target_samples))
        else:
            sub = subsample
        times, Ee, Be, T = analyze_run(d, l, m, n_root, polarization,
                                        start_with_e, amp=amp, verbose=True,
                                        b_half_shift=b_half_shift,
                                        subsample=sub)
        # Drop step 0 (essentially-zero IC error) when computing peak/mean
        # so a near-zero IC value doesn't dominate the average.
        Ee_eval = Ee[1:] if len(Ee) > 1 else Ee
        Be_eval = Be[1:] if len(Be) > 1 else Be
        if metric == 'peak':
            E_val = float(np.max(Ee_eval))
            B_val = float(np.max(Be_eval))
        elif metric == 'mean':
            E_val = float(np.mean(Ee_eval))
            B_val = float(np.mean(Be_eval))
        else:  # final
            E_val = float(Ee[-1])
            B_val = float(Be[-1])

        Ls.append(L_val)
        hs.append(h)
        E_metrics.append(E_val)
        B_metrics.append(B_val)

    Ls = np.array(Ls)
    hs = np.array(hs)
    E_metrics = np.array(E_metrics)
    B_metrics = np.array(B_metrics)

    print(f"\n=== Convergence summary ({metric} over period) ===")
    print(f"{'L':>4} {'h(rad)':>10} {'L2(E)':>14} {'L2(B)':>14}")
    for L, h, eE, eB in zip(Ls, hs, E_metrics, B_metrics):
        print(f"{L:>4} {h:>10.4f} {eE:>14.4e} {eB:>14.4e}")

    # Slopes (least squares fit log-log)
    slope_E = slope_B = None
    if len(Ls) >= 2:
        slope_E = np.polyfit(np.log(hs), np.log(E_metrics), 1)[0]
        slope_B = np.polyfit(np.log(hs), np.log(B_metrics), 1)[0]
        print(f"\nConvergence rates: E -> {slope_E:.3f}, B -> {slope_B:.3f}")

        # Pairwise rates
        if len(Ls) >= 3:
            print("Pairwise slopes:")
            for i in range(len(Ls) - 1):
                seg_E = np.log(E_metrics[i+1]/E_metrics[i]) / np.log(hs[i+1]/hs[i])
                seg_B = np.log(B_metrics[i+1]/B_metrics[i]) / np.log(hs[i+1]/hs[i])
                print(f"  L{Ls[i]}->L{Ls[i+1]}:  E={seg_E:.3f}  B={seg_B:.3f}")

    if plot_path:
        fig, ax = plt.subplots(figsize=(7, 5))
        e_lbl = f'E (slope ~ {slope_E:.2f})' if slope_E is not None else 'E'
        b_lbl = f'B (slope ~ {slope_B:.2f})' if slope_B is not None else 'B'
        ax.loglog(hs, E_metrics, 'o-', label=e_lbl)
        ax.loglog(hs, B_metrics, 's-', label=b_lbl)
        ref_h = np.array([hs.min(), hs.max()])
        ax.loglog(ref_h, B_metrics[-1] * (ref_h / hs[-1])**1, 'k--',
                   alpha=0.4, label='O(h)')
        ax.loglog(ref_h, B_metrics[-1] * (ref_h / hs[-1])**2, 'k:',
                   alpha=0.4, label='O(h²)')
        ax.set_xlabel("h (angular edge length, rad)")
        ax.set_ylabel(f"Relative L2 error ({metric} over period)")
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
    p.add_argument("--b_half_shift", action='store_true',
                    help="Compare numerical B against analytic B at t-dt/2 "
                         "(removes the leapfrog half-step bias)")
    p.add_argument("--subsample", type=int, default=1,
                    help="Score the L2 norm on every Nth interior face/edge "
                         "(default 1 = all). The relative error is essentially "
                         "unchanged for smooth fields.")
    p.add_argument("--target_samples", type=int, default=None,
                    help="Convergence mode: auto-pick subsample per dir so "
                         "that ~target_samples interior faces are scored "
                         "(keeps analysis cost ~L-independent). Overrides "
                         "--subsample for convergence runs.")
    p.add_argument("--quad_order", type=int, default=5,
                    help="Gauss quadrature nodes per direction (default 5).")
    p.add_argument("--metric", choices=['peak', 'mean', 'final'],
                    default='peak',
                    help="Convergence metric: peak / mean / final-snapshot. "
                         "'peak' (default) is most robust to phase mismatches.")
    args = p.parse_args()

    set_quad_order(args.quad_order)

    if args.convergence or len(args.data_dirs) > 1:
        convergence_study(args.data_dirs, args.l, args.m, args.n_root,
                           args.pol, args.start_with_e, amp=args.amp,
                           plot_path=args.plot,
                           b_half_shift=args.b_half_shift,
                           subsample=args.subsample,
                           target_samples=args.target_samples,
                           metric=args.metric)
    else:
        analyze_run(args.data_dirs[0], args.l, args.m, args.n_root,
                     args.pol, args.start_with_e, amp=args.amp,
                     b_half_shift=args.b_half_shift,
                     subsample=args.subsample)


if __name__ == "__main__":
    main()
