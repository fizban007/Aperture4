#!/usr/bin/env python
"""Near-surface tangential-E diagnostic for the prismatic fake-GR runs.

For each radial shell k, restrict to h-edges whose midpoint colatitude is in a
band, split them into theta-like and phi-like by tangent direction, and compare
the measured circulation E_e against the analytic corotation circulation:
  flat: E = -(Omega zhat x r) x B_dip
  MT  : E = -((Omega - w_lt(r)) zhat x r) x B_dip,  w_lt = w0 (1/r)^p
Circulations are evaluated with 4-pt Gauss along the slerp arc.
"""
import sys
import numpy as np
import h5py

MZ = 1000.0
OMEGA = 0.25
W0 = 0.4 * 0.5 * OMEGA   # (2/5) * compactness * Omega = 0.05
LTP = 3

def dipole_B(x):
    r2 = (x * x).sum(-1)
    r = np.sqrt(r2)
    r3 = r2 * r
    r5 = r2 * r3
    mdotr = MZ * x[..., 2]
    B = 3.0 * mdotr[..., None] * x / r5[..., None]
    B[..., 2] -= MZ / r3
    return B

def corot_E(x, sub_lt):
    r = np.sqrt((x * x).sum(-1))
    om = OMEGA - (W0 / r**LTP if sub_lt else 0.0)
    v = np.stack([-om * x[..., 1], om * x[..., 0], np.zeros_like(r)], -1)
    return -np.cross(v, dipole_B(x))

def analyze(run, step, kmax=14, band=(10.0, 25.0)):
    out = f"{run}/output"
    with h5py.File(f"{out}/mesh.h5", "r") as g:
        V = np.stack([g["sphere_vx"][()], g["sphere_vy"][()],
                      g["sphere_vz"][()]], -1).astype(np.float64)
        e0 = g["sphere_edge_v0"][()]
        e1 = g["sphere_edge_v1"][()]
        radii = g["radii"][()].astype(np.float64)
        NEs = int(g["N_edge_s"][()])
    with h5py.File(f"{out}/step_{step:06d}.h5", "r") as f:
        E_e = f["E_e"][()]
        t = float(f["time"][()])

    a = V[e0]; b = V[e1]
    mid = a + b
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    colat = np.degrees(np.arccos(np.clip(mid[:, 2], -1, 1)))
    sel = ((colat > band[0]) & (colat < band[1])) | \
          ((colat > 180 - band[1]) & (colat < 180 - band[0]))

    # tangent at midpoint (chord direction, unitized), theta-like vs phi-like
    chord = b - a
    chord /= np.linalg.norm(chord, axis=1, keepdims=True)
    zhat = np.array([0.0, 0.0, 1.0])
    that = zhat - mid * mid[:, 2:3]        # theta direction (un-normalized)
    that /= np.linalg.norm(that, axis=1, keepdims=True)
    tdot = np.abs((chord * that).sum(1))
    th_like = sel & (tdot > 0.7)
    ph_like = sel & (tdot < 0.3)

    # 4-pt Gauss along slerp
    gx = np.array([0.069431844202973712, 0.33000947820757187,
                   0.66999052179242813, 0.93056815579702623])
    gw = np.array([0.1739274225687269, 0.3260725774312731,
                   0.3260725774312731, 0.1739274225687269])
    ang = np.arccos(np.clip((a * b).sum(1), -1, 1))

    print(f"# {run.split('/')[-1]}  step {step}  t={t:.2f}  t/P={t*OMEGA/2/np.pi:.2f}")
    print(f"# colat band {band} deg;  N_theta-like={th_like.sum()}, N_phi-like={ph_like.sum()}")
    hdr = ("k", "r", "<|E|>th", "<|E_MT|>th", "rat_th", "<|E|>ph", "<|Eph_res|>/MT")
    print(("{:>3} {:>7} " + "{:>11} " * 5).format(*hdr))
    for k in range(kmax + 1):
        r = radii[k]
        Ek = E_e[k * NEs:(k + 1) * NEs].astype(np.float64)
        for tag, mask in (("th", th_like), ("ph", ph_like)):
            pass
        # analytic circulations along the arc at radius r
        def circ(sub_lt, mask):
            am, bm = a[mask], b[mask]
            angm = ang[mask][:, None]
            tt = gx[None, :]
            # slerp points and tangents
            s = np.sin(angm)
            s0 = np.sin((1 - tt) * angm) / s
            s1 = np.sin(tt * angm) / s
            u = s0[..., None] * am[:, None, :] + s1[..., None] * bm[:, None, :]
            du = (-angm * np.cos((1 - tt) * angm) / s)[..., None] * am[:, None, :] \
               + (angm * np.cos(tt * angm) / s)[..., None] * bm[:, None, :]
            x = r * u
            Ean = corot_E(x, sub_lt)
            integ = (Ean * (r * du)).sum(-1)
            return (integ * gw[None, :]).sum(1)
        cM_th = circ(True, th_like)
        cF_th = circ(False, th_like)
        m_th = Ek[th_like]
        m_ph = Ek[ph_like]
        cM_ph = circ(True, ph_like)
        # ratio of mean abs
        r_th = np.abs(m_th).mean() / np.abs(cM_th).mean()
        scale = np.abs(cM_th).mean()
        print(f"{k:3d} {r:7.4f} {np.abs(m_th).mean():11.4g} "
              f"{np.abs(cM_th).mean():11.4g} {r_th:11.4f} "
              f"{np.abs(m_ph).mean():11.4g} {np.abs(m_ph - cM_ph).mean()/scale:11.4f}")
    # signed comparison at k of interest: fraction relative between flat and MT
    print("# signed mean ratios (measured/analytic), theta-like edges:")
    print(("{:>3} {:>7} " + "{:>11} " * 3).format("k", "r", "vs_MT", "vs_flat", "omeff/Om"))
    for k in range(kmax + 1):
        r = radii[k]
        Ek = E_e[k * NEs:(k + 1) * NEs].astype(np.float64)
        def circ(sub_lt, mask):
            am, bm = a[mask], b[mask]
            angm = ang[mask][:, None]
            tt = gx[None, :]
            s = np.sin(angm)
            s0 = np.sin((1 - tt) * angm) / s
            s1 = np.sin(tt * angm) / s
            u = s0[..., None] * am[:, None, :] + s1[..., None] * bm[:, None, :]
            du = (-angm * np.cos((1 - tt) * angm) / s)[..., None] * am[:, None, :] \
               + (angm * np.cos(tt * angm) / s)[..., None] * bm[:, None, :]
            x = r * u
            Ean = corot_E(x, sub_lt)
            integ = (Ean * (r * du)).sum(-1)
            return (integ * gw[None, :]).sum(1)
        cM = circ(True, th_like)
        cF = circ(False, th_like)
        m = Ek[th_like]
        # least-squares scalar fit m ~ s * c
        sM = (m * cM).sum() / (cM * cM).sum()
        sF = (m * cF).sum() / (cF * cF).sum()
        # infer effective rotation rate: m ~ -(om_eff) ... using linearity:
        # circ scales linearly with om; c(om) = cF * om_eff/Omega when w0=0
        om_eff = sF  # since cF ∝ Omega
        print(f"{k:3d} {r:7.4f} {sM:11.4f} {sF:11.4f} {om_eff:11.4f}")

if __name__ == "__main__":
    analyze(sys.argv[1], int(sys.argv[2]))
