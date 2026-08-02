"""Sec 6.6 gather-battery scan: primal Whitney vs vertex-recovery B-gather.

Extends recovery_study.py to the paper-figure battery:
  - levels L = 3, 4, 5 (meshes r in [1,2], N_r = 4*2^(L-2), from
    problems/prismatic_dipole/bin/vacuum_dipole 1-step runs),
  - gyro-radius / cell-size scan r_gyro/h in {1/8, 1/4, 1/2},
  - uniform-B leg (recovery exact by construction -> control) and the
    discriminating dipole-orbit leg,
  - static gather-error convergence on the dipole field (primal 1st vs
    recovery 2nd order),
  - conditioning summary per level (valence-5 vs valence-6).

Usage:
    python recovery_scan.py '<mesh_dir_pattern_with_{L}>' <out_dir> [L ...]

Writes <out_dir>/recovery_scan.npz and prints the summary tables.
"""

import sys
import time
import numpy as np
import prismatic_recovery as pr
from recovery_study import run_orbit, mu_of

RATIOS = (0.125, 0.25, 0.5)


def h_of_shell(mesh, k):
    """Median horizontal edge length on shell k."""
    return float(np.median(mesh.edge_length[k * mesh.N_edge_s:(k + 1) * mesh.N_edge_s]))


def h_at_r(mesh, r):
    k = int(np.clip(np.searchsorted(mesh.radii, r), 0, mesh.N_r))
    return h_of_shell(mesh, k)


def make_fields(mesh, Ee, Bf, Bv, exact):
    return {
        "exact": lambda x, h: exact(x),
        "primal": lambda x, h: pr.primal_gather(mesh, Ee, Bf, x, h)[1],
        "recovery": lambda x, h: pr.recovery_gather(mesh, Bv, x, h),
    }


def orbit_metrics(mesh, hist, mu_ref_field, n_avg=8):
    """Loss fraction, phase-averaged dmu/mu (survivors), gc wander."""
    xs = np.array([h[1] for h in hist])
    r_hist = np.linalg.norm(xs, axis=2)
    lost = (r_hist < mesh.radii[0]) | (r_hist > mesh.radii[-1])
    alive = ~np.maximum.accumulate(lost, axis=0)
    ok = alive[-1]
    loss_frac = 1.0 - ok.mean()
    mu = np.array([mu_of(v, mu_ref_field(x)) for _, x, v, B in hist])[:, ok]
    mu0 = mu[:n_avg].mean(axis=0)
    mu1 = mu[-n_avg:].mean(axis=0)
    dmu_rms = float(np.sqrt(np.nanmean(((mu1 - mu0) / mu0) ** 2)))
    gc0 = xs[:2 * n_avg, ok].mean(axis=0)
    gc1 = xs[-2 * n_avg:, ok].mean(axis=0)
    wander = float(np.median(np.linalg.norm(gc1 - gc0, axis=1)))
    return dict(loss_frac=float(loss_frac), dmu_rms=dmu_rms, gc_wander=wander)


def uniform_leg(mesh, Bf, Bv, ratio, rng, n_ptc=48, n_steps=3000, dt_frac=0.05):
    B0 = np.array([0.0, 0.0, 1.0])
    Ee = np.zeros(mesh.N_edges)
    r_gyro = ratio * h_at_r(mesh, 1.5)  # omega_c = |B0| = 1 -> r_gyro = v_perp
    r0 = rng.uniform(1.3, 1.7, n_ptc)
    u = rng.normal(size=(n_ptc, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    x0 = r0[:, None] * u
    v0 = np.cross(B0[None, :], rng.normal(size=(n_ptc, 3)))
    v0 /= np.linalg.norm(v0, axis=1, keepdims=True)
    v0 = r_gyro * v0
    dt = dt_frac * 2 * np.pi
    out = {}
    for name, f in make_fields(mesh, Ee, Bf, Bv, pr.constant_B(B0)).items():
        hist = run_orbit(x0, v0, f, dt, n_steps)
        out[name] = orbit_metrics(mesh, hist, pr.constant_B(B0))
    return r_gyro, out


def dipole_leg(mesh, Bf, Bv, ratio, rng, n_ptc=32, n_steps=6000, dt_frac=0.02):
    Ee = np.zeros(mesh.N_edges)
    r0 = rng.uniform(1.3, 1.6, n_ptc)
    phi0 = rng.uniform(0, 2 * np.pi, n_ptc)
    th0 = np.pi / 2 + rng.uniform(-0.15, 0.15, n_ptc)
    x0 = np.stack([r0 * np.sin(th0) * np.cos(phi0),
                   r0 * np.sin(th0) * np.sin(phi0),
                   r0 * np.cos(th0)], axis=1)
    B_at = pr.dipole_B(x0)
    Bn = np.linalg.norm(B_at, axis=1)
    b = B_at / Bn[:, None]
    h_loc = np.array([h_at_r(mesh, r) for r in r0])
    v_perp = ratio * h_loc * Bn  # r_gyro = v_perp/|B| = ratio * h_local
    e1 = np.cross(b, rng.normal(size=(n_ptc, 3)))
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    v0 = v_perp[:, None] * e1 + (0.6 * v_perp)[:, None] * b
    dt = dt_frac * 2 * np.pi / Bn.max()
    out = {}
    for name, f in make_fields(mesh, Ee, Bf, Bv, pr.dipole_B).items():
        hist = run_orbit(x0, v0, f, dt, n_steps)
        out[name] = orbit_metrics(mesh, hist, pr.dipole_B)
    return out


def static_error(mesh, Bf, Bv, rng, n_pts=4000):
    """RMS relative gather error vs the exact dipole at random points."""
    u = rng.normal(size=(n_pts, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    pts = rng.uniform(1.2, 1.8, n_pts)[:, None] * u
    exact = pr.dipole_B(pts)
    Ee = np.zeros(mesh.N_edges)
    ref = np.linalg.norm(exact, axis=1)
    out = {}
    for name, B in (("primal", pr.primal_gather(mesh, Ee, Bf, pts)[1]),
                    ("recovery", pr.recovery_gather(mesh, Bv, pts))):
        err = np.linalg.norm(B - exact, axis=1) / ref
        out[name] = float(np.sqrt(np.mean(err ** 2)))
    return out


def main():
    pattern = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    Ls = [int(a) for a in sys.argv[3:]] or [3, 4, 5]
    rng = np.random.default_rng(2026)
    res = {"ratios": np.array(RATIOS), "Ls": np.array(Ls)}

    for L in Ls:
        t0 = time.time()
        mesh = pr.Mesh(pattern.format(L=L) + "/mesh.h5")
        h_mid = h_at_r(mesh, 1.5)
        res[f"L{L}_h_mid"] = h_mid
        print(f"== L={L}  N_r={mesh.N_r}  h_mid={h_mid:.4f} ==", flush=True)

        Bf_uni = pr.flux_cochain(mesh, pr.constant_B(np.array([0., 0., 1.])), n=6)
        Bf_dip = pr.flux_cochain(mesh, pr.dipole_B, n=6)
        rec = pr.build_recovery(mesh, n=4)
        cond = rec["cond"].reshape(mesh.N_r + 1, mesh.N_vert_s)
        for cls, mask in (("v6", mesh.valence == 6), ("v5", mesh.valence == 5)):
            res[f"L{L}_cond_{cls}_med"] = float(np.median(cond[1:-1, mask]))
            res[f"L{L}_cond_{cls}_max"] = float(cond[1:-1, mask].max())
        Bv_uni = pr.vertex_field(mesh, rec, Bf_uni)
        Bv_dip = pr.vertex_field(mesh, rec, Bf_dip)

        se = static_error(mesh, Bf_dip, Bv_dip, rng)
        res[f"L{L}_static_primal"] = se["primal"]
        res[f"L{L}_static_recovery"] = se["recovery"]
        print(f"  static dipole gather rms err: primal {se['primal']:.3e}  "
              f"recovery {se['recovery']:.3e}", flush=True)

        for i, ratio in enumerate(RATIOS):
            r_gyro, uni = uniform_leg(mesh, Bf_uni, Bv_uni, ratio, rng)
            dip = dipole_leg(mesh, Bf_dip, Bv_dip, ratio, rng)
            for name in ("exact", "primal", "recovery"):
                for q, v in uni[name].items():
                    res[f"L{L}_r{i}_uniform_{name}_{q}"] = v
                for q, v in dip[name].items():
                    res[f"L{L}_r{i}_dipole_{name}_{q}"] = v
            print(f"  ratio {ratio:5.3f} (r_gyro {r_gyro:.4f}):", flush=True)
            for leg, d in (("uniform", uni), ("dipole", dip)):
                print("    %-7s " % leg + "  ".join(
                    f"{n}: lost {100*d[n]['loss_frac']:4.1f}% dmu {d[n]['dmu_rms']:.2e}"
                    for n in ("primal", "recovery", "exact")), flush=True)
        print(f"  [L={L} done in {time.time()-t0:.0f} s]", flush=True)

    np.savez(f"{out_dir}/recovery_scan.npz", **res)
    print("saved", f"{out_dir}/recovery_scan.npz")

    print("\n== static gather-error convergence (dipole field) ==")
    for name in ("primal", "recovery"):
        errs = [res[f"L{L}_static_{name}"] for L in Ls]
        line = "  ".join(f"L{L} {e:.3e}" for L, e in zip(Ls, errs))
        ratios = "  ".join(f"{errs[i]/errs[i+1]:.2f}" for i in range(len(errs) - 1))
        print(f"  {name:9s}: {line}   ratios: {ratios}  (2=1st, 4=2nd)")


if __name__ == "__main__":
    main()
