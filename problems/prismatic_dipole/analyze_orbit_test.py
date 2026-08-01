#!/usr/bin/env python3
"""Analysis for ptc_orbit_test: production pusher vs fine-dt reference.

Loads the orbit_*.csv dumps of one or more test legs (same seeding),
integrates a reference relativistic-Boris orbit for the SAME initial
conditions in the EXACT dipole field at dt/dt_ref_div, and reports:
  - guiding-center position error vs time (per leg),
  - mu conservation (Boris legs: p_perp^2/2B at exact B; GCA legs: the
    mu slot),
  - azimuthal drift rate and bounce frequency vs reference,
  - loss counts, gamma drift.
Saves a metrics .npz next to each leg dir.

Usage: analyze_orbit_test.py Data_orbit_L4_boris_rec [more legs...]
       [--bp 100] [--qm -1] [--ref-div 32]
"""
import argparse
import glob
import os
import sys

import numpy as np

GCA_FLAG = 1 << 12


def dipole_B(x, Bp):
    r2 = np.sum(x * x, axis=-1)
    r = np.sqrt(r2)
    ir3 = 1.0 / (r2 * r)
    mdr = x[..., 2] / r
    B = np.empty_like(x)
    B[..., 0] = Bp * ir3 * 3 * mdr * x[..., 0] / r
    B[..., 1] = Bp * ir3 * 3 * mdr * x[..., 1] / r
    B[..., 2] = Bp * ir3 * (3 * mdr * x[..., 2] / r - 1.0)
    return B


def load_dump(fname):
    with open(fname) as f:
        t = float(f.readline().split("=")[1])
    d = np.genfromtxt(fname, delimiter=",", names=True, skip_header=1)
    d = np.atleast_1d(d)
    order = np.argsort(d["id"])
    return t, d[order]


def load_leg(dirname):
    files = sorted(glob.glob(os.path.join(dirname, "orbit_*.csv")))
    ts, xs, ps, Es, flags = [], [], [], [], []
    for f in files:
        t, d = load_dump(f)
        ts.append(t)
        xs.append(np.stack([d["x"], d["y"], d["z"]], axis=1))
        ps.append(np.stack([d["p1"], d["p2"], d["p3"]], axis=1))
        Es.append(d["E"])
        flags.append(d["flag"].astype(np.uint32))
    return (np.array(ts), np.array(xs), np.array(ps), np.array(Es),
            np.array(flags))


def reference_orbit(x0, u0, Bp, qm, t_dumps, dt_ref):
    """Relativistic Boris (E=0) at dt_ref; returns states at t_dumps."""
    n_dump = len(t_dumps)
    x = x0.copy()
    u = u0.copy()
    out_x = np.empty((n_dump,) + x0.shape)
    out_u = np.empty((n_dump,) + u0.shape)
    t = 0.0
    i_dump = 0
    # dump t=0
    while i_dump < n_dump and abs(t_dumps[i_dump] - t) < 0.5 * dt_ref:
        out_x[i_dump] = x
        out_u[i_dump] = u
        i_dump += 1
    n_steps = int(round(t_dumps[-1] / dt_ref))
    for s in range(n_steps):
        gam = np.sqrt(1 + np.sum(u * u, axis=1))
        B = dipole_B(x, Bp)
        tvec = (qm * 0.5 * dt_ref / gam)[:, None] * B
        t2 = np.sum(tvec * tvec, axis=1)
        svec = 2 * tvec / (1 + t2)[:, None]
        up = u + np.cross(u, tvec)
        u = u + np.cross(up, svec)
        gam = np.sqrt(1 + np.sum(u * u, axis=1))
        x = x + (dt_ref / gam)[:, None] * u
        t = (s + 1) * dt_ref
        while i_dump < n_dump and t_dumps[i_dump] <= t + 0.5 * dt_ref:
            out_x[i_dump] = x
            out_u[i_dump] = u
            i_dump += 1
    return out_x, out_u


def gc_of(x, u, Bp, qm):
    """Relativistic guiding center x_gc = x - (u x bhat)/(qm |B|)."""
    B = dipole_B(x, Bp)
    Bn = np.linalg.norm(B, axis=-1, keepdims=True)
    bhat = B / Bn
    return x - np.cross(u, bhat) / (qm * Bn)


def mu_of(x, u, Bp):
    B = dipole_B(x, Bp)
    Bn = np.linalg.norm(B, axis=-1)
    upar = np.sum(u * B, axis=-1) / Bn
    up2 = np.sum(u * u, axis=-1) - upar**2
    return 0.5 * up2 / Bn


def drift_rate(ts, x_gc, half=True):
    """Median azimuthal drift rate from unwrapped phi_gc (last half)."""
    phi = np.unwrap(np.arctan2(x_gc[..., 1], x_gc[..., 0]), axis=0)
    n0 = len(ts) // 2 if half else 0
    rates = []
    for j in range(phi.shape[1]):
        if np.all(np.isfinite(phi[n0:, j])):
            rates.append(np.polyfit(ts[n0:], phi[n0:, j], 1)[0])
    return np.array(rates)


def bounce_freq(ts, x_gc):
    """Dominant |z| oscillation frequency per particle (FFT peak)."""
    z = x_gc[..., 2]
    dtd = ts[1] - ts[0]
    out = []
    for j in range(z.shape[1]):
        s = z[:, j]
        if not np.all(np.isfinite(s)):
            out.append(np.nan)
            continue
        s = s - s.mean()
        f = np.fft.rfftfreq(len(s), dtd)
        a = np.abs(np.fft.rfft(s))
        out.append(f[1 + np.argmax(a[1:])])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("legs", nargs="+")
    ap.add_argument("--bp", type=float, default=100.0)
    ap.add_argument("--qm", type=float, default=-1.0)
    ap.add_argument("--ref-div", type=int, default=32)
    args = ap.parse_args()

    # Reference from the first leg's t=0 dump (all legs share seeding).
    ts, xs, ps, Es, flags = load_leg(args.legs[0])
    dt_dump = ts[1] - ts[0]
    x0, u0 = xs[0], ps[0]
    # infer sim dt from config-free data: dt_ref = dump_dt/ref steps
    dt_ref = dt_dump / (args.ref_div * 4)
    print(f"reference: {len(ts)} dumps to t={ts[-1]}, dt_ref={dt_ref:.2e}")
    rx, ru = reference_orbit(x0, u0, args.bp, args.qm, ts, dt_ref)
    r_gc = gc_of(rx, ru, args.bp, args.qm)
    r_mu = mu_of(rx, ru, args.bp)
    r_drift = drift_rate(ts, r_gc)
    r_bounce = bounce_freq(ts, r_gc)

    shell_scale = np.linalg.norm(x0, axis=1).mean()
    print(f"\n{'leg':26s} {'lost':>4} {'dgamma':>9} {'dmu/mu':>9} "
          f"{'gc_err_fin':>10} {'gc_err_max':>10} {'drift_err':>9} {'bounce_err':>10}")

    for leg in args.legs:
        ts_l, xs_l, ps_l, Es_l, flags_l = load_leg(leg)
        assert np.allclose(ts_l, ts), f"{leg}: dump times differ"
        is_gca = (flags_l & GCA_FLAG) != 0
        alive = np.isfinite(xs_l[..., 0])
        lost = (~alive[-1]).sum()

        # gc trajectory of the leg: raw position for GCA states, computed
        # gc for Cartesian states.
        gc_l = np.where(is_gca[..., None], xs_l,
                        gc_of(xs_l, ps_l, args.bp, args.qm))
        # mu: slot p2 for GCA states, kinematic mu for Cartesian.
        mu_l = np.where(is_gca, ps_l[..., 1], mu_of(xs_l, ps_l, args.bp))

        ok = alive[-1]
        gc_err = np.linalg.norm(gc_l - r_gc, axis=-1) / shell_scale
        dmu = np.abs(mu_l[-1] - r_mu[0]) / r_mu[0]
        dgam = np.nanmax(np.abs(Es_l[-1, ok] - Es_l[0, ok]) / Es_l[0, ok]) \
            if ok.any() else np.nan
        drift_l = drift_rate(ts, gc_l)
        n = min(len(drift_l), len(r_drift))
        drift_err = np.nanmedian(np.abs(drift_l[:n] - r_drift[:n]) /
                                 np.abs(r_drift[:n]))
        bounce_l = bounce_freq(ts, gc_l)
        bounce_err = np.nanmedian(np.abs(bounce_l - r_bounce) / r_bounce)

        print(f"{os.path.basename(leg):26s} {lost:4d} "
              f"{dgam:9.2e} {np.nanmedian(dmu[ok]):9.2e} "
              f"{np.nanmedian(gc_err[-1, ok]):10.2e} "
              f"{np.nanmedian(np.nanmax(gc_err[:, ok], axis=0)):10.2e} "
              f"{drift_err:9.2e} {bounce_err:10.2e}")

        np.savez(leg.rstrip('/') + "_metrics.npz",
                 ts=ts, gc_err=gc_err, mu=mu_l, mu_ref0=r_mu[0],
                 alive=alive, is_gca=is_gca, drift=drift_l,
                 drift_ref=r_drift, bounce=bounce_l, bounce_ref=r_bounce,
                 gamma=Es_l)


if __name__ == "__main__":
    main()
