#!/usr/bin/env python3
"""
Compare numerical E_aux (and the stored D̃) from the IC against the
analytic stationary Kerr-Schild Wald prediction.

E_aux[e] = ∫_e E_i dx^i  and for stationary Wald E_i = -∂_i A_0,
so the analytic value is E_aux_analytic[e] = A_0(v0) - A_0(v1).

Usage: python3 check_ic.py [--data Data_L5_check] [--a 0.998] [--Bp 1.0]
"""
import argparse
import numpy as np
import h5py


def wald_ks_A0(a, r, sth, cth):
    rho2 = r * r + a * a * cth * cth
    return a * r * (1.0 + cth * cth) / rho2 - a


def cart_to_sph(x, y, z):
    r = np.sqrt(x * x + y * y + z * z)
    cth = np.where(r > 0, z / r, 1.0)
    s2 = 1.0 - cth * cth
    sth = np.sqrt(np.maximum(s2, 0.0))
    return r, sth, cth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data_L5_check")
    ap.add_argument("--a", type=float, default=0.998)
    ap.add_argument("--Bp", type=float, default=1.0)
    args = ap.parse_args()

    with h5py.File(f"{args.data}/mesh.h5", "r") as m:
        vx = np.array(m["vert_x"])
        vy = np.array(m["vert_y"])
        vz = np.array(m["vert_z"])
        edge_v0 = np.array(m["edge_v0"])
        edge_v1 = np.array(m["edge_v1"])
        edge_layer = np.array(m["edge_radial_layer"])
        radii = np.array(m["radii"])
        N_edges = edge_v0.size
        N_edge_s = int(m["N_edge_s"][()])
        N_vert_s = int(m["N_vert_s"][()])
        N_r = int(m["N_r"][()])

    with h5py.File(f"{args.data}/ic_aux.h5", "r") as f:
        E_aux_num = np.array(f["E_aux"])
        H_aux_num = np.array(f["H_aux"])
        D_num = np.array(f["D"])
        B_num = np.array(f["B"])

    # Analytic E_aux[e] = Bp · [A_0(v0) - A_0(v1)]
    r0, sth0, cth0 = cart_to_sph(vx[edge_v0], vy[edge_v0], vz[edge_v0])
    r1, sth1, cth1 = cart_to_sph(vx[edge_v1], vy[edge_v1], vz[edge_v1])
    A0_v0 = wald_ks_A0(args.a, r0, sth0, cth0)
    A0_v1 = wald_ks_A0(args.a, r1, sth1, cth1)
    # Note sign: E_i convention in wald_solution.hpp has D^i ∝ +∂_i A_0/α
    # (covariant E_i = +∂_i A_0), so the line integral equals A_0(v1) - A_0(v0).
    E_aux_ana = args.Bp * (A0_v1 - A0_v0)

    print(f"N_edges = {N_edges}  (N_h = {(N_r+1)*N_edge_s}, N_v = {N_r*N_vert_s})")
    print()

    # Global stats
    def stats(lbl, num, ana):
        resid = num - ana
        inf_num = np.max(np.abs(num))
        inf_ana = np.max(np.abs(ana))
        inf_res = np.max(np.abs(resid))
        rms_num = np.sqrt(np.mean(num * num))
        rms_ana = np.sqrt(np.mean(ana * ana))
        rms_res = np.sqrt(np.mean(resid * resid))
        print(f"  {lbl}:")
        print(f"    max |num|={inf_num:.4e}  max |ana|={inf_ana:.4e}  max |resid|={inf_res:.4e}  "
              f"(rel inf = {inf_res/(inf_ana+1e-30):.2e})")
        print(f"    rms |num|={rms_num:.4e}  rms |ana|={rms_ana:.4e}  rms |resid|={rms_res:.4e}  "
              f"(rel rms = {rms_res/(rms_ana+1e-30):.2e})")

    print("Whole mesh E_aux vs A0 jump:")
    stats("E_aux[all edges]", E_aux_num, E_aux_ana)

    # Split by horizontal vs vertical edge.  Horizontal edges have idx in
    # [0, (N_r+1)*N_edge_s).  Vertical edges are the rest.
    is_horiz = np.arange(N_edges) < (N_r + 1) * N_edge_s
    print()
    stats("E_aux[horizontal]", E_aux_num[is_horiz], E_aux_ana[is_horiz])
    stats("E_aux[vertical  ]", E_aux_num[~is_horiz], E_aux_ana[~is_horiz])

    # Per radial shell: show a few shells
    print()
    print("E_aux by shell (k, r, max|num|, max|ana|, max|resid|, rel_inf):")
    for k in [0, 5, 10, 20, 40, 60, 80, N_r]:
        mask = edge_layer == k
        if not np.any(mask):
            continue
        num = E_aux_num[mask]
        ana = E_aux_ana[mask]
        resid = num - ana
        print(f"  k={k:3d}  r={radii[k]:6.3f}  "
              f"|num|={np.max(np.abs(num)):.3e}  |ana|={np.max(np.abs(ana)):.3e}  "
              f"|res|={np.max(np.abs(resid)):.3e}  "
              f"rel={np.max(np.abs(resid))/(np.max(np.abs(ana))+1e-30):.2e}")


    # --- d1 · E_aux: the "curl" of E_aux on primal faces -------------------
    # If E_aux were exactly the gradient of some discrete A_0 (i.e. a pure
    # jump A_0(v1)-A_0(v0) per edge), then d1·E_aux would be identically 0
    # by d∘d = 0.  A nonzero d1·E_aux at a stationary IC means the scheme
    # cannot represent the solution as a pure gradient, regardless of gauge.
    print()
    print("d1·E_aux (discrete curl of E_aux) at IC:")
    with h5py.File(f"{args.data}/mesh.h5", "r") as m:
        d1_row_ptr = np.array(m["d1_row_ptr"])
        d1_col_idx = np.array(m["d1_col_idx"])
        d1_val = np.array(m["d1_val"]).astype(np.float64)
        face_layer = np.array(m["face_radial_layer"])
        N_faces = face_layer.size
        N_tri = int(m["N_tri"][()])

    curl = np.zeros(N_faces, dtype=np.float64)
    for f in range(N_faces):
        s = d1_row_ptr[f]; e = d1_row_ptr[f + 1]
        curl[f] = np.dot(d1_val[s:e], E_aux_num[d1_col_idx[s:e]])

    print(f"  max |d1·E_aux| = {np.max(np.abs(curl)):.4e}")
    print(f"  rms |d1·E_aux| = {np.sqrt(np.mean(curl*curl)):.4e}")
    # Compare to typical B magnitudes
    print(f"  max |B|         = {np.max(np.abs(B_num)):.4e}")
    # Per shell
    print("  d1·E_aux by layer:")
    for k in [0, 5, 10, 20, 40, 60, 80, N_r]:
        mask = face_layer == k
        if not np.any(mask): continue
        print(f"    k={k:3d}  max|curl|={np.max(np.abs(curl[mask])):.3e}  "
              f"max|B|={np.max(np.abs(B_num[mask])):.3e}")


if __name__ == "__main__":
    main()
