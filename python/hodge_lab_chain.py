"""Lean full-chain Hodge probe: subsampled edges, lazy fits, vectorized
quadrature.  Same statistics as hodge_lab2 at ~40x less work.

Chains compared on the exact static dipole (curl-free):
  (diag,diag) : h1i * d1t * (h2*B)              -- baseline
  (W2,  diag) : h1i * d1t * circ_fit(B)         -- B-side corrected
  (W2,  W1  ) : pair_fit( d1t * circ_fit(B) )   -- fully corrected
plus probe2b: pairing exactness on a linear constant-curl field.
"""
import math
import sys

import h5py
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, "/home/alex/Projects/Aperture4/python")
import prismatic_recovery as pr
import hodge_lab

N_SAMPLE = 1500
R_IN, R_OUT = 2.5, 4.0
GQ = np.array([0.1127016653792583, 0.5, 0.8872983346207417])
GW = np.array([5.0 / 18, 8.0 / 18, 5.0 / 18])


def log(*a):
    print(*a)
    sys.stdout.flush()


class Lab:
    def __init__(self, L, NR):
        d = hodge_lab.ensure_mesh(L, NR)
        self.mesh = mesh = pr.Mesh(f"{d}/mesh.h5")
        with h5py.File(f"{d}/mesh.h5") as f:
            self.el = f["edge_radial_layer"][()]
            self.d1t = csr_matrix((f["d1t_val"][()].astype(np.float64),
                                   f["d1t_col_idx"][()], f["d1t_row_ptr"][()]),
                                  shape=(mesh.N_edges, mesh.N_faces))
            self.h2 = f["hodge2"][()].astype(np.float64)
            self.h1i = f["hodge1_inv"][()].astype(np.float64)
            self.ev0 = f["edge_v0"][()]; self.ev1 = f["edge_v1"][()]
            self.vx = f["vert_x"][()]; self.vy = f["vert_y"][()]
            self.vz = f["vert_z"][()]
        with h5py.File(f"{d}/step_000000.h5") as f0:
            self.B = f0["B_f"][()].astype(np.float64)

        v = mesh.sphere_v[mesh.tri_verts]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        flip = np.einsum("tx,tx->t", n, v.mean(axis=1)) < 0
        n[flip] *= -1
        self.circ_dir = n
        self.r_mid = 0.5 * (mesh.radii[:-1] + mesh.radii[1:])
        self.n_h = (mesh.N_r + 1) * mesh.N_edge_s
        self.n_tf = (mesh.N_r + 1) * mesh.N_tri

        self.edge_tris = -np.ones((mesh.N_edge_s, 2), dtype=int)
        for t in range(mesh.N_tri):
            for e in mesh.tri_edges_s[t]:
                if self.edge_tris[e, 0] < 0:
                    self.edge_tris[e, 0] = t
                else:
                    self.edge_tris[e, 1] = t
        # ordered fan (for v-edge dual polygons)
        vtris = [[] for _ in range(mesh.N_vert_s)]
        for t in range(mesh.N_tri):
            for s in mesh.tri_verts[t]:
                vtris[s].append(t)
        self.vert_fan = []
        for s in range(mesh.N_vert_s):
            u = mesh.sphere_v[s]
            a = (np.array([0, 0, 1.0]) if abs(u[2]) < 0.9
                 else np.array([1.0, 0, 0]))
            t1 = np.cross(a, u); t1 /= np.linalg.norm(t1)
            t2 = np.cross(u, t1)
            ang = sorted((math.atan2(float((n[t] - u) @ t2),
                                     float((n[t] - u) @ t1)), t)
                         for t in vtris[s])
            self.vert_fan.append([t for _, t in ang])

        rng = np.random.default_rng(2026)
        re = mesh.radii[self.el]
        pool = np.where((re >= R_IN) & (re < R_OUT))[0]
        self.probe_edges = rng.choice(pool, size=min(N_SAMPLE, len(pool)),
                                      replace=False)
        self._bfit, self._pfit = {}, {}

    # ---------- vectorized curved-patch moments ----------
    @staticmethod
    def _ruled_moments(u0, u1, ra, rb, x_ref):
        """Vector area + moment of the ruled patch r in [ra,rb] along the
        normalized-lerp arc u0->u1 (works for primal rect faces and
        h-edge duals alike)."""
        A2, B2 = np.meshgrid(GQ, GQ, indexing="ij")
        W2 = np.outer(GW, GW).ravel()
        ua, zb = A2.ravel(), B2.ravel()
        P = u0[None, :] * (1 - ua)[:, None] + u1[None, :] * ua[:, None]
        Pn = np.linalg.norm(P, axis=1, keepdims=True)
        Nh = P / Pn
        dNu = ((u1 - u0)[None, :] -
               Nh * (Nh @ (u1 - u0))[:, None]) / Pn
        r = ra + (rb - ra) * zb
        X = r[:, None] * Nh
        dA = np.cross(r[:, None] * dNu, (rb - ra) * Nh)
        A = np.einsum("qx,q->x", dA, W2)
        M = np.einsum("qx,qj,q->xj", dA, X - x_ref[None, :], W2)
        return A, M

    @staticmethod
    def _sphtri_moments(u0, u1, u2, r, x_ref):
        """Vector area + moment of the spherical triangle (u0,u1,u2) at
        radius r (Duffy quadrature, vectorized)."""
        A2, B2 = np.meshgrid(GQ, GQ, indexing="ij")
        l1 = A2.ravel()
        l2 = (B2 * (1 - A2)).ravel()
        W2 = (np.outer(GW, GW) * (1 - A2)).ravel()
        l0 = 1 - l1 - l2
        P = (l0[:, None] * u0[None, :] + l1[:, None] * u1[None, :]
             + l2[:, None] * u2[None, :])
        Pn = np.linalg.norm(P, axis=1, keepdims=True)
        Nh = P / Pn
        d1P, d2P = u1 - u0, u2 - u0
        dN1 = (d1P[None, :] - Nh * (Nh @ d1P)[:, None]) / Pn
        dN2 = (d2P[None, :] - Nh * (Nh @ d2P)[:, None]) / Pn
        dA = np.cross(r * dN1, r * dN2)
        X = r * Nh
        A = np.einsum("qx,q->x", dA, W2)
        M = np.einsum("qx,qj,q->xj", dA, X - x_ref[None, :], W2)
        return A, M

    # ---------- B-side fit (fluxes -> linear field), lazy ----------
    def bfit(self, kv, s):
        key = (kv, s)
        if key in self._bfit:
            return self._bfit[key]
        mesh = self.mesh
        x_v = mesh.radii[kv] * mesh.sphere_v[s]
        faces = []
        for ks in (kv - 1, kv, kv + 1):
            if 0 <= ks <= mesh.N_r:
                for t in mesh.vert_tris[s]:
                    faces.append(("tri", ks, t))
        for kl in (kv - 1, kv):
            if 0 <= kl < mesh.N_r:
                for e in mesh.vert_sedges[s]:
                    faces.append(("rect", kl, e))
        rows, scales, gidx = [], [], []
        for kind, k, idx in faces:
            if kind == "tri":
                tv = mesh.tri_verts[idx]
                A, M = self._sphtri_moments(mesh.sphere_v[tv[0]],
                                            mesh.sphere_v[tv[1]],
                                            mesh.sphere_v[tv[2]],
                                            mesh.radii[k], x_v)
                gidx.append(k * mesh.N_tri + idx)
            else:
                s0, s1 = mesh.sphere_edges[idx]
                A, M = self._ruled_moments(mesh.sphere_v[s0],
                                           mesh.sphere_v[s1],
                                           mesh.radii[k], mesh.radii[k + 1],
                                           x_v)
                gidx.append(self.n_tf + k * mesh.N_edge_s + idx)
            an = np.linalg.norm(A)
            rows.append(np.concatenate([A, M.reshape(9)]) / an)
            scales.append(1.0 / an)
        R = np.asarray(rows)
        Rc = R @ pr._N_TRACELESS
        W = (pr._N_TRACELESS @ np.linalg.solve(Rc.T @ Rc, Rc.T)
             ) * np.asarray(scales)[None, :]
        self._bfit[key] = (np.array(gidx), W)
        return self._bfit[key]

    # ---------- pairing fit (dual fluxes -> linear field), lazy ----------
    def dual_moments(self, ei, x_ref):
        mesh = self.mesh
        if ei < self.n_h:
            k = ei // mesh.N_edge_s
            e = ei - k * mesh.N_edge_s
            t0, t1 = self.edge_tris[e]
            A, M = self._ruled_moments(self.circ_dir[t0], self.circ_dir[t1],
                                       self.r_mid[k - 1], self.r_mid[k],
                                       x_ref)
            s0, s1 = mesh.sphere_edges[e]
            tvec = mesh.sphere_v[s1] - mesh.sphere_v[s0]
        else:
            li = ei - self.n_h
            k = li // mesh.N_vert_s
            s = li - k * mesh.N_vert_s
            fan = self.vert_fan[s]
            u = mesh.sphere_v[s]
            A = np.zeros(3); M = np.zeros((3, 3))
            for i in range(len(fan)):
                c0 = self.circ_dir[fan[i]]
                c1 = self.circ_dir[fan[(i + 1) % len(fan)]]
                Ai, Mi = self._sphtri_moments(u, c0, c1, self.r_mid[k], x_ref)
                A += Ai; M += Mi
            tvec = u
        if float(A @ tvec) < 0:
            A, M = -A, -M
        return A, M

    def pfit(self, kv, s):
        key = (kv, s)
        if key in self._pfit:
            return self._pfit[key]
        mesh = self.mesh
        x_v = mesh.radii[kv] * mesh.sphere_v[s]
        eids = set()
        for ks in (kv - 1, kv, kv + 1):
            if 1 <= ks <= mesh.N_r - 1:
                for e in mesh.vert_sedges[s]:
                    eids.add(ks * mesh.N_edge_s + e)
        for kl in (kv - 1, kv):
            if 0 <= kl < mesh.N_r:
                eids.add(self.n_h + kl * mesh.N_vert_s + s)
                for e in mesh.vert_sedges[s]:
                    s2 = int(mesh.sphere_edges[e][0]
                             + mesh.sphere_edges[e][1] - s)
                    eids.add(self.n_h + kl * mesh.N_vert_s + s2)
        eids = sorted(eids)
        rows, scales = [], []
        for ei in eids:
            A, M = self.dual_moments(ei, x_v)
            an = np.linalg.norm(A)
            rows.append(np.concatenate([A, M.reshape(9)]) / an)
            scales.append(1.0 / an)
        R = np.asarray(rows)
        Rc = R @ pr._N_TRACELESS
        W = (pr._N_TRACELESS @ np.linalg.solve(Rc.T @ Rc, Rc.T)
             ) * np.asarray(scales)[None, :]
        self._pfit[key] = (np.array(eids), W)
        return self._pfit[key]

    # ---------- corrected per-face dual-segment circulation ----------
    def circ_face(self, f, field):
        mesh = self.mesh
        if f < self.n_tf:
            k = f // mesh.N_tri
            t = f - k * mesh.N_tri
            if k == 0 or k == mesh.N_r:
                return self.h2[f] * field[f]
            u = self.circ_dir[t]
            p0, p1 = self.r_mid[k - 1] * u, self.r_mid[k] * u
            X = p0[None, :] + (p1 - p0)[None, :] * GQ[:, None]
            dX = np.repeat((p1 - p0)[None, :], len(GQ), axis=0)
            wq = GW
            svert, kv, sgn = mesh.tri_verts[t][0], k, 1.0
        else:
            fi = f - self.n_tf
            k = fi // mesh.N_edge_s
            e = fi - k * mesh.N_edge_s
            t0, t1 = self.edge_tris[e]
            u0, u1 = self.circ_dir[t0], self.circ_dir[t1]
            P = u0[None, :] * (1 - GQ)[:, None] + u1[None, :] * GQ[:, None]
            Pn = np.linalg.norm(P, axis=1, keepdims=True)
            X = self.r_mid[k] * P / Pn
            dX = self.r_mid[k] * ((u1 - u0)[None, :]
                                  - (P / Pn) * ((P / Pn) @ (u1 - u0))[:, None]
                                  ) / Pn
            wq = GW
            sa = int(mesh.rect_face_v0[fi]) % mesh.N_vert_s
            sb = int(mesh.rect_face_v1[fi]) % mesh.N_vert_s
            ua, ub = mesh.sphere_v[sa], mesh.sphere_v[sb]
            um = ua + ub; um /= np.linalg.norm(um)
            nf = np.cross(ub - ua, um)
            sgn = 1.0 if float((X[-1] - X[0]) @ nf) >= 0 else -1.0
            svert, kv = sa, min(max(k, 1), mesh.N_r - 1)
        gidx, W = self.bfit(kv, svert)
        x_v = mesh.radii[kv] * mesh.sphere_v[svert]
        chord = np.einsum("qx,q->x", dX, wq)
        K = np.einsum("qx,qj,q->xj", dX, X - x_v[None, :], wq)
        coef = np.concatenate([chord, K.reshape(9)]) @ W
        return sgn * float(coef @ field[gidx])

    # ---------- corrected pairing: edge circulation from dual fluxes ----
    def pair_edge(self, ei, Phi):
        mesh = self.mesh
        if ei < self.n_h:
            k = ei // mesh.N_edge_s
            e = ei - k * mesh.N_edge_s
            svert = int(mesh.sphere_edges[e][0])
            kv = min(max(k, 1), mesh.N_r - 1)
            s0, s1 = mesh.sphere_edges[e]
            u0, u1 = mesh.sphere_v[s0], mesh.sphere_v[s1]
            r = mesh.radii[k]
            P = u0[None, :] * (1 - GQ)[:, None] + u1[None, :] * GQ[:, None]
            Pn = np.linalg.norm(P, axis=1, keepdims=True)
            X = r * P / Pn
            dX = r * ((u1 - u0)[None, :]
                      - (P / Pn) * ((P / Pn) @ (u1 - u0))[:, None]) / Pn
        else:
            li = ei - self.n_h
            k = li // mesh.N_vert_s
            svert = li - k * mesh.N_vert_s
            kv = min(max(k, 1), mesh.N_r - 1)
            u = mesh.sphere_v[svert]
            ra, rb = mesh.radii[k], mesh.radii[k + 1]
            rq = ra + (rb - ra) * GQ
            X = rq[:, None] * u[None, :]
            dX = np.repeat((rb - ra) * u[None, :], len(GQ), axis=0)
        eids, W = self.pfit(kv, svert)
        x_v = mesh.radii[kv] * mesh.sphere_v[svert]
        chord = np.einsum("qx,q->x", dX, GW)
        K = np.einsum("qx,qj,q->xj", dX, X - x_v[None, :], GW)
        coef = np.concatenate([chord, K.reshape(9)]) @ W
        return float(coef @ Phi[eids]), coef, eids, chord


def run_level(L, NR):
    lab = Lab(L, NR)
    mesh, d1t = lab.mesh, lab.d1t
    B = lab.B
    pe = lab.probe_edges

    # faces needed: those adjacent to probe edges (for corrected circs we
    # need circ on ALL faces feeding the d1t rows of probe edges) AND all
    # faces feeding the W1 patches' dual fluxes... For the (W2,W1) chain,
    # W1 reads dual fluxes at patch edges, each of which needs its d1t row
    # of corrected circs.  Collect the closure.
    ind, idx = d1t.indptr, d1t.indices
    need_edges = set(pe.tolist())
    for ei in pe:
        _, eids = None, None
        # peek the patch without computing
        if ei < lab.n_h:
            k = ei // mesh.N_edge_s
            e = ei - k * mesh.N_edge_s
            svert = int(mesh.sphere_edges[e][0])
            kv = min(max(k, 1), mesh.N_r - 1)
        else:
            li = ei - lab.n_h
            k = li // mesh.N_vert_s
            svert = li - k * mesh.N_vert_s
            kv = min(max(k, 1), mesh.N_r - 1)
        eids2, _ = lab.pfit(kv, svert)
        need_edges.update(eids2.tolist())
    need_edges = np.array(sorted(need_edges))
    faces_needed = np.unique(np.concatenate(
        [idx[ind[e]:ind[e + 1]] for e in need_edges]))

    circ_diag = lab.h2 * B
    circ = circ_diag.copy()
    for f in faces_needed:
        circ[f] = lab.circ_face(f, B)

    Phi = d1t @ circ
    Phi_diag = d1t @ circ_diag

    # chains on probe edges
    absref_d = lab.h1i * (np.abs(d1t) @ np.abs(circ_diag))
    e_dd = (np.linalg.norm((lab.h1i * Phi_diag)[pe])
            / np.linalg.norm(absref_d[pe]))
    e_cd = (np.linalg.norm((lab.h1i * Phi)[pe])
            / np.linalg.norm(absref_d[pe]))
    # field-scale reference (same as the diagonal chains) so all three
    # chain errors are directly comparable
    num = 0.0
    for ei in pe:
        val, coef, eids, _ = lab.pair_edge(ei, Phi)
        num += val * val
    e_cc = math.sqrt(num) / np.linalg.norm(absref_d[pe])

    # mini probe2b: pairing exactness on a linear constant-curl field
    curl = np.array([0.0, 0.0, 1.0])
    # needed dual fluxes computed lazily during evaluation
    cache = {}
    def phi_lin(ei):
        if ei not in cache:
            A, _ = lab.dual_moments(ei, np.zeros(3))
            cache[ei] = float(curl @ A)
        return cache[ei]
    err2b_n, err2b_d = 0.0, 0.0
    for ei in pe[:50]:
        _, coef, eids, chord = lab.pair_edge(ei, Phi)
        val = float(coef @ np.array([phi_lin(int(x)) for x in eids]))
        exact = float(chord @ curl)
        err2b_n += (val - exact) ** 2
        err2b_d += exact ** 2
    log(f"L{L}: probe2b (pairing on linear field) rel = "
        f"{math.sqrt(err2b_n / err2b_d):.3e}")
    log(f"L{L}: (diag,diag)={e_dd:.3e}  (W2,diag)={e_cd:.3e}  "
        f"(W2,W1)={e_cc:.3e}")
    return e_dd, e_cd, e_cc


if __name__ == "__main__":
    out = {}
    for L, NR in [(4, 51), (5, 102)]:
        out[L] = run_level(L, NR)
    log("order ratios L4->L5: " + "  ".join(
        f"{n}={out[4][i]/out[5][i]:.2f}"
        for i, n in enumerate(["(diag,diag)", "(W2,diag)", "(W2,W1)"])))
