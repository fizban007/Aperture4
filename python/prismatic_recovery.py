"""Prototype library for the prismatic-mesh particle B-gather study (roadmap A1).

Loads a mesh.h5 written by prismatic_data_exporter and provides, in pure
numpy:

  - analytic-field face-flux cochains by Gauss quadrature (matching the
    mesh's curved-face parametrization and orientation conventions);
  - the primal Whitney gather (port of interpolate_fields in
    prismatic_deposit.h / python/prismatic_interp.cpp);
  - the vertex least-squares recovery gather: per-vertex linear-reproducing
    LSQ fit of adjacent face fluxes, hat-interpolated to points.

Used by the A1 validation studies (convergence, continuity, conditioning,
particle scattering) before the production CUDA implementation.
"""

import numpy as np
import h5py


# =========================================================================
# Mesh container
# =========================================================================

class Mesh:
    """In-memory copy of mesh.h5 plus derived adjacency tables."""

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            for key in f.keys():
                val = f[key][()]
                setattr(self, key, val)
        self.L = int(self.L)
        self.N_r = int(self.N_r)
        self.N_tri = int(self.N_tri)
        self.N_vert_s = int(self.N_vert_s)
        self.N_edge_s = int(self.N_edge_s)
        self.N_verts = int(self.N_verts)
        self.N_edges = int(self.N_edges)
        self.N_faces = int(self.N_faces)

        self.sphere_v = np.stack(
            [self.sphere_vx, self.sphere_vy, self.sphere_vz], axis=1
        ).astype(np.float64)
        self.radii = self.radii.astype(np.float64)
        self.tri_verts = self.tri_verts.reshape(self.N_tri, 3)
        self.tri_edges_s = self.tri_edges_s.reshape(self.N_tri, 3)
        self.tri_edge_signs = self.tri_edge_signs.reshape(self.N_tri, 3)
        self.tri_neighbor = self.tri_neighbor.reshape(self.N_tri, 3)

        # Sphere-edge endpoints from the layer-0 horizontal edges
        # (edge (k,e) endpoints are 3D vertex ids k*N_vert_s + s).
        h0_v0 = self.edge_v0[: self.N_edge_s]
        h0_v1 = self.edge_v1[: self.N_edge_s]
        assert h0_v0.max() < self.N_vert_s and h0_v1.max() < self.N_vert_s
        self.sphere_edges = np.stack([h0_v0, h0_v1], axis=1)

        # Vertex → incident triangles / sphere edges (ragged lists).
        self.vert_tris = [[] for _ in range(self.N_vert_s)]
        for t in range(self.N_tri):
            for s in self.tri_verts[t]:
                self.vert_tris[s].append(t)
        self.vert_sedges = [[] for _ in range(self.N_vert_s)]
        for e in range(self.N_edge_s):
            for s in self.sphere_edges[e]:
                self.vert_sedges[s].append(e)
        self.valence = np.array([len(v) for v in self.vert_tris])

    # --- cochain index helpers (match prismatic_mesh.h conventions) ---
    def h_edge_idx(self, k, e):
        return k * self.N_edge_s + e

    def v_edge_idx(self, k, s):
        return (self.N_r + 1) * self.N_edge_s + k * self.N_vert_s + s

    def tri_face_idx(self, k, t):
        return k * self.N_tri + t

    def rect_face_idx(self, k, e):
        return (self.N_r + 1) * self.N_tri + k * self.N_edge_s + e

    def vert_idx(self, k, s):
        return k * self.N_vert_s + s


# =========================================================================
# Quadrature rules
# =========================================================================

def _gauss_legendre_01(n):
    """Gauss-Legendre nodes/weights on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def tri_quad(n):
    """Quadrature on the reference triangle {l1,l2>=0, l1+l2<=1} via a
    Duffy-type collapsed tensor rule.  Returns (l1, l2, w)."""
    xa, wa = _gauss_legendre_01(n)
    xb, wb = _gauss_legendre_01(n)
    A, B = np.meshgrid(xa, xb, indexing="ij")
    W = np.outer(wa, wb)
    l1 = A
    l2 = B * (1.0 - A)
    w = W * (1.0 - A)  # Duffy Jacobian
    return l1.ravel(), l2.ravel(), w.ravel()


# =========================================================================
# Analytic-field flux cochains (curved faces, mesh orientation)
# =========================================================================

def flux_cochain(mesh, B_func, n=8):
    """Face 2-cochain of an analytic field: B_f[f] = ∫_f B·dA.

    B_func(X) takes (N,3) Cartesian points, returns (N,3) field values.

    Faces are the mesh's curved elements: shell (spherical) triangles at
    radius radii[k], parametrized X = r·normalize(Σ λ_i u_i); ruled
    rectangular faces spanning [radii[k], radii[k+1]] over a sphere edge,
    X = r(ζ)·normalize((1-u) u0 + u u1).  Orientation follows the stored
    vertex order (validated against the solver's own IC in the A1 tests).
    """
    B_f = np.zeros(mesh.N_faces)

    # ---- shell triangles ----
    l1, l2, w = tri_quad(n)
    l0 = 1.0 - l1 - l2
    lam = np.stack([l0, l1, l2], axis=1)  # (nq, 3)
    u = mesh.sphere_v[mesh.tri_verts]     # (N_tri, 3verts, 3)
    # P(t, q, :) = Σ_i lam(q,i) u(t,i,:)
    P = np.einsum("qi,tix->tqx", lam, u)
    Pn = np.linalg.norm(P, axis=2, keepdims=True)
    N_hat = P / Pn
    # dX/dλ1 = r * d/dλ1 normalize(P), with dP/dλ1 = u1 - u0, dP/dλ2 = u2 - u0
    d1P = (u[:, 1, :] - u[:, 0, :])[:, None, :]
    d2P = (u[:, 2, :] - u[:, 0, :])[:, None, :]

    def dnormalize(P, Pn, N_hat, dP):
        # d/dt normalize(P) = (dP - N_hat (N_hat·dP)) / |P|
        return (dP - N_hat * np.sum(N_hat * dP, axis=2, keepdims=True)) / Pn

    dN1 = dnormalize(P, Pn, N_hat, d1P)
    dN2 = dnormalize(P, Pn, N_hat, d2P)
    for k in range(mesh.N_r + 1):
        r = mesh.radii[k]
        X = r * N_hat
        dA = np.cross(r * dN1, r * dN2)  # (N_tri, nq, 3)
        Bq = B_func(X.reshape(-1, 3)).reshape(X.shape)
        flux = np.einsum("tqx,tqx,q->t", Bq, dA, w)
        B_f[k * mesh.N_tri : (k + 1) * mesh.N_tri] = flux

    # ---- rectangular (ruled) faces ----
    xu, wu = _gauss_legendre_01(n)
    xz, wz = _gauss_legendre_01(n)
    U, Z = np.meshgrid(xu, xz, indexing="ij")
    W2 = np.outer(wu, wz).ravel()
    U = U.ravel()
    Z = Z.ravel()
    s0 = mesh.sphere_edges[:, 0]
    s1 = mesh.sphere_edges[:, 1]
    u0 = mesh.sphere_v[s0]  # (N_edge_s, 3)
    u1 = mesh.sphere_v[s1]
    P = u0[:, None, :] * (1.0 - U)[None, :, None] + u1[:, None, :] * U[None, :, None]
    Pn = np.linalg.norm(P, axis=2, keepdims=True)
    N_hat = P / Pn
    dPu = (u1 - u0)[:, None, :]
    dNu = (dPu - N_hat * np.sum(N_hat * dPu, axis=2, keepdims=True)) / Pn
    n_tri_faces = (mesh.N_r + 1) * mesh.N_tri
    for k in range(mesh.N_r):
        ra, rb = mesh.radii[k], mesh.radii[k + 1]
        r = ra + (rb - ra) * Z  # (nq,)
        X = r[None, :, None] * N_hat
        dXu = r[None, :, None] * dNu
        dXz = (rb - ra) * N_hat
        dA = np.cross(dXu, dXz)
        Bq = B_func(X.reshape(-1, 3)).reshape(X.shape)
        flux = np.einsum("eqx,eqx,q->e", Bq, dA, W2)
        B_f[n_tri_faces + k * mesh.N_edge_s : n_tri_faces + (k + 1) * mesh.N_edge_s] = flux

    return B_f


# =========================================================================
# Point location
# =========================================================================

def barycentric(mesh, t, p_hat):
    """Central (gnomonic) projection barycentric coords of unit vector
    p_hat in triangle t: solve p_hat ∝ λ0 v0 + λ1 v1 + λ2 v2, Σλ = 1.

    Unlike the perpendicular-projection formula in the production C++
    (prismatic_mesh_ptrs.h compute_barycentric), this predicate tiles the
    sphere EXACTLY — radial rays partition the convex icosphere surface —
    so point location has no orphan slivers and the walk terminates.
    The two definitions agree to O(1e-3) in λ; production should adopt
    this one (see roadmap A1 notes)."""
    if not hasattr(mesh, "_tri_vinv"):
        V = mesh.sphere_v[mesh.tri_verts]           # (T, 3verts, 3xyz)
        mesh._tri_vinv = np.linalg.inv(V.transpose(0, 2, 1))  # (T, 3, 3)
    lam_raw = mesh._tri_vinv[t] @ p_hat
    s = lam_raw.sum()
    # s < 0 means the ray hits the ANTIPODAL triangle; dividing by s would
    # flip all-negative coords to all-positive and fool the walk.  Divide
    # by |s| so far-side triangles keep negative coordinates.
    return lam_raw / abs(s)


def locate(mesh, pts, hints=None):
    """Locate Cartesian points: returns (tri, layer, lam(3), zeta) arrays.
    Points outside the radial range get layer = -1.

    hints: optional int array of per-point starting triangles; updated
    in place with the found triangles (pass the previous step's array in
    a particle loop to make walks O(1))."""
    pts = np.asarray(pts, dtype=np.float64)
    N = len(pts)
    tri = np.zeros(N, dtype=int)
    layer = np.zeros(N, dtype=int)
    lam = np.zeros((N, 3))
    zeta = np.zeros(N)
    opp = (1, 2, 0)
    hint = 0
    for i in range(N):
        if hints is not None:
            hint = int(hints[i])
        r = np.linalg.norm(pts[i])
        if r < mesh.radii[0] or r > mesh.radii[-1]:
            layer[i] = -1
            continue
        k = min(np.searchsorted(mesh.radii, r, side="right") - 1, mesh.N_r - 1)
        layer[i] = k
        zeta[i] = (r - mesh.radii[k]) / (mesh.radii[k + 1] - mesh.radii[k])
        p_hat = pts[i] / r
        t = hint
        found = False
        for _ in range(mesh.N_tri):
            lm = barycentric(mesh, t, p_hat)
            if lm.min() >= -1e-12:
                found = True
                break
            nxt = mesh.tri_neighbor[t, opp[int(np.argmin(lm))]]
            if nxt < 0:
                break
            t = nxt
        if not found:
            # Global fallback (should not trigger with the exact central-
            # projection predicate; kept for robustness).
            lam_raw = mesh._tri_vinv @ p_hat          # (T, 3)
            lam_all = lam_raw / np.abs(lam_raw.sum(axis=1, keepdims=True))
            t = int(np.argmax(lam_all.min(axis=1)))
        tri[i] = t
        hint = t
        if hints is not None:
            hints[i] = t
        lam[i] = np.clip(barycentric(mesh, t, p_hat), 0.0, None)
        lam[i] /= lam[i].sum()
    return tri, layer, lam, zeta


# =========================================================================
# Primal Whitney gather (port of interp_fields in prismatic_interp.cpp)
# =========================================================================

def primal_gather(mesh, E_e, B_f, pts, hints=None):
    """Whitney-form E and B at Cartesian points from primal cochains.
    Returns (E, B) arrays of shape (N, 3)."""
    tri, layer, lam, zeta = locate(mesh, pts, hints)
    N = len(pts)
    E = np.zeros((N, 3))
    B = np.zeros((N, 3))
    for i in range(N):
        if layer[i] < 0:
            continue
        t, k = tri[i], layer[i]
        sv = mesh.tri_verts[t]
        r_mid = 0.5 * (mesh.radii[k] + mesh.radii[k + 1])
        p = r_mid * mesh.sphere_v[sv]  # (3 verts, 3)
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = n @ n
        # gradients of planar barycentric coords
        gl = np.empty((3, 3))
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            gl[a] = np.cross(n, p[c] - p[b]) / nn
        # radial direction at the point, and d(zeta)/dx
        rh = lam[i] @ mesh.sphere_v[sv]
        rh = rh / np.linalg.norm(rh)
        dr = mesh.radii[k + 1] - mesh.radii[k]
        dz = rh / dr
        phi = (1.0 - zeta[i], zeta[i])

        edges = np.empty(9, dtype=int)
        for j in range(3):
            edges[j] = mesh.h_edge_idx(k, mesh.tri_edges_s[t, j])
            edges[3 + j] = mesh.h_edge_idx(k + 1, mesh.tri_edges_s[t, j])
            edges[6 + j] = mesh.v_edge_idx(k, sv[j])

        # E: horizontal Whitney 1-forms on both shells + vertical hats
        for j in range(3):
            fi, ti = j, (j + 1) % 3
            sign = mesh.tri_edge_signs[t, j]
            wvec = lam[i, fi] * gl[ti] - lam[i, ti] * gl[fi]
            for kk in range(2):
                E[i] += sign * E_e[edges[j + 3 * kk]] * phi[kk] * wvec
        for a in range(3):
            E[i] += E_e[edges[6 + a]] * lam[i, a] * dz

        # B: tri-face Whitney 2-forms (radial) + rect-face (tangential)
        dl12 = np.cross(gl[0], gl[1])
        for kk in range(2):
            B[i] += 2.0 * B_f[mesh.tri_face_idx(k + kk, t)] * phi[kk] * dl12
        for j in range(3):
            fi, ti = j, (j + 1) % 3
            sign = mesh.tri_edge_signs[t, j]
            se = mesh.tri_edges_s[t, j]
            wvec = lam[i, fi] * gl[ti] - lam[i, ti] * gl[fi]
            B[i] += sign * B_f[mesh.rect_face_idx(k, se)] * np.cross(wvec, dz)
    return E, B


# =========================================================================
# Vertex least-squares recovery
# =========================================================================

def _face_moments(mesh, faces_k, faces_id, x_v, n=4):
    """Vector area A_vec and first moment M[i,j] = ∫ (x-x_v)_j dA_i for a
    list of faces given as (kind, k, idx): kind 0 = shell tri (k, t),
    kind 1 = rect (layer k, sphere edge e).  Returns (nf,3), (nf,3,3)."""
    nf = len(faces_k)
    A = np.zeros((nf, 3))
    M = np.zeros((nf, 3, 3))
    l1, l2, w = tri_quad(n)
    lam = np.stack([1.0 - l1 - l2, l1, l2], axis=1)
    xu, wu = _gauss_legendre_01(n)
    xz, wz = _gauss_legendre_01(n)
    U, Z = np.meshgrid(xu, xz, indexing="ij")
    W2 = np.outer(wu, wz).ravel()
    U, Z = U.ravel(), Z.ravel()

    for i in range(nf):
        kind, k, idx = faces_k[i], faces_id[i][0], faces_id[i][1]
        if kind == 0:  # shell triangle (k=shell, idx=tri)
            u = mesh.sphere_v[mesh.tri_verts[idx]]
            P = lam @ u
            Pn = np.linalg.norm(P, axis=1, keepdims=True)
            Nh = P / Pn
            dP1, dP2 = u[1] - u[0], u[2] - u[0]
            dN1 = (dP1 - Nh * (Nh @ dP1)[:, None]) / Pn
            dN2 = (dP2 - Nh * (Nh @ dP2)[:, None]) / Pn
            r = mesh.radii[k]
            X = r * Nh
            dA = np.cross(r * dN1, r * dN2)
        else:  # rect face (k=layer, idx=sphere edge)
            s0, s1 = mesh.sphere_edges[idx]
            u0, u1 = mesh.sphere_v[s0], mesh.sphere_v[s1]
            P = u0[None, :] * (1 - U)[:, None] + u1[None, :] * U[:, None]
            Pn = np.linalg.norm(P, axis=1, keepdims=True)
            Nh = P / Pn
            dPu = u1 - u0
            dNu = (dPu - Nh * (Nh @ dPu)[:, None]) / Pn
            ra, rb = mesh.radii[k], mesh.radii[k + 1]
            r = ra + (rb - ra) * Z
            X = r[:, None] * Nh
            dXu = r[:, None] * dNu
            dXz = (rb - ra) * Nh
            dA = np.cross(dXu, dXz)
        wq = w if kind == 0 else W2
        A[i] = np.einsum("qx,q->x", dA, wq)
        M[i] = np.einsum("qi,qj,q->ij", dA, X - x_v, wq)
    return A, M


def vertex_patch(mesh, s, k):
    """Face list (kind, shell/layer, idx) for the recovery patch of 3D
    vertex (sphere vertex s, shell k).

    Tri-face fans at shells k-1, k, k+1 (the off-shell fans provide the
    dB_r/dr information that a single shell cannot — without them the
    G_rr gradient direction is a null space of the fit) + rect-face fans
    in layers k-1 and k.  At radial boundaries the missing side is
    compensated with one extra layer of rect faces (deeper one-sided
    patch)."""
    faces = []
    for ks in (k - 1, k, k + 1):
        if 0 <= ks <= mesh.N_r:
            for t in mesh.vert_tris[s]:
                faces.append((0, ks, t))
    for kl in (k - 1, k):
        if 0 <= kl < mesh.N_r:
            for e in mesh.vert_sedges[s]:
                faces.append((1, kl, e))
    if k == 0 or k == mesh.N_r:
        kl2 = 1 if k == 0 else mesh.N_r - 2
        if 0 <= kl2 < mesh.N_r:
            for e in mesh.vert_sedges[s]:
                faces.append((1, kl2, e))
    return faces


def face_global_index(mesh, face):
    kind, k, idx = face
    return mesh.tri_face_idx(k, idx) if kind == 0 else mesh.rect_face_idx(k, idx)


# Basis of trace-free 3x3 matrices (8-dim): 6 off-diagonal + 2 diagonal.
# Used to impose div B = 0 on the linear fit — without it the fit has a
# weak/null "trace" mode (exactly null at the 12 valence-5 vertices,
# sigma ~ 1e-2 at valence-6) that wrecks the recovery; with it every
# patch has condition number ~11.
def _traceless_embedding():
    N = np.zeros((12, 11))
    N[:3, :3] = np.eye(3)
    basis = []
    for i in range(3):
        for j in range(3):
            if i != j:
                E = np.zeros((3, 3))
                E[i, j] = 1.0
                basis.append(E)
    basis.append(np.diag([1.0, -1.0, 0.0]) / np.sqrt(2.0))
    basis.append(np.diag([1.0, 1.0, -2.0]) / np.sqrt(6.0))
    for b, Bm in enumerate(basis):
        N[3:, 3 + b] = Bm.ravel()
    return N


_N_TRACELESS = _traceless_embedding()


def build_recovery(mesh, n=4, div_free=True):
    """Precompute per-vertex recovery: for each 3D vertex, the list of
    patch face global indices and the (3, nf) weight matrix mapping their
    fluxes to the fitted B0 (and the (9, nf) gradient weights).

    Returns dict with 'faces' (list of index arrays), 'w_B' (list of
    (3,nf)), 'w_G' (list of (9,nf)), 'cond' (array of LSQ condition
    numbers, for the conditioning study)."""
    faces_all, w_B_all, w_G_all, cond_all = [], [], [], []
    for k in range(mesh.N_r + 1):
        r = mesh.radii[k]
        for s in range(mesh.N_vert_s):
            x_v = r * mesh.sphere_v[s]
            patch = vertex_patch(mesh, s, k)
            kinds = [f[0] for f in patch]
            ids = [(f[1], f[2]) for f in patch]
            A, M = _face_moments(mesh, kinds, ids, x_v, n=n)
            nf = len(patch)
            # rows: flux_f = A_f · B0 + M_f : G   (12 unknowns)
            rows = np.concatenate([A, M.reshape(nf, 9)], axis=1)
            # normalize each row by |A_f| to balance face sizes
            scale = 1.0 / np.linalg.norm(A, axis=1)
            rows_n = rows * scale[:, None]
            if div_free:
                # solve in the trace-free-G subspace (div B = 0)
                rc = rows_n @ _N_TRACELESS
                pinv = _N_TRACELESS @ np.linalg.pinv(rc, rcond=1e-8)
                sv_mat = rc
            else:
                pinv = np.linalg.pinv(rows_n, rcond=1e-10)
                sv_mat = rows_n
            weights = pinv * scale[None, :]
            faces_all.append(
                np.array([face_global_index(mesh, f) for f in patch], dtype=int)
            )
            w_B_all.append(weights[:3])
            w_G_all.append(weights[3:])
            sv = np.linalg.svd(sv_mat, compute_uv=False)
            cond_all.append(sv[0] / sv[-1])
    return {
        "faces": faces_all,
        "w_B": w_B_all,
        "w_G": w_G_all,
        "cond": np.array(cond_all),
    }


def vertex_field(mesh, rec, B_f):
    """Apply recovery weights: fitted B vector at every 3D vertex, (N_verts, 3)."""
    Bv = np.zeros((mesh.N_verts, 3))
    for vi in range(mesh.N_verts):
        f = rec["faces"][vi]
        Bv[vi] = rec["w_B"][vi] @ B_f[f]
    return Bv


def vertex_gradient(mesh, rec, B_f):
    """Fitted gradient tensor G (row-major dB_i/dx_j) at every vertex."""
    Gv = np.zeros((mesh.N_verts, 3, 3))
    for vi in range(mesh.N_verts):
        f = rec["faces"][vi]
        Gv[vi] = (rec["w_G"][vi] @ B_f[f]).reshape(3, 3)
    return Gv


def recovery_gather(mesh, Bv, pts, hints=None):
    """Hat-function (Whitney 0-form × linear-in-zeta) interpolation of
    per-vertex B vectors at Cartesian points.  Fully C0."""
    tri, layer, lam, zeta = locate(mesh, pts, hints)
    N = len(pts)
    B = np.zeros((N, 3))
    for i in range(N):
        if layer[i] < 0:
            continue
        t, k = tri[i], layer[i]
        sv = mesh.tri_verts[t]
        for a in range(3):
            B[i] += lam[i, a] * (
                (1.0 - zeta[i]) * Bv[mesh.vert_idx(k, sv[a])]
                + zeta[i] * Bv[mesh.vert_idx(k + 1, sv[a])]
            )
    return B


def circulation_cochain(mesh, E_func, n=8):
    """Edge 1-cochain of an analytic field: E_e[e] = ∫_e E·dl.

    Horizontal edges are great-circle arcs at shell radius (parametrized
    by normalized linear interpolation, matching the mesh convention);
    vertical edges are radial segments."""
    E_e = np.zeros(mesh.N_edges)
    x, w = _gauss_legendre_01(n)

    # horizontal edges, all shells
    s0 = mesh.sphere_edges[:, 0]
    s1 = mesh.sphere_edges[:, 1]
    u0 = mesh.sphere_v[s0]
    u1 = mesh.sphere_v[s1]
    P = u0[:, None, :] * (1 - x)[None, :, None] + u1[:, None, :] * x[None, :, None]
    Pn = np.linalg.norm(P, axis=2, keepdims=True)
    Nh = P / Pn
    dPu = (u1 - u0)[:, None, :]
    dNu = (dPu - Nh * np.sum(Nh * dPu, axis=2, keepdims=True)) / Pn
    for k in range(mesh.N_r + 1):
        r = mesh.radii[k]
        X = r * Nh
        dl = r * dNu
        Eq = E_func(X.reshape(-1, 3)).reshape(X.shape)
        E_e[k * mesh.N_edge_s : (k + 1) * mesh.N_edge_s] = np.einsum(
            "eqx,eqx,q->e", Eq, dl, w)

    # vertical edges
    off = (mesh.N_r + 1) * mesh.N_edge_s
    for k in range(mesh.N_r):
        ra, rb = mesh.radii[k], mesh.radii[k + 1]
        rq = ra + (rb - ra) * x                     # (nq,)
        X = rq[None, :, None] * mesh.sphere_v[:, None, :]   # (N_vert_s, nq, 3)
        Eq = E_func(X.reshape(-1, 3)).reshape(X.shape)
        # dl = (rb-ra) * u_hat dt
        circ = (rb - ra) * np.einsum("sqx,sx,q->s", Eq, mesh.sphere_v, w)
        E_e[off + k * mesh.N_vert_s : off + (k + 1) * mesh.N_vert_s] = circ
    return E_e


# =========================================================================
# Analytic fields for validation
# =========================================================================


def deutsch_fields(t, Bp=1.0, Omega=0.2, alpha=np.deg2rad(60.0)):
    """Retarded rotating-point-dipole ("Deutsch") B and E at time t, in
    the code's conventions (dec_field_solver_impl.hpp deutsch_*_impl).
    Returns (B_func, E_func) taking (N,3) points."""
    m_perp = Bp * np.sin(alpha)
    m_par = Bp * np.cos(alpha)

    def moments(tr):
        c, s = np.cos(Omega * tr), np.sin(Omega * tr)
        m = np.stack([m_perp * c, m_perp * s, np.full_like(c, m_par)], axis=-1)
        dm = np.stack([-m_perp * Omega * s, m_perp * Omega * c,
                       np.zeros_like(c)], axis=-1)
        ddm = np.stack([-m_perp * Omega**2 * c, -m_perp * Omega**2 * s,
                        np.zeros_like(c)], axis=-1)
        return m, dm, ddm

    def B_func(X):
        X = np.asarray(X, dtype=np.float64)
        r = np.linalg.norm(X, axis=1)
        n = X / r[:, None]
        m, dm, ddm = moments(t - r)
        ndm = np.sum(n * m, axis=1, keepdims=True)
        nddm = np.sum(n * dm, axis=1, keepdims=True)
        nd2m = np.sum(n * ddm, axis=1, keepdims=True)
        Bn = (3 * ndm * n - m) / r[:, None] ** 3
        Bi = (3 * nddm * n - dm) / r[:, None] ** 2
        Br = (nd2m * n - ddm) / r[:, None]
        return Bn + Bi + Br

    def E_func(X):
        X = np.asarray(X, dtype=np.float64)
        r = np.linalg.norm(X, axis=1)
        n = X / r[:, None]
        _, dm, ddm = moments(t - r)
        return (np.cross(n, dm) / r[:, None] ** 2
                + np.cross(n, ddm) / r[:, None])

    return B_func, E_func

def dipole_B(X, m=np.array([0.0, 0.0, 1.0])):
    """Point dipole: B = (3 (m·r̂) r̂ − m)/r³ (Bp=1 convention of the code)."""
    X = np.asarray(X, dtype=np.float64)
    r = np.linalg.norm(X, axis=1, keepdims=True)
    rh = X / r
    mdotr = rh @ m
    return (3.0 * mdotr[:, None] * rh - m[None, :]) / r**3


def constant_B(B0):
    B0 = np.asarray(B0, dtype=np.float64)
    return lambda X: np.broadcast_to(B0, (len(X), 3)).copy()


def linear_B(B0, G):
    """B(x) = B0 + G·x with trace-free G (so div B = 0)."""
    B0 = np.asarray(B0, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    return lambda X: B0[None, :] + X @ G.T
