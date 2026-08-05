#pragma once

#include "core/buffer.hpp"
#include "core/exec_tags.h"
#include "systems/prismatic/dec_solver_geometry.hpp"
#include "systems/prismatic/prismatic_d1_local.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_local.h"
#include "systems/prismatic/prismatic_mesh_local_ptrs.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "utils/gauss_quadrature.h"
#include "utils/logger.h"
#include <vector>

namespace Aperture {

// =========================================================================
// Phase 4.1b B1 — distributed DEC solver core.
//
// Framework-free operator library bound to one rank's partition: builds
// the local mesh data + d1/d1^T blocks and implements the flat solver's
// per-step kernels over them.  Field state is NOT owned here — every
// kernel takes combined local-layout buffers:
//
//   edge buffers: [h_edge locals | v_edge locals],  split at e_split()
//   face buffers: [tri_face locals | rect_face locals], split at b_split()
//
// Each block is a cochain layout's local ordering (owned ascending
// global, then ghosts ascending global).  Under a single_rank partition
// this is EXACTLY the global mesh ordering, so the existing registered
// prismatic_edge_field / prismatic_face_field buffers can be passed in
// unchanged and results are bit-compatible with the global kernels.
//
// Halo contract (see PHASE_4_1B_PLAN.md): kernels write OWNED rows only
// and read owned + ghost columns.  The caller must refresh ghosts at
// the sync points:  exchange E (h+v) before faraday(), exchange B
// (tri+rect) before ampere().  Boundary / damping routines touch owned
// elements only and need no halo.
//
// Kernels do not sync internally (same-stream launches serialize);
// callers sync before host reads, exactly like dec_field_solver.
// =========================================================================

// Inner rotating-conductor boundary parameters (mirrors the members of
// dec_field_solver; see apply_inner_bc there for the physics notes).
struct dec_inner_bc_params {
  Scalar Bp = 1.0;
  Scalar Omega = 1.0;
  Scalar obliquity = 0.0;
  // Body-frame multipole extensions (shifted dipole / quadrupole /
  // shifted quadrupole; see stellar_extras in dec_solver_geometry.hpp).
  // The all-zero default is the centered dipole and reproduces the
  // legacy BC bitwise.  Incompatible with use_deutsch (the retarded
  // solution is point-dipole only; dec_field_solver::init aborts).
  stellar_extras stellar;
  bool use_deutsch = false;
  bool overwrite_b = true;
  // Frame dragging ("fake GR"): the corotation EMF is set by the star's
  // rotation RELATIVE TO the local dragged frame,
  // omega_eff(r) = (Omega - omega_lt(r)) / alpha(r) with
  // omega_lt(r) = omega_lt0 * (lt_r_star / r)^lt_p about the SPIN axis
  // (z).  omega_lt0 = 0 recovers flat spacetime exactly.
  //
  // THE 1/alpha IS NOT DECORATION.  The BC prescribes the FIDO electric
  // field, and the FIDO velocity of material moving at COORDINATE rate
  // Omega follows from inverting the transport law the pusher integrates,
  // dx/dt = alpha v - beta:
  //
  //     v = (dx/dt - v_LT)/alpha = (Omega - omega_lt(r)) (zhat x r) / alpha
  //
  // so E = -v x B carries the 1/alpha.  Omitting it makes the imposed
  // surface EMF 29% too small at compactness 0.5 -- a permanent
  // boundary-vs-interior mismatch that pumps a boundary layer.  That is
  // what killed job 5162728 (see that run dir's ABORTED.md).
  //
  // lapse_compactness MUST be the EFFECTIVE one: zero whenever the solver
  // and pusher are running alpha == 1 (use_gr_lapse = false), or the BC
  // would divide by a lapse the rest of the scheme does not use -- the
  // same class of inconsistency in the opposite direction.
  Scalar omega_lt0 = 0.0;
  Scalar lt_r_star = 1.0;
  int lt_p = 3;
  Scalar lapse_compactness = 0.0;
};

template <typename ExecPolicy>
class dec_solver_dist {
 public:
  void build(const prismatic_mesh& mesh, const prismatic_mesh_partition& mp) {
    m_mesh = &mesh;
    m_mp = &mp;
    m_ml = prismatic_mesh_local::build(mesh, mp, ExecPolicy::data_mem_type());
    m_d1 = prismatic_d1_local::build(mesh, mp, ExecPolicy::data_mem_type());

    m_e_split = mp.layout(cochain_type::h_edge).local_size();
    m_n_edges_local = m_e_split + mp.layout(cochain_type::v_edge).local_size();
    m_b_split = mp.layout(cochain_type::tri_face).local_size();
    m_n_faces_local =
        m_b_split + mp.layout(cochain_type::rect_face).local_size();

    m_lp_host = make_local_ptrs_host(m_ml, m_d1);
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    if (ExecPolicy::data_mem_type() != MemType::host_only) {
      m_ml.copy_to_device();
      auto dev_copy = [](auto& blk) {
        blk.row_ptr.copy_to_device();
        blk.col_idx.copy_to_device();
        blk.val.copy_to_device();
      };
      dev_copy(m_d1.d1_tri_h);
      dev_copy(m_d1.d1_rect_h);
      dev_copy(m_d1.d1_rect_v);
      dev_copy(m_d1.d1t_h_tri);
      dev_copy(m_d1.d1t_h_rect);
      dev_copy(m_d1.d1t_v_rect);
      m_lp_dev = make_local_ptrs_dev(m_ml, m_d1);
    }
#endif
  }

  // ---- Sizes for allocating field buffers ----
  int n_edges_local() const { return m_n_edges_local; }
  int n_faces_local() const { return m_n_faces_local; }
  int e_split() const { return m_e_split; }
  int b_split() const { return m_b_split; }

  const prismatic_mesh_partition& partition() const { return *m_mp; }

  prismatic_mesh_local_ptrs get_lp(exec_tags::host) const { return m_lp_host; }
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  prismatic_mesh_local_ptrs get_lp(exec_tags::device) const { return m_lp_dev; }
#endif

  // -----------------------------------------------------------------------
  // Faraday half-step: B[f] -= dt * (d1 E)[f] on owned faces.
  // Requires fresh E ghosts (h + v).
  // -----------------------------------------------------------------------
  void faraday(buffer<Scalar>& E, buffer<Scalar>& B, double dt) {
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, dt, es = m_e_split, bs = m_b_split] LAMBDA(auto E_e, auto B_f) {
          ExecPolicy::loop(0, lp.n_owned_tri, [&] LAMBDA(int f) {
            Scalar curl_E = Scalar(0);
            for (int j = lp.d1_tri_h_row[f]; j < lp.d1_tri_h_row[f + 1]; j++) {
              curl_E += lp.d1_tri_h_val[j] * E_e[lp.d1_tri_h_col[j]];
            }
            B_f[f] -= dt * curl_E;
          });
          ExecPolicy::loop(0, lp.n_owned_rect, [&] LAMBDA(int f) {
            Scalar curl_E = Scalar(0);
            for (int j = lp.d1_rect_h_row[f]; j < lp.d1_rect_h_row[f + 1]; j++) {
              curl_E += lp.d1_rect_h_val[j] * E_e[lp.d1_rect_h_col[j]];
            }
            for (int j = lp.d1_rect_v_row[f]; j < lp.d1_rect_v_row[f + 1]; j++) {
              curl_E += lp.d1_rect_v_val[j] * E_e[es + lp.d1_rect_v_col[j]];
            }
            B_f[bs + f] -= dt * curl_E;
          });
        },
        E, B);
  }

  // -----------------------------------------------------------------------
  // Frame dragging ("fake GR", Philippov+2015b / Philippov & Spitkovsky
  // 2018 in slow-rotation shift-only form): Faraday advances B with the
  // EFFECTIVE circulation of  E + v_LT x B,  v_LT = omega_lt(r) ẑ x r.
  // The steady corotation state of a star spun at Omega with surface EMF
  // (Omega - omega_lt) — the Muslimov-Tsygan reduced-rho_GJ configuration
  // — is an exact equilibrium of this pair of modifications; without the
  // volume term the sub-corotation profile omega_eff(r) cannot develop.
  //
  // Discretely, W_e = ∮_e (v_LT x B)·dl is built as a per-edge linear
  // functional of the ADJACENT face fluxes (the d1t sparsity):
  //     W_e = sum_f w_ef B_f,   sum_f w_ef N_f = A_e,
  // where N_f is the face's uniform-field probe vector (flux of the
  // Cartesian basis fields through f, computed with the SAME quadrature
  // as every other flux in this file — so orientation conventions are
  // inherited, not re-derived) and A_e = ∮_e dl x v_LT.
  //
  // ACCURACY (measured, tests/test_dec_frame_drag.cpp): v-edges are
  // exact to round-off for uniform B.  h-edges carry an O(h²)
  // truncation — every adjacent face normal is ⊥ the edge tangent to
  // O(h), so the B_parallel part of the circulation (itself O(h²) on
  // the curved arc) is unrepresentable by the stencil; max edge error
  // converges 1.09e-3 / 2.60e-4 / 6.5e-5 at L2/L3/L4 (ratios 4.19,
  // 3.99).  Second order on a term that is itself an
  // O(omega_lt/Omega) correction — far below the solver's own
  // first-order quasi-static Hodge tier.
  //
  // Because W enters ONLY through d1 (Faraday), div B stays exactly
  // conserved (d∘d = 0) and charge conservation is untouched.  The
  // Ampere-side shift term (v_LT x E) is dropped: it is
  // O(omega_lt r/c · E/B) ~ 1% of B at the star and dies as r^-3 —
  // documented approximation, matching the "modified Faraday equation"
  // scope of PS18.
  // -----------------------------------------------------------------------

  // Build the per-edge weights (host, init-time; O(N_local) quadratures).
  void build_frame_drag(Scalar omega0, Scalar r_star, int lt_p) {
    auto lp = m_lp_host;
    auto mp = m_mesh->host_ptrs();

    m_fd_h_tri_val.set_memtype(ExecPolicy::data_mem_type());
    m_fd_h_rect_val.set_memtype(ExecPolicy::data_mem_type());
    m_fd_v_rect_val.set_memtype(ExecPolicy::data_mem_type());
    m_fd_h_tri_val.resize(m_d1.d1t_h_tri.val.size());
    m_fd_h_rect_val.resize(m_d1.d1t_h_rect.val.size());
    m_fd_v_rect_val.resize(m_d1.d1t_v_rect.val.size());

    // ---- Per-local-face uniform-field probe vectors N_f = ∫ dA ----
    // (3 components accumulated in one quadrature pass per face.)
    const double gxs[5] = {0.1488743389816312, 0.4333953941292472,
                           0.6794095682990244, 0.8650633666889845,
                           0.9739065285171717};
    const double gws[5] = {0.2955242247147529, 0.2692667193099963,
                           0.2190863625159821, 0.1494513491505806,
                           0.0666713443086881};
    // 10 nodes/weights on [0,1]
    double n01[10], w01[10];
    for (int i = 0; i < 5; i++) {
      n01[2 * i] = 0.5 + 0.5 * gxs[i];
      n01[2 * i + 1] = 0.5 - 0.5 * gxs[i];
      w01[2 * i] = 0.5 * gws[i];
      w01[2 * i + 1] = 0.5 * gws[i];
    }

    std::vector<double> Ntri(size_t(3) * lp.n_local_tri, 0.0);
    for (int f = 0; f < lp.n_local_tri; f++) {
      gidx_t g = lp.tri_face_l2g[f];
      gidx_t vi0, vi1, vi2;
      tri_face_vertex_ids(mp, g, vi0, vi1, vi2);
      Scalar r0, a0x, a0y, a0z, r1, a1x, a1y, a1z, r2, a2x, a2y, a2z;
      vertex_unit(mp, vi0, r0, a0x, a0y, a0z);
      vertex_unit(mp, vi1, r1, a1x, a1y, a1z);
      vertex_unit(mp, vi2, r2, a2x, a2y, a2z);
      double nxs = 0, nys = 0, nzs = 0;
      for (int iu = 0; iu < 10; iu++) {
        for (int it = 0; it < 10; it++) {
          Scalar x, y, z, nx, ny, nz;
          tri_sphere_sample(r0, a0x, a0y, a0z, a1x, a1y, a1z, a2x, a2y,
                            a2z, Scalar(n01[iu]), Scalar(n01[it]), x, y, z,
                            nx, ny, nz);
          double w = w01[iu] * w01[it];
          nxs += w * nx; nys += w * ny; nzs += w * nz;
        }
      }
      Ntri[3 * f] = nxs; Ntri[3 * f + 1] = nys; Ntri[3 * f + 2] = nzs;
    }
    std::vector<double> Nrect(size_t(3) * lp.n_local_rect, 0.0);
    for (int f = 0; f < lp.n_local_rect; f++) {
      gidx_t g = lp.rect_face_l2g[f];
      gidx_t vi0, vi1, vi3;
      rect_face_vertex_ids(mp, g, vi0, vi1, vi3);
      Scalar r_lo, uax, uay, uaz, r_tmp, ubx, uby, ubz, r_hi, u3x, u3y, u3z;
      vertex_unit(mp, vi0, r_lo, uax, uay, uaz);
      vertex_unit(mp, vi1, r_tmp, ubx, uby, ubz);
      vertex_unit(mp, vi3, r_hi, u3x, u3y, u3z);
      (void)r_tmp; (void)u3x; (void)u3y; (void)u3z;
      double nxs = 0, nys = 0, nzs = 0;
      for (int iu = 0; iu < 10; iu++) {
        for (int iv = 0; iv < 10; iv++) {
          Scalar x, y, z, nx, ny, nz;
          rect_sphere_sample(r_lo, r_hi, uax, uay, uaz, ubx, uby, ubz,
                             Scalar(n01[iu]), Scalar(n01[iv]), x, y, z, nx,
                             ny, nz);
          double w = w01[iu] * w01[iv];
          nxs += w * nx; nys += w * ny; nzs += w * nz;
        }
      }
      Nrect[3 * f] = nxs; Nrect[3 * f + 1] = nys; Nrect[3 * f + 2] = nzs;
    }

    // ---- Per-owned-edge target A_e = ∮ dl x v_LT and min-norm solve ----
    // Full 3-component constraint sum_f w_f N_f = A_e with the min-norm
    // ansatz w_f = c · N_f:  (N Nᵀ) c = A_e.  All three components must
    // be enforced where the stencil supports them — a solve restricted
    // to the plane ⊥ the edge lets the edge-parallel component of
    // sum w_f N_f float free, and B_parallel then leaks an O(h) error
    // into W_e (measured 2-8% at L2).  But the Gram is GENUINELY rank-2
    // on v-edges (all adjacent rect normals ⊥ r̂ — and so is the target,
    // A_e ∝ θ̂) and near-rank-2 on boundary h-edges, so the solve is a
    // rank-adaptive pseudo-inverse: 3x3 Jacobi eigensolve, invert only
    // eigenvalues > eps_rel · lambda_max, project the target onto the
    // kept directions.  Dropped-direction residuals are O(h²)|A_e| —
    // second order on a term that is itself an O(omega_lt/Omega)
    // correction.
    int n_degenerate = 0;
    auto solve_edge = [&](const double Le[3], const double Ae[3],
                          const double* Nf, const int* cols, int nf,
                          Scalar* wout) {
      (void)Le;
      double G[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
      for (int j = 0; j < nf; j++) {
        const double* N = &Nf[3 * cols[j]];
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) G[a][b] += N[a] * N[b];
      }
      // Jacobi eigensolve of the symmetric 3x3 Gram: G = V diag(lam) Vᵀ.
      double V[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
      double A[3][3];
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) A[a][b] = G[a][b];
      for (int sweep = 0; sweep < 30; sweep++) {
        double off = std::abs(A[0][1]) + std::abs(A[0][2]) +
                     std::abs(A[1][2]);
        if (off < 1e-14 * (std::abs(A[0][0]) + std::abs(A[1][1]) +
                           std::abs(A[2][2]) + 1e-300)) {
          break;
        }
        for (int p = 0; p < 2; p++) {
          for (int q = p + 1; q < 3; q++) {
            if (std::abs(A[p][q]) < 1e-300) continue;
            double theta = 0.5 * (A[q][q] - A[p][p]) / A[p][q];
            double t = (theta >= 0 ? 1.0 : -1.0) /
                       (std::abs(theta) + std::sqrt(theta * theta + 1.0));
            double cth = 1.0 / std::sqrt(t * t + 1.0), sth = t * cth;
            for (int k = 0; k < 3; k++) {
              double akp = A[k][p], akq = A[k][q];
              A[k][p] = cth * akp - sth * akq;
              A[k][q] = sth * akp + cth * akq;
            }
            for (int k = 0; k < 3; k++) {
              double apk = A[p][k], aqk = A[q][k];
              A[p][k] = cth * apk - sth * aqk;
              A[q][k] = sth * apk + cth * aqk;
            }
            for (int k = 0; k < 3; k++) {
              double vkp = V[k][p], vkq = V[k][q];
              V[k][p] = cth * vkp - sth * vkq;
              V[k][q] = sth * vkp + cth * vkq;
            }
          }
        }
      }
      double lam[3] = {A[0][0], A[1][1], A[2][2]};
      double lam_max = std::max({lam[0], lam[1], lam[2]});
      if (!(lam_max > 0)) {
        for (int j = 0; j < nf; j++) wout[j] = Scalar(0);
        n_degenerate++;
        return;
      }
      // c = sum_kept  v_k (v_k · A_e) / lam_k
      double c[3] = {0, 0, 0};
      const double eps_rel = 1e-6;
      for (int k = 0; k < 3; k++) {
        if (lam[k] <= eps_rel * lam_max) continue;
        double vk[3] = {V[0][k], V[1][k], V[2][k]};
        double proj = (vk[0]*Ae[0] + vk[1]*Ae[1] + vk[2]*Ae[2]) / lam[k];
        c[0] += proj * vk[0];
        c[1] += proj * vk[1];
        c[2] += proj * vk[2];
      }
      for (int j = 0; j < nf; j++) {
        const double* N = &Nf[3 * cols[j]];
        wout[j] = Scalar(c[0]*N[0] + c[1]*N[1] + c[2]*N[2]);
      }
    };

    // The circulation is built from the SHIFT beta = -v_LT, not from v_LT
    // itself: Faraday advances B with curl(alpha E + beta x B).  See
    // frame_drag_shift in dec_solver_geometry.hpp for why the sign is the
    // difference between having a stationary state and having none.
    auto vlt = [&](double x, double y, double z, double v[3]) {
      Scalar vx, vy, vz;
      frame_drag_shift(Scalar(x), Scalar(y), Scalar(z), omega0, r_star,
                       lt_p, vx, vy, vz);
      v[0] = vx; v[1] = vy; v[2] = vz;
    };

    for (int e = 0; e < lp.n_owned_he; e++) {
      gidx_t g = lp.h_edge_l2g[e];
      gidx_t v0, v1;
      h_edge_vertex_ids(mp, g, v0, v1);
      Scalar r0, a0x, a0y, a0z, r1, a1x, a1y, a1z;
      vertex_unit(mp, v0, r0, a0x, a0y, a0z);
      vertex_unit(mp, v1, r1, a1x, a1y, a1z);
      double Le[3] = {double(r1)*a1x - double(r0)*a0x,
                      double(r1)*a1y - double(r0)*a0y,
                      double(r1)*a1z - double(r0)*a0z};
      double Ae[3] = {0, 0, 0};
      for (int i = 0; i < 10; i++) {
        Scalar x, y, z, dlx, dly, dlz;
        h_edge_sphere_sample(r0, a0x, a0y, a0z, a1x, a1y, a1z,
                             Scalar(n01[i]), x, y, z, dlx, dly, dlz);
        double v[3];
        vlt(x, y, z, v);
        Ae[0] += w01[i] * (double(dly)*v[2] - double(dlz)*v[1]);
        Ae[1] += w01[i] * (double(dlz)*v[0] - double(dlx)*v[2]);
        Ae[2] += w01[i] * (double(dlx)*v[1] - double(dly)*v[0]);
      }
      // Adjacent faces: tri block then rect block, solved TOGETHER (one
      // combined stencil) so the weights share one min-norm solution.
      int cols[8]; double Nf[24]; Scalar w[8];
      int nf = 0;
      const int jt0 = lp.d1t_h_tri_row[e], jt1 = lp.d1t_h_tri_row[e + 1];
      const int jr0 = lp.d1t_h_rect_row[e], jr1 = lp.d1t_h_rect_row[e + 1];
      for (int j = jt0; j < jt1 && nf < 8; j++, nf++) {
        int f = lp.d1t_h_tri_col[j];
        Nf[3*nf] = Ntri[3*f]; Nf[3*nf+1] = Ntri[3*f+1]; Nf[3*nf+2] = Ntri[3*f+2];
        cols[nf] = nf;
      }
      for (int j = jr0; j < jr1 && nf < 8; j++, nf++) {
        int f = lp.d1t_h_rect_col[j];
        Nf[3*nf] = Nrect[3*f]; Nf[3*nf+1] = Nrect[3*f+1]; Nf[3*nf+2] = Nrect[3*f+2];
        cols[nf] = nf;
      }
      solve_edge(Le, Ae, Nf, cols, nf, w);
      int k = 0;
      for (int j = jt0; j < jt1; j++, k++) m_fd_h_tri_val[j] = w[k];
      for (int j = jr0; j < jr1; j++, k++) m_fd_h_rect_val[j] = w[k];
    }

    for (int e = 0; e < lp.n_owned_ve; e++) {
      gidx_t g = lp.v_edge_l2g[e];
      gidx_t v0, v1;
      v_edge_vertex_ids(mp, g, v0, v1);
      Scalar r0, a0x, a0y, a0z, r1, a1x, a1y, a1z;
      vertex_unit(mp, v0, r0, a0x, a0y, a0z);
      vertex_unit(mp, v1, r1, a1x, a1y, a1z);
      (void)a1x; (void)a1y; (void)a1z;
      double Le[3] = {double(r1 - r0) * a0x, double(r1 - r0) * a0y,
                      double(r1 - r0) * a0z};
      double Ae[3] = {0, 0, 0};
      for (int i = 0; i < 10; i++) {
        double rt = (1.0 - n01[i]) * r0 + n01[i] * r1;
        double x = rt * a0x, y = rt * a0y, z = rt * a0z;
        double dl[3] = {double(r1 - r0) * a0x, double(r1 - r0) * a0y,
                        double(r1 - r0) * a0z};
        double v[3];
        vlt(x, y, z, v);
        Ae[0] += w01[i] * (dl[1]*v[2] - dl[2]*v[1]);
        Ae[1] += w01[i] * (dl[2]*v[0] - dl[0]*v[2]);
        Ae[2] += w01[i] * (dl[0]*v[1] - dl[1]*v[0]);
      }
      int cols[8]; double Nf[24]; Scalar w[8];
      int nf = 0;
      const int jr0 = lp.d1t_v_rect_row[e], jr1 = lp.d1t_v_rect_row[e + 1];
      for (int j = jr0; j < jr1 && nf < 8; j++, nf++) {
        int f = lp.d1t_v_rect_col[j];
        Nf[3*nf] = Nrect[3*f]; Nf[3*nf+1] = Nrect[3*f+1]; Nf[3*nf+2] = Nrect[3*f+2];
        cols[nf] = nf;
      }
      solve_edge(Le, Ae, Nf, cols, nf, w);
      int k = 0;
      for (int j = jr0; j < jr1; j++, k++) m_fd_v_rect_val[j] = w[k];
    }

    if (n_degenerate > 0) {
      Logger::print_err(
          "build_frame_drag: {} degenerate edge stencils (weights zeroed)",
          n_degenerate);
    }
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    if (ExecPolicy::data_mem_type() != MemType::host_only) {
      m_fd_h_tri_val.copy_to_device();
      m_fd_h_rect_val.copy_to_device();
      m_fd_v_rect_val.copy_to_device();
    }
#endif
    m_fd_built = true;
    build_frame_drag_transpose();
  }

  // Eeff[e] = alpha_e E[e] + W_e(Bdelta + B0) on OWNED edges -- the full
  // 3+1 Faraday operand curl(alpha E + beta x B) (caller exchanges Eeff
  // ghosts, then feeds it to faraday()/compute_rhs() in place of E).
  // Requires fresh Bdelta ghosts; B0 is static (exchanged once at init).
  //
  // W is built from the SHIFT beta = -v_LT (build_frame_drag), so this is
  // alpha E - v_LT x B.  alpha is applied only when build_lapse has run;
  // with gr_compactness = 0 it is identically 1 and skipped entirely, so
  // the shift-only scheme and the flat scheme are both exactly recovered.
  void frame_drag_eff_E(buffer<Scalar>& E, buffer<Scalar>& Bdelta,
                        buffer<Scalar>& B0, buffer<Scalar>& Eeff) {
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, es = m_e_split, bs = m_b_split, lap = m_lapse_built]
        LAMBDA(auto E_e, auto Bd, auto B0_f, auto Ef, auto wht, auto whr,
               auto wvr, auto alpha_e) {
          ExecPolicy::loop(0, lp.n_owned_he, [&] LAMBDA(int e) {
            Scalar W = Scalar(0);
            for (int j = lp.d1t_h_tri_row[e]; j < lp.d1t_h_tri_row[e + 1];
                 j++) {
              int f = lp.d1t_h_tri_col[j];
              W += wht[j] * (Bd[f] + B0_f[f]);
            }
            for (int j = lp.d1t_h_rect_row[e]; j < lp.d1t_h_rect_row[e + 1];
                 j++) {
              int f = lp.d1t_h_rect_col[j];
              W += whr[j] * (Bd[bs + f] + B0_f[bs + f]);
            }
            Ef[e] = (lap ? alpha_e[e] * E_e[e] : E_e[e]) + W;
          });
          ExecPolicy::loop(0, lp.n_owned_ve, [&] LAMBDA(int e) {
            Scalar W = Scalar(0);
            for (int j = lp.d1t_v_rect_row[e]; j < lp.d1t_v_rect_row[e + 1];
                 j++) {
              int f = lp.d1t_v_rect_col[j];
              W += wvr[j] * (Bd[bs + f] + B0_f[bs + f]);
            }
            Ef[es + e] =
                (lap ? alpha_e[es + e] * E_e[es + e] : E_e[es + e]) + W;
          });
        },
        E, Bdelta, B0, Eeff, m_fd_h_tri_val, m_fd_h_rect_val,
        m_fd_v_rect_val, m_alpha_e);
  }

  bool frame_drag_built() const { return m_fd_built; }
  bool lapse_built() const { return m_lapse_built; }

  // -----------------------------------------------------------------------
  // Adjoint pairing of the frame-drag coupling (the Ampere-side beta x E
  // term, restored for STABILITY rather than accuracy).
  //
  // The Faraday-only W coupling is not skew-adjoint in the discrete
  // energy: sym(h2 d1 W) has O(1) grid-scale eigenvalues localized in
  // the first shells above the star, and the resulting instability
  // (gamma ~ 0.2 at L6, sign-independent, saturating at E/B ~ 0.6 with
  // plasma) is what produced the near-surface tangential-E layer in
  // every fake-GR run -- it grows in VACUUM, no plasma needed.  See
  // legacy/frame_drag_instability/ for the evidence chain and the
  // validating testbed.
  //
  // The cure needs no Whitney mass matrices: with the diagonal Hodge,
  // the exact energy partner of W is its literal transpose.  Adding
  //
  //     H_aux[f] = h2 B[f] + F[f],   F = W^T h1inv^-1 E
  //
  // to Ampere makes
  //
  //     U_full = 1/2 E'h1inv^-1 E + 1/2 B'h2 B + E'h1inv^-1 W B
  //
  // an exact semi-discrete invariant for ANY W (every spurious quadratic
  // cancels algebraically), and W^T h1inv^-1 is simultaneously the
  // weak-form-consistent discretization of the beta x E face flux (the
  // triple-product identity int (beta x B).E = -int (beta x E).B holds
  // exactly under the diagonal masses).  The lapse, when built, scales
  // h2 B only -- the continuum beta x E term carries no alpha.
  //
  // TIME CENTERING IS MANDATORY: under leapfrog, both couplings must
  // act on their time midpoints (W on (B^{n-1/2}+B^{n+1/2})/2, W^T on
  // (E^n+E^{n+1})/2, two Picard sweeps each; dec_field_solver's
  // update_explicit orchestrates this).  Measured at L4 in vacuum:
  // one-sided gamma = 0.038; explicit adjoint 0.2 (worse); Ampere-side
  // centering only 0.015; both centered: monotone energy decay over
  // 14 periods.
  //
  // DISTRIBUTED ASSEMBLY: the face-major transpose below carries only
  // this rank's OWNED edges (the exact transpose of what this rank owns
  // of W).  frame_drag_aux_F therefore produces a PARTIAL sum on every
  // local face; the caller completes it with reduce_face (ghost slots
  // accumulate to their owner -- each global edge is owned exactly once,
  // so the reduced sum is the exact global W^T row) followed by
  // exchange_face to refill ghosts.  Single-rank both are no-ops.
  // -----------------------------------------------------------------------

  // Face-major transpose of the W CSR blocks with 1/h1inv folded into
  // the values (host, init-time; called by build_frame_drag).  Column
  // indices address the E buffer directly (h edges as-is, v edges at
  // e_split + e).
  void build_frame_drag_transpose() {
    auto lp = m_lp_host;
    std::vector<int> t_row(size_t(lp.n_local_tri) + 1, 0);
    std::vector<int> r_row(size_t(lp.n_local_rect) + 1, 0);
    for (int e = 0; e < lp.n_owned_he; e++) {
      for (int j = lp.d1t_h_tri_row[e]; j < lp.d1t_h_tri_row[e + 1]; j++)
        t_row[lp.d1t_h_tri_col[j] + 1]++;
      for (int j = lp.d1t_h_rect_row[e]; j < lp.d1t_h_rect_row[e + 1]; j++)
        r_row[lp.d1t_h_rect_col[j] + 1]++;
    }
    for (int e = 0; e < lp.n_owned_ve; e++)
      for (int j = lp.d1t_v_rect_row[e]; j < lp.d1t_v_rect_row[e + 1]; j++)
        r_row[lp.d1t_v_rect_col[j] + 1]++;
    for (int f = 0; f < lp.n_local_tri; f++) t_row[f + 1] += t_row[f];
    for (int f = 0; f < lp.n_local_rect; f++) r_row[f + 1] += r_row[f];

    std::vector<int> t_col(t_row[lp.n_local_tri]);
    std::vector<Scalar> t_val(t_row[lp.n_local_tri]);
    std::vector<int> r_col(r_row[lp.n_local_rect]);
    std::vector<Scalar> r_val(r_row[lp.n_local_rect]);
    std::vector<int> pt(t_row.begin(), t_row.end() - 1);
    std::vector<int> pr(r_row.begin(), r_row.end() - 1);
    for (int e = 0; e < lp.n_owned_he; e++) {
      const Scalar wi = Scalar(1) / lp.h_edge_hodge1_inv[e];
      for (int j = lp.d1t_h_tri_row[e]; j < lp.d1t_h_tri_row[e + 1]; j++) {
        int f = lp.d1t_h_tri_col[j];
        t_col[pt[f]] = e;
        t_val[pt[f]] = m_fd_h_tri_val[j] * wi;
        pt[f]++;
      }
      for (int j = lp.d1t_h_rect_row[e]; j < lp.d1t_h_rect_row[e + 1]; j++) {
        int f = lp.d1t_h_rect_col[j];
        r_col[pr[f]] = e;
        r_val[pr[f]] = m_fd_h_rect_val[j] * wi;
        pr[f]++;
      }
    }
    for (int e = 0; e < lp.n_owned_ve; e++) {
      const Scalar wi = Scalar(1) / lp.v_edge_hodge1_inv[e];
      for (int j = lp.d1t_v_rect_row[e]; j < lp.d1t_v_rect_row[e + 1]; j++) {
        int f = lp.d1t_v_rect_col[j];
        r_col[pr[f]] = m_e_split + e;
        r_val[pr[f]] = m_fd_v_rect_val[j] * wi;
        pr[f]++;
      }
    }

    auto upload = [&](auto& dst, auto& src) {
      dst.set_memtype(ExecPolicy::data_mem_type());
      dst.resize(src.size());
      for (size_t i = 0; i < src.size(); i++) dst[i] = src[i];
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
      if (ExecPolicy::data_mem_type() != MemType::host_only)
        dst.copy_to_device();
#endif
    };
    upload(m_fdt_tri_row, t_row);
    upload(m_fdt_tri_col, t_col);
    upload(m_fdt_tri_val, t_val);
    upload(m_fdt_rect_row, r_row);
    upload(m_fdt_rect_col, r_col);
    upload(m_fdt_rect_val, r_val);
    m_fdt_built = true;
  }

  // F[f] = this rank's owned-edge part of (W^T h1inv^-1 E)[f], on ALL
  // local faces (gather over the face-major transpose; race-free).
  // Reads OWNED edge slots of E only.  Caller must reduce_face +
  // exchange_face before feeding F to ampere_fd / compute_rhs_fd.
  void frame_drag_aux_F(buffer<Scalar>& E, buffer<Scalar>& F) {
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, bs = m_b_split]
        LAMBDA(auto E_e, auto F_f, auto trow, auto tcol, auto tval,
               auto rrow, auto rcol, auto rval) {
          ExecPolicy::loop(0, lp.n_local_tri, [&] LAMBDA(int f) {
            Scalar a = Scalar(0);
            for (int j = trow[f]; j < trow[f + 1]; j++)
              a += tval[j] * E_e[tcol[j]];
            F_f[f] = a;
          });
          ExecPolicy::loop(0, lp.n_local_rect, [&] LAMBDA(int f) {
            Scalar a = Scalar(0);
            for (int j = rrow[f]; j < rrow[f + 1]; j++)
              a += rval[j] * E_e[rcol[j]];
            F_f[bs + f] = a;
          });
        },
        E, F, m_fdt_tri_row, m_fdt_tri_col, m_fdt_tri_val, m_fdt_rect_row,
        m_fdt_rect_col, m_fdt_rect_val);
  }

  // Ampere with the adjoint frame-drag term: H_aux = [alpha] h2 B + F.
  // Separate from ampere_impl so the flat and one-sided kernels stay
  // byte-identical (this file has been bitten by a codegen cliff,
  // b3e4c17a).
  template <bool WithLapse>
  void ampere_fd_impl(buffer<Scalar>& E, buffer<Scalar>& B,
                      buffer<Scalar>& F, buffer<Scalar>& J, double dt) {
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, dt, es = m_e_split, bs = m_b_split]
        LAMBDA(auto E_e, auto B_f, auto F_f, auto J_e, auto alpha_f) {
          ExecPolicy::loop(0, lp.n_owned_he, [&] LAMBDA(int e) {
            Scalar curl_H = Scalar(0);
            for (int j = lp.d1t_h_tri_row[e]; j < lp.d1t_h_tri_row[e + 1];
                 j++) {
              int f = lp.d1t_h_tri_col[j];
              Scalar H = lp.tri_face_hodge2[f] * B_f[f];
              if constexpr (WithLapse) H *= alpha_f[f];
              curl_H += lp.d1t_h_tri_val[j] * (H + F_f[f]);
            }
            for (int j = lp.d1t_h_rect_row[e]; j < lp.d1t_h_rect_row[e + 1];
                 j++) {
              int f = lp.d1t_h_rect_col[j];
              Scalar H = lp.rect_face_hodge2[f] * B_f[bs + f];
              if constexpr (WithLapse) H *= alpha_f[bs + f];
              curl_H += lp.d1t_h_rect_val[j] * (H + F_f[bs + f]);
            }
            E_e[e] += dt * lp.h_edge_hodge1_inv[e] * (curl_H - J_e[e]);
          });
          ExecPolicy::loop(0, lp.n_owned_ve, [&] LAMBDA(int e) {
            Scalar curl_H = Scalar(0);
            for (int j = lp.d1t_v_rect_row[e]; j < lp.d1t_v_rect_row[e + 1];
                 j++) {
              int f = lp.d1t_v_rect_col[j];
              Scalar H = lp.rect_face_hodge2[f] * B_f[bs + f];
              if constexpr (WithLapse) H *= alpha_f[bs + f];
              curl_H += lp.d1t_v_rect_val[j] * (H + F_f[bs + f]);
            }
            E_e[es + e] +=
                dt * lp.v_edge_hodge1_inv[e] * (curl_H - J_e[es + e]);
          });
        },
        E, B, F, J, m_alpha_f);
  }

  void ampere_fd(buffer<Scalar>& E, buffer<Scalar>& B, buffer<Scalar>& F,
                 buffer<Scalar>& J, double dt) {
    if (m_lapse_built) ampere_fd_impl<true>(E, B, F, J, dt);
    else ampere_fd_impl<false>(E, B, F, J, dt);
  }

  bool frame_drag_transpose_built() const { return m_fdt_built; }

  // Test access: the per-element lapse arrays (edge / face layout as E / B).
  const buffer<Scalar>& alpha_e() const { return m_alpha_e; }
  const buffer<Scalar>& alpha_f() const { return m_alpha_f; }

  // -----------------------------------------------------------------------
  // Per-element lapse alpha(r) = sqrt(1 - compactness * r_star / r) for the
  // 3+1 constitutive relations
  //
  //     E_aux = alpha E + (beta x B)      (Faraday, in frame_drag_eff_E)
  //     H_aux = alpha (hodge2 B)          (Ampere, in ampere<true>)
  //
  // ACCURACY, and why it is better than it looks: alpha depends only on r,
  // and this mesh is a sphere-cross-radius product, so
  //   - h-edges lie at constant r  -> alpha is EXACT on them
  //   - triangular faces likewise  -> EXACT
  //   - v-edges span [r_k, r_k+1]  -> length-averaged (dl = dr)
  //   - rect faces span the same   -> area-averaged (dA ~ r dr dphi)
  // so the only approximation is a radial average over one log shell,
  // O(h^2 alpha''), on the two element kinds that straddle shells.
  //
  // NOT exactly skew-adjoint.  The GR KS solver gets discrete energy
  // conservation by folding the lapse inside the Whitney mass matrices
  // (M1alpha sandwiched symmetrically); with this solver's DIAGONAL Hodge
  // that is unavailable, and the residual is the commutator [d1, alpha],
  // i.e. O(h * dlnalpha/dlnr) -- the same first-order tier as the
  // solver's quasi-static truncation, but it means long GR runs should
  // have their energy budget watched rather than assumed.
  // -----------------------------------------------------------------------
  void build_lapse(Scalar compactness, Scalar r_star) {
    auto lp = m_lp_host;
    auto mp = m_mesh->host_ptrs();
    m_alpha_e.set_memtype(ExecPolicy::data_mem_type());
    m_alpha_f.set_memtype(ExecPolicy::data_mem_type());
    m_alpha_e.resize(n_edges_local());
    m_alpha_f.resize(n_faces_local());
    m_alpha_e.assign(Scalar(1));
    m_alpha_f.assign(Scalar(1));

    const int N_es = mp.N_edge_s, N_vs = mp.N_vert_s, N_tri = mp.N_tri;
    // Radial average of alpha over [ra, rb] with weight r^wpow.
    auto avg = [&](double ra, double rb, int wpow) {
      double num = gauss_quad(
          [&](double t) -> double {
            double r = ra + t * (rb - ra);
            double w = (wpow == 1) ? r : 1.0;
            return w * double(gr_lapse(Scalar(r), compactness, r_star));
          },
          0.0, 1.0);
      double den = gauss_quad(
          [&](double t) -> double {
            double r = ra + t * (rb - ra);
            return (wpow == 1) ? r : 1.0;
          },
          0.0, 1.0);
      return Scalar(num / den);
    };

    // FULL LOCAL range, ghosts included: ampere_impl<WithLapse> and
    // ampere_fd read alpha_f at d1t COLUMN indices, which reach ghost
    // faces.  Filling owned slots only leaves alpha = 1 ghosts -- a
    // permanent (1 - alpha(R*)) H_aux mismatch along every partition
    // seam, which is what killed the first full-GR pilot (job 5166539:
    // charge runaways at the pentagon-vertex seam corners, E/B > 100).
    // alpha is deterministic geometry, so ghosts are computed directly
    // from l2g -- no exchange needed, and every rank agrees exactly.
    // Regression: "Lapse: alpha arrays are ghost-consistent across
    // partitions" (tests/test_dec_frame_drag.cpp).
    for (int e = 0; e < lp.n_local_he; e++) {
      const int k = int(lp.h_edge_l2g[e] / N_es);      // constant r
      m_alpha_e[e] = gr_lapse(mp.radii[k], compactness, r_star);
    }
    for (int e = 0; e < lp.n_local_ve; e++) {
      const int k = int(lp.v_edge_l2g[e] / N_vs);
      m_alpha_e[m_e_split + e] = avg(mp.radii[k], mp.radii[k + 1], 0);
    }
    for (int f = 0; f < lp.n_local_tri; f++) {
      const int k = int(lp.tri_face_l2g[f] / N_tri);   // constant r
      m_alpha_f[f] = gr_lapse(mp.radii[k], compactness, r_star);
    }
    for (int f = 0; f < lp.n_local_rect; f++) {
      const int k = int(lp.rect_face_l2g[f] / N_es);
      m_alpha_f[m_b_split + f] = avg(mp.radii[k], mp.radii[k + 1], 1);
    }
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    m_alpha_e.copy_to_device();
    m_alpha_f.copy_to_device();
#endif
    m_lapse_built = true;
  }

  // -----------------------------------------------------------------------
  // Ampere half-step: E[e] += dt * h1inv[e] * ((d1^T h2 B)[e] - J[e]) on
  // owned edges.  Requires fresh B ghosts (tri + rect).
  // -----------------------------------------------------------------------
  // Dispatches on whether the GR lapse is active.  Templating rather than
  // branching keeps the FLAT kernel byte-for-byte what it was -- this is
  // the hot kernel, and this file has already been bitten once by a
  // codegen cliff (the fp64 quadrature unroll, commit b3e4c17a).
  //
  // NOTE J carries NO alpha.  PCTS15's Ampere source is alpha*j - rho*beta
  // with j the FIDO current; since alpha*j - rho*beta = sum q (alpha v -
  // beta) delta = sum q (dx/dt) delta, a charge-conserving deposit along
  // the actual COORDINATE displacement produces that combination already.
  // Adding alpha here would double-count it (and break Gauss's law).  The
  // pusher's shift term is therefore not optional for GR runs: it is what
  // puts the -rho*beta current into J.
  template <bool WithLapse>
  void ampere_impl(buffer<Scalar>& E, buffer<Scalar>& B, buffer<Scalar>& J,
                   double dt) {
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, dt, es = m_e_split, bs = m_b_split]
        LAMBDA(auto E_e, auto B_f, auto J_e, auto alpha_f) {
          ExecPolicy::loop(0, lp.n_owned_he, [&] LAMBDA(int e) {
            Scalar curl_H = Scalar(0);
            for (int j = lp.d1t_h_tri_row[e]; j < lp.d1t_h_tri_row[e + 1]; j++) {
              int f = lp.d1t_h_tri_col[j];
              Scalar H = lp.tri_face_hodge2[f] * B_f[f];
              if constexpr (WithLapse) H *= alpha_f[f];
              curl_H += lp.d1t_h_tri_val[j] * H;
            }
            for (int j = lp.d1t_h_rect_row[e]; j < lp.d1t_h_rect_row[e + 1]; j++) {
              int f = lp.d1t_h_rect_col[j];
              Scalar H = lp.rect_face_hodge2[f] * B_f[bs + f];
              if constexpr (WithLapse) H *= alpha_f[bs + f];
              curl_H += lp.d1t_h_rect_val[j] * H;
            }
            E_e[e] += dt * lp.h_edge_hodge1_inv[e] * (curl_H - J_e[e]);
          });
          ExecPolicy::loop(0, lp.n_owned_ve, [&] LAMBDA(int e) {
            Scalar curl_H = Scalar(0);
            for (int j = lp.d1t_v_rect_row[e]; j < lp.d1t_v_rect_row[e + 1]; j++) {
              int f = lp.d1t_v_rect_col[j];
              Scalar H = lp.rect_face_hodge2[f] * B_f[bs + f];
              if constexpr (WithLapse) H *= alpha_f[bs + f];
              curl_H += lp.d1t_v_rect_val[j] * H;
            }
            E_e[es + e] +=
                dt * lp.v_edge_hodge1_inv[e] * (curl_H - J_e[es + e]);
          });
        },
        E, B, J, m_alpha_f);
  }

  void ampere(buffer<Scalar>& E, buffer<Scalar>& B, buffer<Scalar>& J,
              double dt) {
    if (m_lapse_built) ampere_impl<true>(E, B, J, dt);
    else ampere_impl<false>(E, B, J, dt);
  }

  // -----------------------------------------------------------------------
  // RHS for the semi-implicit path (mirrors dec_field_solver::compute_rhs):
  //   dB_out[f] = -(d1 E)[f],   dE_out[e] = h1inv[e]((d1^T h2 B)[e] - J[e])
  // on owned rows.  Requires fresh E AND B ghosts.
  // -----------------------------------------------------------------------
  void compute_rhs(buffer<Scalar>& E_in, buffer<Scalar>& B_in,
                   buffer<Scalar>& J, buffer<Scalar>& dE_out,
                   buffer<Scalar>& dB_out) {
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, es = m_e_split, bs = m_b_split]
        LAMBDA(auto E_e, auto B_f, auto J_e, auto dE, auto dB) {
          ExecPolicy::loop(0, lp.n_owned_tri, [&] LAMBDA(int f) {
            Scalar curl_E = Scalar(0);
            for (int j = lp.d1_tri_h_row[f]; j < lp.d1_tri_h_row[f + 1]; j++) {
              curl_E += lp.d1_tri_h_val[j] * E_e[lp.d1_tri_h_col[j]];
            }
            dB[f] = -curl_E;
          });
          ExecPolicy::loop(0, lp.n_owned_rect, [&] LAMBDA(int f) {
            Scalar curl_E = Scalar(0);
            for (int j = lp.d1_rect_h_row[f]; j < lp.d1_rect_h_row[f + 1]; j++) {
              curl_E += lp.d1_rect_h_val[j] * E_e[lp.d1_rect_h_col[j]];
            }
            for (int j = lp.d1_rect_v_row[f]; j < lp.d1_rect_v_row[f + 1]; j++) {
              curl_E += lp.d1_rect_v_val[j] * E_e[es + lp.d1_rect_v_col[j]];
            }
            dB[bs + f] = -curl_E;
          });
          ExecPolicy::loop(0, lp.n_owned_he, [&] LAMBDA(int e) {
            Scalar curl_H = Scalar(0);
            for (int j = lp.d1t_h_tri_row[e]; j < lp.d1t_h_tri_row[e + 1]; j++) {
              int f = lp.d1t_h_tri_col[j];
              curl_H += lp.d1t_h_tri_val[j] * lp.tri_face_hodge2[f] * B_f[f];
            }
            for (int j = lp.d1t_h_rect_row[e]; j < lp.d1t_h_rect_row[e + 1]; j++) {
              int f = lp.d1t_h_rect_col[j];
              curl_H +=
                  lp.d1t_h_rect_val[j] * lp.rect_face_hodge2[f] * B_f[bs + f];
            }
            dE[e] = lp.h_edge_hodge1_inv[e] * (curl_H - J_e[e]);
          });
          ExecPolicy::loop(0, lp.n_owned_ve, [&] LAMBDA(int e) {
            Scalar curl_H = Scalar(0);
            for (int j = lp.d1t_v_rect_row[e]; j < lp.d1t_v_rect_row[e + 1]; j++) {
              int f = lp.d1t_v_rect_col[j];
              curl_H +=
                  lp.d1t_v_rect_val[j] * lp.rect_face_hodge2[f] * B_f[bs + f];
            }
            dE[es + e] =
                lp.v_edge_hodge1_inv[e] * (curl_H - J_e[es + e]);
          });
        },
        E_in, B_in, J, dE_out, dB_out);
  }

  // compute_rhs with the adjoint frame-drag face term folded into the
  // Ampere rows: H_aux = h2 B + F (see the adjoint-pairing header note).
  // F must be reduced + exchanged by the caller.  The Faraday rows are
  // identical to compute_rhs (the caller passes Eeff there).
  void compute_rhs_fd(buffer<Scalar>& E_in, buffer<Scalar>& B_in,
                      buffer<Scalar>& F, buffer<Scalar>& J,
                      buffer<Scalar>& dE_out, buffer<Scalar>& dB_out) {
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, es = m_e_split, bs = m_b_split]
        LAMBDA(auto E_e, auto B_f, auto F_f, auto J_e, auto dE, auto dB) {
          ExecPolicy::loop(0, lp.n_owned_tri, [&] LAMBDA(int f) {
            Scalar curl_E = Scalar(0);
            for (int j = lp.d1_tri_h_row[f]; j < lp.d1_tri_h_row[f + 1]; j++) {
              curl_E += lp.d1_tri_h_val[j] * E_e[lp.d1_tri_h_col[j]];
            }
            dB[f] = -curl_E;
          });
          ExecPolicy::loop(0, lp.n_owned_rect, [&] LAMBDA(int f) {
            Scalar curl_E = Scalar(0);
            for (int j = lp.d1_rect_h_row[f]; j < lp.d1_rect_h_row[f + 1]; j++) {
              curl_E += lp.d1_rect_h_val[j] * E_e[lp.d1_rect_h_col[j]];
            }
            for (int j = lp.d1_rect_v_row[f]; j < lp.d1_rect_v_row[f + 1]; j++) {
              curl_E += lp.d1_rect_v_val[j] * E_e[es + lp.d1_rect_v_col[j]];
            }
            dB[bs + f] = -curl_E;
          });
          ExecPolicy::loop(0, lp.n_owned_he, [&] LAMBDA(int e) {
            Scalar curl_H = Scalar(0);
            for (int j = lp.d1t_h_tri_row[e]; j < lp.d1t_h_tri_row[e + 1]; j++) {
              int f = lp.d1t_h_tri_col[j];
              curl_H += lp.d1t_h_tri_val[j] *
                        (lp.tri_face_hodge2[f] * B_f[f] + F_f[f]);
            }
            for (int j = lp.d1t_h_rect_row[e]; j < lp.d1t_h_rect_row[e + 1]; j++) {
              int f = lp.d1t_h_rect_col[j];
              curl_H += lp.d1t_h_rect_val[j] *
                        (lp.rect_face_hodge2[f] * B_f[bs + f] + F_f[bs + f]);
            }
            dE[e] = lp.h_edge_hodge1_inv[e] * (curl_H - J_e[e]);
          });
          ExecPolicy::loop(0, lp.n_owned_ve, [&] LAMBDA(int e) {
            Scalar curl_H = Scalar(0);
            for (int j = lp.d1t_v_rect_row[e]; j < lp.d1t_v_rect_row[e + 1]; j++) {
              int f = lp.d1t_v_rect_col[j];
              curl_H += lp.d1t_v_rect_val[j] *
                        (lp.rect_face_hodge2[f] * B_f[bs + f] + F_f[bs + f]);
            }
            dE[es + e] =
                lp.v_edge_hodge1_inv[e] * (curl_H - J_e[es + e]);
          });
        },
        E_in, B_in, F, J, dE_out, dB_out);
  }

  // -----------------------------------------------------------------------
  // Owned-slot linear combinations for the semi-implicit predictor /
  // corrector (mirrors the update_semi_implicit kernels).  Ghost slots
  // are left untouched — the Picard loop refreshes them by exchange
  // before every compute_rhs.
  //   euler_predict:  tmp = F + dt * rhs
  //   picard_combine: tmp = F + dt * (alpha * rhs_n + beta * rhs_new)
  // -----------------------------------------------------------------------
  void euler_predict(buffer<Scalar>& E, buffer<Scalar>& dE,
                     buffer<Scalar>& tmpE, buffer<Scalar>& B,
                     buffer<Scalar>& dB, buffer<Scalar>& tmpB, double dt) {
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, dt, es = m_e_split, bs = m_b_split]
        LAMBDA(auto E_e, auto dE_e, auto tE, auto B_f, auto dB_f, auto tB) {
          ExecPolicy::loop(0, lp.n_owned_he, [&] LAMBDA(int e) {
            tE[e] = E_e[e] + dt * dE_e[e];
          });
          ExecPolicy::loop(0, lp.n_owned_ve, [&] LAMBDA(int e) {
            tE[es + e] = E_e[es + e] + dt * dE_e[es + e];
          });
          ExecPolicy::loop(0, lp.n_owned_tri, [&] LAMBDA(int f) {
            tB[f] = B_f[f] + dt * dB_f[f];
          });
          ExecPolicy::loop(0, lp.n_owned_rect, [&] LAMBDA(int f) {
            tB[bs + f] = B_f[bs + f] + dt * dB_f[bs + f];
          });
        },
        E, dE, tmpE, B, dB, tmpB);
  }

  void picard_combine(buffer<Scalar>& E, buffer<Scalar>& dE_n,
                      buffer<Scalar>& dE_new, buffer<Scalar>& tmpE,
                      buffer<Scalar>& B, buffer<Scalar>& dB_n,
                      buffer<Scalar>& dB_new, buffer<Scalar>& tmpB,
                      double dt, Scalar alpha, Scalar beta) {
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, dt, alpha, beta, es = m_e_split, bs = m_b_split]
        LAMBDA(auto E_e, auto dEn, auto dEw, auto tE,
               auto B_f, auto dBn, auto dBw, auto tB) {
          ExecPolicy::loop(0, lp.n_owned_he, [&] LAMBDA(int e) {
            tE[e] = E_e[e] + dt * (alpha * dEn[e] + beta * dEw[e]);
          });
          ExecPolicy::loop(0, lp.n_owned_ve, [&] LAMBDA(int e) {
            tE[es + e] =
                E_e[es + e] + dt * (alpha * dEn[es + e] + beta * dEw[es + e]);
          });
          ExecPolicy::loop(0, lp.n_owned_tri, [&] LAMBDA(int f) {
            tB[f] = B_f[f] + dt * (alpha * dBn[f] + beta * dBw[f]);
          });
          ExecPolicy::loop(0, lp.n_owned_rect, [&] LAMBDA(int f) {
            tB[bs + f] =
                B_f[bs + f] + dt * (alpha * dBn[bs + f] + beta * dBw[bs + f]);
          });
        },
        E, dE_n, dE_new, tmpE, B, dB_n, dB_new, tmpB);
  }

  // -----------------------------------------------------------------------
  // Outer damping layer (mirrors dec_field_solver::apply_damping): each
  // rank damps the owned elements whose radial layer falls in the zone.
  // -----------------------------------------------------------------------
  void apply_damping(buffer<Scalar>& E, buffer<Scalar>& B, double dt,
                     int damping_length, Scalar damping_coef,
                     Scalar damping_exponent) {
    if (damping_length <= 0) return;
    const int N_r = m_mesh->m_N_r;
    int k_start = N_r - damping_length;
    if (k_start < 1) k_start = 1;
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, k_start, damping_coef, damping_length, damping_exponent, dt,
         es = m_e_split, bs = m_b_split] LAMBDA(auto E_e, auto B_f) {
          auto sigma_of = [&] LAMBDA(int k) {
            Scalar ramp = Scalar(k - k_start + 1) / Scalar(damping_length);
            return damping_coef * std::pow(ramp, damping_exponent);
          };
          ExecPolicy::loop(0, lp.n_owned_he, [&] LAMBDA(int e) {
            int k = lp.h_edge_radial_layer[e];
            if (k >= k_start)
              E_e[e] *= std::exp(-sigma_of(k) * Scalar(dt));
          });
          ExecPolicy::loop(0, lp.n_owned_ve, [&] LAMBDA(int e) {
            int k = lp.v_edge_radial_layer[e];
            if (k >= k_start)
              E_e[es + e] *= std::exp(-sigma_of(k) * Scalar(dt));
          });
          ExecPolicy::loop(0, lp.n_owned_tri, [&] LAMBDA(int f) {
            int k = lp.tri_face_radial_layer[f];
            if (k >= k_start)
              B_f[f] *= std::exp(-sigma_of(k) * Scalar(dt));
          });
          ExecPolicy::loop(0, lp.n_owned_rect, [&] LAMBDA(int f) {
            int k = lp.rect_face_radial_layer[f];
            if (k >= k_start)
              B_f[bs + f] *= std::exp(-sigma_of(k) * Scalar(dt));
          });
        },
        E, B);
  }

  // -----------------------------------------------------------------------
  // PEC boundary: zero tangential E (h edges) and normal B (tri faces)
  // on shells k = 0 and k = N_r.  Owned elements only — ranks whose slab
  // excludes the boundary shells no-op.
  // -----------------------------------------------------------------------
  void apply_pec_bc(buffer<Scalar>& E, buffer<Scalar>& B) {
    const int N_r = m_mesh->m_N_r;
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, N_r] LAMBDA(auto E_e, auto B_f) {
          ExecPolicy::loop(0, lp.n_owned_he, [&] LAMBDA(int e) {
            int k = lp.h_edge_radial_layer[e];
            if (k == 0 || k == N_r) E_e[e] = Scalar(0);
          });
          ExecPolicy::loop(0, lp.n_owned_tri, [&] LAMBDA(int f) {
            int k = lp.tri_face_radial_layer[f];
            if (k == 0 || k == N_r) B_f[f] = Scalar(0);
          });
        },
        E, B);
  }

  // -----------------------------------------------------------------------
  // Inner rotating-conductor / Deutsch boundary (mirrors
  // dec_field_solver::apply_inner_bc, including the E/B time staggering
  // and the delta-formulation B0 subtraction).  Quadratures reach the
  // replicated global mesh geometry through the l2g maps.  Only ranks
  // owning boundary-flagged elements do work.
  // -----------------------------------------------------------------------
  void apply_inner_bc(buffer<Scalar>& E, buffer<Scalar>& B,
                      buffer<Scalar>& B0, const dec_inner_bc_params& par,
                      double time_E, double time_B) {
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    auto mp = m_mesh->get_ptrs(typename ExecPolicy::exec_tag{});
    Scalar Bp_val = par.Bp;
    Scalar Omega_val = par.Omega;
    Scalar obliq = par.obliquity;
    bool deutsch = par.use_deutsch;

    // Lab-frame multipole snapshots at the two evaluation times.  For the
    // default (centered-dipole) extras this computes exactly the legacy
    // rotated moment and stellar_B_impl is bitwise dipole_B_impl.
    stellar_moments mom_B =
        stellar_moments_at(Bp_val, obliq, par.stellar, Omega_val * time_B);
    stellar_moments mom_E =
        stellar_moments_at(Bp_val, obliq, par.stellar, Omega_val * time_E);
    Scalar t_bc_B = static_cast<Scalar>(time_B);
    Scalar t_bc_E = static_cast<Scalar>(time_E);

    if (par.overwrite_b) {
      ExecPolicy::launch(
          [lp, mp, mom_i = mom_B, Bp_val, Omega_val, obliq,
           deutsch, t_bc = t_bc_B, bs = m_b_split]
          LAMBDA(auto B_f, auto B0_f) {
            ExecPolicy::loop(0, lp.n_owned_tri, [&] LAMBDA(int l) {
              if (lp.tri_face_boundary[l] != 1) return;
              gidx_t g = lp.tri_face_l2g[l];  // global tri-face index
              gidx_t vi0, vi1, vi2;
              tri_face_vertex_ids(mp, g, vi0, vi1, vi2);
              Scalar r0, a0x, a0y, a0z, r1, a1x, a1y, a1z, r2, a2x, a2y, a2z;
              vertex_unit(mp, vi0, r0, a0x, a0y, a0z);
              vertex_unit(mp, vi1, r1, a1x, a1y, a1z);
              vertex_unit(mp, vi2, r2, a2x, a2y, a2z);
              Scalar r_face = r0;

              Scalar flux = gauss_quad([&](double u) -> double {
                return gauss_quad([&](double t) -> double {
                  Scalar x, y, z, nx, ny, nz;
                  tri_sphere_sample(r_face, a0x, a0y, a0z, a1x, a1y, a1z,
                                    a2x, a2y, a2z, u, t, x, y, z, nx, ny, nz);
                  Scalar bx, by, bz;
                  if (deutsch) {
                    deutsch_B_impl(x, y, z, t_bc, Bp_val, Omega_val, obliq,
                                   bx, by, bz);
                  } else {
                    stellar_B_impl(x, y, z, mom_i, bx, by, bz);
                  }
                  return bx*nx + by*ny + bz*nz;
                }, 0.0, 1.0);
              }, 0.0, 1.0);
              B_f[l] = static_cast<Scalar>(flux) - B0_f[l];
            });
            ExecPolicy::loop(0, lp.n_owned_rect, [&] LAMBDA(int l) {
              if (lp.rect_face_boundary[l] != 1) return;
              gidx_t g = lp.rect_face_l2g[l];  // global rect-face index
              gidx_t vi0, vi1, vi3;
              rect_face_vertex_ids(mp, g, vi0, vi1, vi3);
              Scalar r_lo, uax, uay, uaz;
              Scalar r_tmp, ubx, uby, ubz;
              Scalar r_hi, uax3, uay3, uaz3;
              vertex_unit(mp, vi0, r_lo, uax, uay, uaz);
              vertex_unit(mp, vi1, r_tmp, ubx, uby, ubz);
              vertex_unit(mp, vi3, r_hi, uax3, uay3, uaz3);
              (void)r_tmp; (void)uax3; (void)uay3; (void)uaz3;

              Scalar flux = gauss_quad([&](double u) -> double {
                return gauss_quad([&](double v) -> double {
                  Scalar x, y, z, nx, ny, nz;
                  rect_sphere_sample(r_lo, r_hi, uax, uay, uaz, ubx, uby, ubz,
                                     u, v, x, y, z, nx, ny, nz);
                  Scalar bbx, bby, bbz;
                  if (deutsch) {
                    deutsch_B_impl(x, y, z, t_bc, Bp_val, Omega_val, obliq,
                                   bbx, bby, bbz);
                  } else {
                    stellar_B_impl(x, y, z, mom_i, bbx, bby, bbz);
                  }
                  return bbx*nx + bby*ny + bbz*nz;
                }, 0.0, 1.0);
              }, 0.0, 1.0);
              B_f[bs + l] = static_cast<Scalar>(flux) - B0_f[bs + l];
            });
          },
          B, B0);
    }

    Scalar wlt0 = par.omega_lt0;
    Scalar wlt_rs = par.lt_r_star;
    Scalar wlt_p = par.lt_p;
    Scalar lapse_c = par.lapse_compactness;
    ExecPolicy::launch(
        [lp, mp, mom_i = mom_E, Bp_val, Omega_val, obliq,
         deutsch, t_bc = t_bc_E, es = m_e_split, wlt0, wlt_rs, wlt_p, lapse_c,
         N_h_edges = (m_mesh->m_N_r + 1) * m_mesh->m_N_edge_s]
        LAMBDA(auto E_e) {
          // Corotation E = -(v × B) with v = ((Ω - ω_LT(r))/α(r)) ẑ × r,
          // shared by both edge kinds.  ω_LT = 0 and α = 1 in flat
          // spacetime; with frame dragging the star's EMF is set by its
          // rotation relative to the local dragged frame (Muslimov &
          // Tsygan 1992), and the 1/α converts that COORDINATE rate into
          // the FIDO velocity the ideal-MHD condition needs — see the
          // derivation on dec_inner_bc_params.  The Deutsch BC is
          // flat-vacuum analytic and incompatible with ω_LT != 0 — the
          // solver init aborts on that combination.
          auto corot_E = [&] LAMBDA(Scalar x, Scalar y, Scalar z,
                                    Scalar& ex, Scalar& ey, Scalar& ez) {
            if (deutsch) {
              deutsch_E_impl(x, y, z, t_bc, Bp_val, Omega_val, obliq,
                             ex, ey, ez);
            } else {
              Scalar bx, by, bz;
              stellar_B_impl(x, y, z, mom_i, bx, by, bz);
              Scalar om = Omega_val;
              if (wlt0 != Scalar(0) || lapse_c > Scalar(0)) {
                Scalar r = std::sqrt(x * x + y * y + z * z);
                om -= frame_drag_omega(r, wlt0, wlt_rs, wlt_p);
                om /= gr_lapse(r, lapse_c, wlt_rs);
              }
              Scalar vx = -om * y, vy = om * x;
              ex = -(vy * bz);
              ey = -(-vx * bz);
              ez = -(vx * by - vy * bx);
            }
          };
          ExecPolicy::loop(0, lp.n_owned_he, [&] LAMBDA(int l) {
            if (lp.h_edge_boundary[l] != 1) return;
            gidx_t g = lp.h_edge_l2g[l];  // global h-edge index
            gidx_t v0, v1;
            h_edge_vertex_ids(mp, g, v0, v1);
            Scalar r0, a0x, a0y, a0z, r1, a1x, a1y, a1z;
            vertex_unit(mp, v0, r0, a0x, a0y, a0z);
            vertex_unit(mp, v1, r1, a1x, a1y, a1z);
            Scalar circ = gauss_quad([&](double t) -> double {
              Scalar x, y, z, dlx, dly, dlz;
              h_edge_sphere_sample(r0, a0x, a0y, a0z, a1x, a1y, a1z,
                                   static_cast<Scalar>(t),
                                   x, y, z, dlx, dly, dlz);
              Scalar ex, ey, ez;
              corot_E(x, y, z, ex, ey, ez);
              return ex*dlx + ey*dly + ez*dlz;
            }, 0.0, 1.0);
            E_e[l] = static_cast<Scalar>(circ);
          });
          ExecPolicy::loop(0, lp.n_owned_ve, [&] LAMBDA(int l) {
            if (lp.v_edge_boundary[l] != 1) return;
            gidx_t g = lp.v_edge_l2g[l];  // global v-edge index
            gidx_t v0, v1;
            v_edge_vertex_ids(mp, g, v0, v1);
            Scalar r0, a0x, a0y, a0z, r1, a1x, a1y, a1z;
            vertex_unit(mp, v0, r0, a0x, a0y, a0z);
            vertex_unit(mp, v1, r1, a1x, a1y, a1z);
            (void)a1x; (void)a1y; (void)a1z;
            Scalar circ = gauss_quad([&](double t) -> double {
              Scalar rt = (Scalar(1) - static_cast<Scalar>(t)) * r0 +
                          static_cast<Scalar>(t) * r1;
              Scalar dr = r1 - r0;
              Scalar x = rt * a0x, y = rt * a0y, z = rt * a0z;
              Scalar dlx = dr * a0x, dly = dr * a0y, dlz = dr * a0z;
              Scalar ex, ey, ez;
              corot_E(x, y, z, ex, ey, ez);
              return ex*dlx + ey*dly + ez*dlz;
            }, 0.0, 1.0);
            E_e[es + l] = static_cast<Scalar>(circ);
          });
        },
        E);
  }

  // -----------------------------------------------------------------------
  // Initial conditions (owned cells only; ghosts get their values from
  // the first sync-point exchange, or are simply consistent because
  // every rank evaluates the same analytic fields).
  // -----------------------------------------------------------------------

  // Exact point-dipole face fluxes of moment (mx, my, mz) via Gauss
  // quadrature (ICs and the static background).  Thin wrapper over
  // fill_stellar_B: a centered dipole moment takes the identical
  // arithmetic path (see the stellar_extras note in
  // dec_solver_geometry.hpp), so this stays bit-compatible with the
  // pre-multipole implementation.
  void fill_dipole_B(buffer<Scalar>& B, Scalar mx_v, Scalar my_v,
                     Scalar mz_v) {
    stellar_moments mom;
    mom.mx = mx_v;
    mom.my = my_v;
    mom.mz = mz_v;
    fill_stellar_B(B, mom);
  }

  // Face fluxes of the full stellar multipole snapshot (shifted dipole +
  // optional shifted quadrupole) via the same Gauss quadrature.
  void fill_stellar_B(buffer<Scalar>& B, const stellar_moments& mom) {
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    auto mp = m_mesh->get_ptrs(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, mp, mom, bs = m_b_split] LAMBDA(auto B_f) {
          ExecPolicy::loop(0, lp.n_owned_tri, [&] LAMBDA(int l) {
            gidx_t g = lp.tri_face_l2g[l];
            gidx_t vi0, vi1, vi2;
            tri_face_vertex_ids(mp, g, vi0, vi1, vi2);
            Scalar r_face, a0x, a0y, a0z, r1_, a1x, a1y, a1z, r2_, a2x, a2y, a2z;
            vertex_unit(mp, vi0, r_face, a0x, a0y, a0z);
            vertex_unit(mp, vi1, r1_, a1x, a1y, a1z);
            vertex_unit(mp, vi2, r2_, a2x, a2y, a2z);
            (void)r1_; (void)r2_;
            double flux = gauss_quad([&](double u) -> double {
              return gauss_quad([&](double t) -> double {
                Scalar x, y, z, nx, ny, nz;
                tri_sphere_sample(r_face, a0x, a0y, a0z, a1x, a1y, a1z,
                                  a2x, a2y, a2z,
                                  static_cast<Scalar>(u), static_cast<Scalar>(t),
                                  x, y, z, nx, ny, nz);
                Scalar bx, by, bz;
                stellar_B_impl(x, y, z, mom, bx, by, bz);
                return bx*nx + by*ny + bz*nz;
              }, 0.0, 1.0);
            }, 0.0, 1.0);
            B_f[l] = static_cast<Scalar>(flux);
          });
          ExecPolicy::loop(0, lp.n_owned_rect, [&] LAMBDA(int l) {
            gidx_t g = lp.rect_face_l2g[l];
            gidx_t vi0, vi1, vi3;
            rect_face_vertex_ids(mp, g, vi0, vi1, vi3);
            Scalar r_lo, uax, uay, uaz, r_tmp, ubx, uby, ubz, r_hi, ux3, uy3, uz3;
            vertex_unit(mp, vi0, r_lo, uax, uay, uaz);
            vertex_unit(mp, vi1, r_tmp, ubx, uby, ubz);
            vertex_unit(mp, vi3, r_hi, ux3, uy3, uz3);
            (void)r_tmp; (void)ux3; (void)uy3; (void)uz3;
            double flux = gauss_quad([&](double u) -> double {
              return gauss_quad([&](double v) -> double {
                Scalar x, y, z, nx, ny, nz;
                rect_sphere_sample(r_lo, r_hi, uax, uay, uaz, ubx, uby, ubz,
                                   static_cast<Scalar>(u), static_cast<Scalar>(v),
                                   x, y, z, nx, ny, nz);
                Scalar bx, by, bz;
                stellar_B_impl(x, y, z, mom, bx, by, bz);
                return bx*nx + by*ny + bz*nz;
              }, 0.0, 1.0);
            }, 0.0, 1.0);
            B_f[bs + l] = static_cast<Scalar>(flux);
          });
        },
        B);
    ExecPolicy::sync();
  }

  // Full Deutsch retarded IC: B face fluxes at time t_B, E edge
  // circulations at time t_E (leapfrog staggering handled by the
  // caller; see dec_field_solver::set_initial_deutsch).
  void set_initial_deutsch(buffer<Scalar>& E, buffer<Scalar>& B,
                           Scalar Bp, Scalar Omega, Scalar obliquity,
                           Scalar t_E, Scalar t_B) {
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    auto mp = m_mesh->get_ptrs(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, mp, Bp, Omega, obliquity, t_B, bs = m_b_split] LAMBDA(auto B_f) {
          ExecPolicy::loop(0, lp.n_owned_tri, [&] LAMBDA(int l) {
            gidx_t g = lp.tri_face_l2g[l];
            gidx_t vi0, vi1, vi2;
            tri_face_vertex_ids(mp, g, vi0, vi1, vi2);
            Scalar r_face, a0x, a0y, a0z, r1_, a1x, a1y, a1z, r2_, a2x, a2y, a2z;
            vertex_unit(mp, vi0, r_face, a0x, a0y, a0z);
            vertex_unit(mp, vi1, r1_, a1x, a1y, a1z);
            vertex_unit(mp, vi2, r2_, a2x, a2y, a2z);
            (void)r1_; (void)r2_;
            double flux = gauss_quad([&](double u) -> double {
              return gauss_quad([&](double t) -> double {
                Scalar x, y, z, nx, ny, nz;
                tri_sphere_sample(r_face, a0x, a0y, a0z, a1x, a1y, a1z,
                                  a2x, a2y, a2z,
                                  static_cast<Scalar>(u), static_cast<Scalar>(t),
                                  x, y, z, nx, ny, nz);
                Scalar bx, by, bz;
                deutsch_B_impl(x, y, z, t_B, Bp, Omega, obliquity, bx, by, bz);
                return bx*nx + by*ny + bz*nz;
              }, 0.0, 1.0);
            }, 0.0, 1.0);
            B_f[l] = static_cast<Scalar>(flux);
          });
          ExecPolicy::loop(0, lp.n_owned_rect, [&] LAMBDA(int l) {
            gidx_t g = lp.rect_face_l2g[l];
            gidx_t vi0, vi1, vi3;
            rect_face_vertex_ids(mp, g, vi0, vi1, vi3);
            Scalar r_lo, uax, uay, uaz, r_tmp, ubx, uby, ubz, r_hi, ux3, uy3, uz3;
            vertex_unit(mp, vi0, r_lo, uax, uay, uaz);
            vertex_unit(mp, vi1, r_tmp, ubx, uby, ubz);
            vertex_unit(mp, vi3, r_hi, ux3, uy3, uz3);
            (void)r_tmp; (void)ux3; (void)uy3; (void)uz3;
            double flux = gauss_quad([&](double u) -> double {
              return gauss_quad([&](double v) -> double {
                Scalar x, y, z, nx, ny, nz;
                rect_sphere_sample(r_lo, r_hi, uax, uay, uaz, ubx, uby, ubz,
                                   static_cast<Scalar>(u), static_cast<Scalar>(v),
                                   x, y, z, nx, ny, nz);
                Scalar bx, by, bz;
                deutsch_B_impl(x, y, z, t_B, Bp, Omega, obliquity, bx, by, bz);
                return bx*nx + by*ny + bz*nz;
              }, 0.0, 1.0);
            }, 0.0, 1.0);
            B_f[bs + l] = static_cast<Scalar>(flux);
          });
        },
        B);
    ExecPolicy::launch(
        [lp, mp, Bp, Omega, obliquity, t_E, es = m_e_split,
         N_h_edges = (m_mesh->m_N_r + 1) * m_mesh->m_N_edge_s]
        LAMBDA(auto E_e) {
          ExecPolicy::loop(0, lp.n_owned_he, [&] LAMBDA(int l) {
            gidx_t g = lp.h_edge_l2g[l];
            gidx_t v0, v1;
            h_edge_vertex_ids(mp, g, v0, v1);
            Scalar r0, a0x, a0y, a0z, r1, a1x, a1y, a1z;
            vertex_unit(mp, v0, r0, a0x, a0y, a0z);
            vertex_unit(mp, v1, r1, a1x, a1y, a1z);
            double circ = gauss_quad([&](double t) -> double {
              Scalar x, y, z, dlx, dly, dlz;
              h_edge_sphere_sample(r0, a0x, a0y, a0z, a1x, a1y, a1z,
                                   static_cast<Scalar>(t),
                                   x, y, z, dlx, dly, dlz);
              Scalar ex, ey, ez;
              deutsch_E_impl(x, y, z, t_E, Bp, Omega, obliquity, ex, ey, ez);
              return ex*dlx + ey*dly + ez*dlz;
            }, 0.0, 1.0);
            E_e[l] = static_cast<Scalar>(circ);
          });
          ExecPolicy::loop(0, lp.n_owned_ve, [&] LAMBDA(int l) {
            gidx_t g = lp.v_edge_l2g[l];  // global v-edge index
            gidx_t v0, v1;
            v_edge_vertex_ids(mp, g, v0, v1);
            Scalar r0, a0x, a0y, a0z, r1, a1x, a1y, a1z;
            vertex_unit(mp, v0, r0, a0x, a0y, a0z);
            vertex_unit(mp, v1, r1, a1x, a1y, a1z);
            (void)a1x; (void)a1y; (void)a1z;
            double circ = gauss_quad([&](double t) -> double {
              Scalar rt = (Scalar(1) - static_cast<Scalar>(t)) * r0 +
                          static_cast<Scalar>(t) * r1;
              Scalar dr = r1 - r0;
              Scalar x = rt * a0x, y = rt * a0y, z = rt * a0z;
              Scalar dlx = dr * a0x, dly = dr * a0y, dlz = dr * a0z;
              Scalar ex, ey, ez;
              deutsch_E_impl(x, y, z, t_E, Bp, Omega, obliquity, ex, ey, ez);
              return ex*dlx + ey*dly + ez*dlz;
            }, 0.0, 1.0);
            E_e[es + l] = static_cast<Scalar>(circ);
          });
        },
        E);
    ExecPolicy::sync();
  }

  // out -= sub, on all local slots of a face buffer (delta formulation:
  // subtract the static background cochain after an IC fill).
  void subtract_face(buffer<Scalar>& out, buffer<Scalar>& sub) {
    ExecPolicy::launch(
        [n = m_n_faces_local] LAMBDA(auto o, auto s) {
          ExecPolicy::loop(0, n, [&] LAMBDA(int f) { o[f] -= s[f]; });
        },
        out, sub);
  }

  // -----------------------------------------------------------------------
  // Host-side layout copies between global-indexed cochains (combined
  // [h|v] edge / [tri|rect] face arrays of global size) and this rank's
  // local combined buffers.  Build-time / test utilities.
  // -----------------------------------------------------------------------
  void edge_from_global(const Scalar* Eg, buffer<Scalar>& El) const {
    auto const& L_he = m_mp->layout(cochain_type::h_edge);
    auto const& L_ve = m_mp->layout(cochain_type::v_edge);
    const gidx_t N_h = gidx_t(m_mesh->m_N_r + 1) * m_mesh->m_N_edge_s;
    for (int l = 0; l < L_he.local_size(); ++l)
      El[l] = Eg[L_he.to_global(l)];
    for (int l = 0; l < L_ve.local_size(); ++l)
      El[m_e_split + l] = Eg[N_h + L_ve.to_global(l)];
  }
  void face_from_global(const Scalar* Bg, buffer<Scalar>& Bl) const {
    auto const& L_tri = m_mp->layout(cochain_type::tri_face);
    auto const& L_rect = m_mp->layout(cochain_type::rect_face);
    const gidx_t N_t = gidx_t(m_mesh->m_N_r + 1) * m_mesh->m_N_tri;
    for (int l = 0; l < L_tri.local_size(); ++l)
      Bl[l] = Bg[L_tri.to_global(l)];
    for (int l = 0; l < L_rect.local_size(); ++l)
      Bl[m_b_split + l] = Bg[N_t + L_rect.to_global(l)];
  }
  // Scatter OWNED local values into a global-indexed array (ghosts are
  // skipped — the owning rank writes them).
  void edge_owned_to_global(const buffer<Scalar>& El, Scalar* Eg) const {
    auto const& L_he = m_mp->layout(cochain_type::h_edge);
    auto const& L_ve = m_mp->layout(cochain_type::v_edge);
    const gidx_t N_h = gidx_t(m_mesh->m_N_r + 1) * m_mesh->m_N_edge_s;
    for (int l = 0; l < L_he.owned_size(); ++l)
      Eg[L_he.to_global(l)] = El[l];
    for (int l = 0; l < L_ve.owned_size(); ++l)
      Eg[N_h + L_ve.to_global(l)] = El[m_e_split + l];
  }
  void face_owned_to_global(const buffer<Scalar>& Bl, Scalar* Bg) const {
    auto const& L_tri = m_mp->layout(cochain_type::tri_face);
    auto const& L_rect = m_mp->layout(cochain_type::rect_face);
    const gidx_t N_t = gidx_t(m_mesh->m_N_r + 1) * m_mesh->m_N_tri;
    for (int l = 0; l < L_tri.owned_size(); ++l)
      Bg[L_tri.to_global(l)] = Bl[l];
    for (int l = 0; l < L_rect.owned_size(); ++l)
      Bg[N_t + L_rect.to_global(l)] = Bl[m_b_split + l];
  }

 private:
  const prismatic_mesh* m_mesh = nullptr;
  const prismatic_mesh_partition* m_mp = nullptr;
  prismatic_mesh_local m_ml;
  prismatic_d1_local m_d1;
  prismatic_mesh_local_ptrs m_lp_host;
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  prismatic_mesh_local_ptrs m_lp_dev;
#endif

  int m_e_split = 0, m_n_edges_local = 0;
  int m_b_split = 0, m_n_faces_local = 0;

  // Frame-drag EMF weights, CSR-value arrays aligned with the d1t_*
  // sparsity blocks (see build_frame_drag).
  buffer<Scalar> m_fd_h_tri_val, m_fd_h_rect_val, m_fd_v_rect_val;
  bool m_fd_built = false;
  // Face-major transpose of W with 1/h1inv folded in (see
  // build_frame_drag_transpose): rows are LOCAL faces, columns are this
  // rank's OWNED edges addressed in E-buffer layout.
  buffer<int> m_fdt_tri_row, m_fdt_tri_col, m_fdt_rect_row, m_fdt_rect_col;
  buffer<Scalar> m_fdt_tri_val, m_fdt_rect_val;
  bool m_fdt_built = false;
  // Per-element lapse (see build_lapse).  Edge layout matches E
  // ([0,e_split) h, [e_split,..) v); face layout matches B.  Allocated
  // only when the GR path is on, so flat runs pay nothing.
  buffer<Scalar> m_alpha_e, m_alpha_f;
  bool m_lapse_built = false;
};

}  // namespace Aperture
