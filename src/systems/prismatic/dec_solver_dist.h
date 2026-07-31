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
  bool use_deutsch = false;
  bool overwrite_b = true;
  // Frame dragging ("fake GR"): the corotation EMF is set by the star's
  // rotation RELATIVE TO the local dragged frame,
  // omega_eff(r) = Omega - omega_lt(r) with
  // omega_lt(r) = omega_lt0 * (lt_r_star / r)^lt_p about the SPIN axis
  // (z).  omega_lt0 = 0 recovers flat spacetime exactly.
  Scalar omega_lt0 = 0.0;
  Scalar lt_r_star = 1.0;
  int lt_p = 3;
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

    auto vlt = [&](double x, double y, double z, double v[3]) {
      Scalar vx, vy, vz;
      frame_drag_velocity(Scalar(x), Scalar(y), Scalar(z), omega0, r_star,
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
  }

  // Eeff[e] = E[e] + W_e(Bdelta + B0) on OWNED edges (caller exchanges
  // Eeff ghosts, then feeds it to faraday()/compute_rhs() in place of E).
  // Requires fresh Bdelta ghosts; B0 is static (exchanged once at init).
  void frame_drag_eff_E(buffer<Scalar>& E, buffer<Scalar>& Bdelta,
                        buffer<Scalar>& B0, buffer<Scalar>& Eeff) {
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, es = m_e_split, bs = m_b_split]
        LAMBDA(auto E_e, auto Bd, auto B0_f, auto Ef, auto wht, auto whr,
               auto wvr) {
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
            Ef[e] = E_e[e] + W;
          });
          ExecPolicy::loop(0, lp.n_owned_ve, [&] LAMBDA(int e) {
            Scalar W = Scalar(0);
            for (int j = lp.d1t_v_rect_row[e]; j < lp.d1t_v_rect_row[e + 1];
                 j++) {
              int f = lp.d1t_v_rect_col[j];
              W += wvr[j] * (Bd[bs + f] + B0_f[bs + f]);
            }
            Ef[es + e] = E_e[es + e] + W;
          });
        },
        E, Bdelta, B0, Eeff, m_fd_h_tri_val, m_fd_h_rect_val,
        m_fd_v_rect_val);
  }

  bool frame_drag_built() const { return m_fd_built; }

  // -----------------------------------------------------------------------
  // Ampere half-step: E[e] += dt * h1inv[e] * ((d1^T h2 B)[e] - J[e]) on
  // owned edges.  Requires fresh B ghosts (tri + rect).
  // -----------------------------------------------------------------------
  void ampere(buffer<Scalar>& E, buffer<Scalar>& B, buffer<Scalar>& J,
              double dt) {
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, dt, es = m_e_split, bs = m_b_split]
        LAMBDA(auto E_e, auto B_f, auto J_e) {
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
            E_e[e] += dt * lp.h_edge_hodge1_inv[e] * (curl_H - J_e[e]);
          });
          ExecPolicy::loop(0, lp.n_owned_ve, [&] LAMBDA(int e) {
            Scalar curl_H = Scalar(0);
            for (int j = lp.d1t_v_rect_row[e]; j < lp.d1t_v_rect_row[e + 1]; j++) {
              int f = lp.d1t_v_rect_col[j];
              curl_H +=
                  lp.d1t_v_rect_val[j] * lp.rect_face_hodge2[f] * B_f[bs + f];
            }
            E_e[es + e] +=
                dt * lp.v_edge_hodge1_inv[e] * (curl_H - J_e[es + e]);
          });
        },
        E, B, J);
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

    Scalar mx_B = Bp_val * std::sin(obliq) * std::cos(Omega_val * time_B);
    Scalar my_B = Bp_val * std::sin(obliq) * std::sin(Omega_val * time_B);
    Scalar mx_E = Bp_val * std::sin(obliq) * std::cos(Omega_val * time_E);
    Scalar my_E = Bp_val * std::sin(obliq) * std::sin(Omega_val * time_E);
    Scalar mz_i = Bp_val * std::cos(obliq);
    Scalar t_bc_B = static_cast<Scalar>(time_B);
    Scalar t_bc_E = static_cast<Scalar>(time_E);

    if (par.overwrite_b) {
      ExecPolicy::launch(
          [lp, mp, mx_i = mx_B, my_i = my_B, mz_i, Bp_val, Omega_val, obliq,
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
                    dipole_B_impl(x, y, z, mx_i, my_i, mz_i, bx, by, bz);
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
                    dipole_B_impl(x, y, z, mx_i, my_i, mz_i, bbx, bby, bbz);
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
    ExecPolicy::launch(
        [lp, mp, mx_i = mx_E, my_i = my_E, mz_i, Bp_val, Omega_val, obliq,
         deutsch, t_bc = t_bc_E, es = m_e_split, wlt0, wlt_rs, wlt_p,
         N_h_edges = (m_mesh->m_N_r + 1) * m_mesh->m_N_edge_s]
        LAMBDA(auto E_e) {
          // Corotation E = -(v × B) with v = (Ω - ω_LT(r)) × r, shared by
          // both edge kinds.  ω_LT = 0 in flat spacetime; with frame
          // dragging the star's EMF is set by its rotation relative to
          // the local dragged frame (Muslimov & Tsygan 1992).  The
          // Deutsch BC is flat-vacuum analytic and incompatible with
          // ω_LT != 0 — the solver init aborts on that combination.
          auto corot_E = [&] LAMBDA(Scalar x, Scalar y, Scalar z,
                                    Scalar& ex, Scalar& ey, Scalar& ez) {
            if (deutsch) {
              deutsch_E_impl(x, y, z, t_bc, Bp_val, Omega_val, obliq,
                             ex, ey, ez);
            } else {
              Scalar bx, by, bz;
              dipole_B_impl(x, y, z, mx_i, my_i, mz_i, bx, by, bz);
              Scalar om = Omega_val;
              if (wlt0 != Scalar(0)) {
                Scalar r = std::sqrt(x * x + y * y + z * z);
                om -= frame_drag_omega(r, wlt0, wlt_rs, wlt_p);
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
  // quadrature (ICs and the static background).
  void fill_dipole_B(buffer<Scalar>& B, Scalar mx_v, Scalar my_v,
                     Scalar mz_v) {
    auto lp = get_lp(typename ExecPolicy::exec_tag{});
    auto mp = m_mesh->get_ptrs(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [lp, mp, mx_v, my_v, mz_v, bs = m_b_split] LAMBDA(auto B_f) {
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
                dipole_B_impl(x, y, z, mx_v, my_v, mz_v, bx, by, bz);
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
                dipole_B_impl(x, y, z, mx_v, my_v, mz_v, bx, by, bz);
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
};

}  // namespace Aperture
