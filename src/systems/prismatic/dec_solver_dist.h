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
};

template <typename ExecPolicy>
class dec_solver_dist {
 public:
  void build(const prismatic_mesh& mesh, const prismatic_mesh_partition& mp) {
    m_mesh = &mesh;
    m_mp = &mp;
    m_ml = prismatic_mesh_local::build(mesh, mp, ExecPolicy::data_mem_type());
    m_d1 = prismatic_d1_local::build(mesh, mp);

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
              int g = lp.tri_face_l2g[l];  // global tri-face index
              int vi0 = mp.tri_face_v0[g];
              int vi1 = mp.tri_face_v1[g];
              int vi2 = mp.tri_face_v2[g];
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
              int g = lp.rect_face_l2g[l];  // global rect-face index
              int vi0 = mp.rect_face_v0[g];
              int vi1 = mp.rect_face_v1[g];
              int vi3 = mp.rect_face_v3[g];
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

    ExecPolicy::launch(
        [lp, mp, mx_i = mx_E, my_i = my_E, mz_i, Bp_val, Omega_val, obliq,
         deutsch, t_bc = t_bc_E, es = m_e_split,
         N_h_edges = (m_mesh->m_N_r + 1) * m_mesh->m_N_edge_s]
        LAMBDA(auto E_e) {
          // Corotation E = -(v × B), v = Ω × r, shared by both edge kinds.
          auto corot_E = [&] LAMBDA(Scalar x, Scalar y, Scalar z,
                                    Scalar& ex, Scalar& ey, Scalar& ez) {
            if (deutsch) {
              deutsch_E_impl(x, y, z, t_bc, Bp_val, Omega_val, obliq,
                             ex, ey, ez);
            } else {
              Scalar bx, by, bz;
              dipole_B_impl(x, y, z, mx_i, my_i, mz_i, bx, by, bz);
              Scalar vx = -Omega_val * y, vy = Omega_val * x;
              ex = -(vy * bz);
              ey = -(-vx * bz);
              ez = -(vx * by - vy * bx);
            }
          };
          ExecPolicy::loop(0, lp.n_owned_he, [&] LAMBDA(int l) {
            if (lp.h_edge_boundary[l] != 1) return;
            int g = lp.h_edge_l2g[l];  // global h-edge == global edge idx
            int v0 = mp.edge_v0[g], v1 = mp.edge_v1[g];
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
            int g = N_h_edges + lp.v_edge_l2g[l];  // global edge index
            int v0 = mp.edge_v0[g], v1 = mp.edge_v1[g];
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
  // Host-side layout copies between global-indexed cochains (combined
  // [h|v] edge / [tri|rect] face arrays of global size) and this rank's
  // local combined buffers.  Build-time / test utilities.
  // -----------------------------------------------------------------------
  void edge_from_global(const Scalar* Eg, buffer<Scalar>& El) const {
    auto const& L_he = m_mp->layout(cochain_type::h_edge);
    auto const& L_ve = m_mp->layout(cochain_type::v_edge);
    const int N_h = (m_mesh->m_N_r + 1) * m_mesh->m_N_edge_s;
    for (int l = 0; l < L_he.local_size(); ++l)
      El[l] = Eg[L_he.to_global(l)];
    for (int l = 0; l < L_ve.local_size(); ++l)
      El[m_e_split + l] = Eg[N_h + L_ve.to_global(l)];
  }
  void face_from_global(const Scalar* Bg, buffer<Scalar>& Bl) const {
    auto const& L_tri = m_mp->layout(cochain_type::tri_face);
    auto const& L_rect = m_mp->layout(cochain_type::rect_face);
    const int N_t = (m_mesh->m_N_r + 1) * m_mesh->m_N_tri;
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
    const int N_h = (m_mesh->m_N_r + 1) * m_mesh->m_N_edge_s;
    for (int l = 0; l < L_he.owned_size(); ++l)
      Eg[L_he.to_global(l)] = El[l];
    for (int l = 0; l < L_ve.owned_size(); ++l)
      Eg[N_h + L_ve.to_global(l)] = El[m_e_split + l];
  }
  void face_owned_to_global(const buffer<Scalar>& Bl, Scalar* Bg) const {
    auto const& L_tri = m_mp->layout(cochain_type::tri_face);
    auto const& L_rect = m_mp->layout(cochain_type::rect_face);
    const int N_t = (m_mesh->m_N_r + 1) * m_mesh->m_N_tri;
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
};

}  // namespace Aperture
