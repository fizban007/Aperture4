/*
 * Copyright (c) 2020 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "core/cuda_control.h"
#include "core/multi_array_exp.hpp"
#include "core/ndsubset_dev.hpp"
#include "field_solver_gr_ks.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "utils/kernel_helper.hpp"
#include "utils/timer.h"
#include <cusparse.h>

namespace Aperture {

namespace {

cusparseHandle_t sp_handle;

template <typename Conf>
void
axis_boundary_e(vector_field<Conf>& D, const grid_ks_t<Conf>& grid) {
  typedef typename Conf::idx_t idx_t;
  kernel_launch(
      [] __device__(auto D) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        for (auto n0 : grid_stride_range(0, grid.dims[0])) {
          auto n1_0 = grid.guard[1];
          auto n1_pi = grid.dims[1] - grid.guard[1];
          if (abs(grid_ks_t<Conf>::theta(grid.template pos<1>(n1_0, true))) <
              0.1f * grid.delta[1]) {
            // At the theta = 0 axis

            // Set E_phi and B_theta to zero
            auto idx = idx_t(index_t<2>(n0, n1_0), ext);
            D[2][idx] = 0.0f;
            // e[1][idx] = 0.0;
            D[1][idx.dec_y()] = D[1][idx];
            D[0][idx.dec_y()] = D[0][idx];
            // e[0][idx] = 0.0f;
          }
          // printf("boundary pi at %f\n", grid.template pos<1>(n1_pi, true));
          if (abs(grid_ks_t<Conf>::theta(grid.template pos<1>(n1_pi, true)) -
                  M_PI) < 0.1f * grid.delta[1]) {
            // At the theta = pi axis
            auto idx = idx_t(index_t<2>(n0, n1_pi), ext);
            D[2][idx] = 0.0f;
            // e[1][idx] = 0.0;
            D[1][idx] = D[1][idx.dec_y()];
            D[0][idx] = D[0][idx.dec_y()];
            // e[0][idx] = 0.0f;
          }
        }
      },
      D.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
axis_boundary_b(vector_field<Conf>& B, const grid_ks_t<Conf>& grid) {
  typedef typename Conf::idx_t idx_t;
  kernel_launch(
      [] __device__(auto B) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        for (auto n0 : grid_stride_range(0, grid.dims[0])) {
          for (int n1_0 = grid.guard[1]; n1_0 >= 0; n1_0--) {
            if (grid_ks_t<Conf>::theta(grid.template pos<1>(n1_0, true)) <
                0.1f * grid.delta[1]) {
              // At the theta = 0 axis

              // Set E_phi and B_theta to zero
              auto idx = idx_t(index_t<2>(n0, n1_0), ext);
              B[1][idx] = 0.0f;
              B[2][idx] = 0.0f;
              // B[2][idx.dec_y()] = B[2][idx];
              B[0][idx.dec_y()] = B[0][idx];
            }
          }
          for (int n1_pi = grid.dims[1] - grid.guard[1];
               n1_pi <= grid.dims[1] - 1; n1_pi++) {
            // printf("boundary pi at %f\n", grid.template pos<1>(n1_pi, true));
            if (abs(grid_ks_t<Conf>::theta(grid.template pos<1>(n1_pi, true)) -
                    M_PI) < 0.1f * grid.delta[1]) {
              // At the theta = pi axis
              auto idx = idx_t(index_t<2>(n0, n1_pi), ext);
              B[1][idx] = 0.0f;
              // B[1][idx.dec_y()] = 0.0f;
              // B[2][idx] = B[2][idx.dec_y()];
              B[2][idx] = 0.0f;
              B[2][idx.dec_y()] = 0.0f;
              B[0][idx] = B[0][idx.dec_y()];
            }
          }
        }
      },
      B.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
horizon_boundary(vector_field<Conf>& D, vector_field<Conf>& B,
                 const vector_field<Conf>& D0, const vector_field<Conf>& B0,
                 const grid_ks_t<Conf>& grid) {
  kernel_launch(
      [] __device__(auto D, auto D0, auto B, auto B0) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        int boundary_length = 2;
        for (auto n1 : grid_stride_range(0, grid.dims[1])) {
          auto pos_ref = index_t<2>(grid.guard[0] + boundary_length, n1);
          auto idx_ref = Conf::idx(pos_ref, ext);
          for (int n0 = 0; n0 < grid.guard[0] + boundary_length; n0++) {
            auto pos = index_t<2>(n0, n1);
            auto idx = Conf::idx(pos, ext);

            // B[0][idx] = 0.0f;
            // B[1][idx] = 0.0f;
            // B[2][idx] = 0.0f;
            // D[0][idx] = 0.0f;
            // D[1][idx] = 0.0f;
            // D[2][idx] = 0.0f;

            B[1][idx] = B[1][idx_ref];
            B[2][idx] = B[2][idx_ref];
            D[0][idx] = D[0][idx_ref];

            B[0][idx] = B[0][idx_ref];
            D[1][idx] = D[1][idx_ref];
            D[2][idx] = D[2][idx_ref];

            // B[1][idx] = B0[1][idx];
            // B[2][idx] = B0[2][idx];
            // D[0][idx] = D0[0][idx];

            // B[0][idx] = B0[0][idx];
            // D[1][idx] = D0[1][idx];
            // D[2][idx] = D0[2][idx];
          }
        }
      },
      D.get_ptrs(), D0.get_ptrs(), B.get_ptrs(), B0.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_flux(scalar_field<Conf>& flux, const vector_field<Conf>& b,
             const grid_ks_t<Conf>& grid) {
  flux.init();
  auto ext = grid.extent();
  kernel_launch(
      [ext] __device__(auto flux, auto b, auto a) {
        auto& grid = dev_grid<Conf::dim>();
        for (auto n0 : grid_stride_range(0, grid.dims[0])) {
          auto r = grid_ks_t<Conf>::radius(grid.template pos<0>(n0, true));

          for (int n1 = grid.guard[1]; n1 < grid.dims[1] - grid.guard[1];
               n1++) {
            Scalar th = grid_ks_t<Conf>::theta(grid.template pos<1>(n1, false));
            Scalar th_p =
                grid_ks_t<Conf>::theta(grid.template pos<1>(n1 + 1, true));
            Scalar th_m =
                grid_ks_t<Conf>::theta(grid.template pos<1>(n1, true));
            auto dth = th_p - th_m;

            auto pos = index_t<Conf::dim>(n0, n1);
            auto idx = typename Conf::idx_t(pos, ext);

            flux[idx] = flux[idx.dec_y()] +
                        b[0][idx] * Metric_KS::sqrt_gamma(a, r, th) * dth;
          }
        }
      },
      flux.dev_ndptr(), b.get_ptrs(), grid.a);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

}  // namespace

template <typename Conf>
field_solver_gr_ks_cu<Conf>::~field_solver_gr_ks_cu() {
  cusparseDestroy(sp_handle);

  // sp_buffer.resize(0);
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::init() {
  field_solver<Conf>::init();

  this->m_env.params().get_value("bh_spin", m_a);
  Logger::print_info("bh_spin in field solver is {}", m_a);
  this->m_env.params().get_value("implicit_beta", this->m_beta);

  cusparseCreate(&sp_handle);
  auto extl = this->m_grid.extent_less();

  m_tmp_rhs.set_memtype(MemType::device_only);
  m_tmp_rhs.resize(extl);

  m_tmp_prev_field.set_memtype(MemType::device_only);
  m_tmp_prev_field.resize(this->m_grid.extent());

  m_tri_dl.set_memtype(MemType::device_only);
  m_tri_dl.resize(extl[0]);
  m_tri_dl.assign_dev(0.0f);

  m_tri_d.set_memtype(MemType::device_only);
  m_tri_d.resize(extl[0]);
  m_tri_d.assign_dev(0.0f);

  m_tri_du.set_memtype(MemType::device_only);
  m_tri_du.resize(extl[0]);
  m_tri_du.assign_dev(0.0f);

  size_t buffer_size;
#if USE_DOUBLE
  cusparseDgtsv2_bufferSizeExt(sp_handle, extl[0], extl[1], m_tri_dl.dev_ptr(),
                               m_tri_d.dev_ptr(), m_tri_du.dev_ptr(),
                               m_tmp_rhs.dev_ptr(), extl[0], &buffer_size);
#else
  cusparseSgtsv2_bufferSizeExt(sp_handle, extl[0], extl[1], m_tri_dl.dev_ptr(),
                               m_tri_d.dev_ptr(), m_tri_du.dev_ptr(),
                               m_tmp_rhs.dev_ptr(), extl[0], &buffer_size);
#endif
  Logger::print_info("tri-diagonal buffer size is {}", buffer_size);
  sp_buffer.set_memtype(MemType::device_only);
  sp_buffer.resize(buffer_size);
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::register_data_components() {
  field_solver_cu<Conf>::register_data_components();

  flux = this->m_env.template register_data<scalar_field<Conf>>(
      "flux", this->m_grid, field_type::vert_centered);
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::solve_tridiagonal() {
  // Solve the assembled tri-diagonal system using cusparse
  cusparseStatus_t status;
  auto ext = this->m_grid.extent_less();
#if USE_DOUBLE
  status = cusparseDgtsv2(sp_handle, ext[0], ext[1], m_tri_dl.dev_ptr(),
                          m_tri_d.dev_ptr(), m_tri_du.dev_ptr(),
                          m_tmp_rhs.dev_ptr(), ext[0], sp_buffer.dev_ptr());
#else
  status = cusparseSgtsv2(sp_handle, ext[0], ext[1], m_tri_dl.dev_ptr(),
                          m_tri_d.dev_ptr(), m_tri_du.dev_ptr(),
                          m_tmp_rhs.dev_ptr(), ext[0], sp_buffer.dev_ptr());
#endif
  CudaSafeCall(cudaDeviceSynchronize());
  if (status != CUSPARSE_STATUS_SUCCESS) {
    Logger::print_err("cusparse failure during field update! Error code {}",
                      status);
  }
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::update_Bth(vector_field<Conf>& B,
                                        const vector_field<Conf>& B0,
                                        const vector_field<Conf>& D,
                                        const vector_field<Conf>& D0,
                                        value_t dt) {
  m_tmp_rhs.assign_dev(0.0f);
  m_tmp_prev_field.copy_from(B[1]);

  // First assemble the right hand side and the diagonals of the tri-diagonal
  // equation
  kernel_launch(
      [dt] __device__(auto B, auto B0, auto D, auto D0, auto rhs, auto d,
                      auto dl, auto du, auto a, auto beta) {
        using namespace Metric_KS;

        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        auto extl = grid.extent_less();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            value_t r =
                grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], false));
            value_t r_sp =
                grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0] + 1, true));
            value_t r_sm =
                grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], true));
            value_t dr = r_sp - r_sm;
            // value_t dr = grid.delta[0];

            value_t th = grid.template pos<1>(pos[1], true);
            // if (math::abs(th) < TINY) th = sgn(th) * 1.0e-4;
            if (math::abs(th) < TINY)
              // th = (th < 0.0f ? -1.0f : 1.0f) * 0.01 * grid.delta[1];
              th = 0.01f * grid.delta[1];

            value_t sth = math::sin(th);
            value_t cth = math::cos(th);
            value_t prefactor = dt / (sqrt_gamma(a, r, sth, cth) * dr);

            auto Eph1 = ag_33(a, r_sp, sth, cth) * D[2][idx.inc_x()] +
                        ag_13(a, r_sp, sth, cth) * 0.5f *
                            (D[0][idx.inc_x()] + D[0][idx]) +
                        sq_gamma_beta(a, r_sp, sth, cth) * 0.5f * (1.0 - beta) *
                            (B[1][idx.inc_x()] + B[1][idx]);

            auto Eph0 = ag_33(a, r_sm, sth, cth) * D[2][idx] +
                        ag_13(a, r_sm, sth, cth) * 0.5f *
                            (D[0][idx] + D[0][idx.dec_x()]) +
                        sq_gamma_beta(a, r_sm, sth, cth) * 0.5f * (1.0 - beta) *
                            (B[1][idx] + B[1][idx.dec_x()]);

            auto idx_rhs = Conf::idx(
                index(pos[0] - grid.guard[0], pos[1] - grid.guard[1]), extl);
            rhs[idx_rhs] = B[1][idx] - prefactor * (Eph0 - Eph1);

            // if (pos[0] == 6 && pos[1] == 200) {
            // printf(
            //     "Eph1 is %f, Eph0 is %f, dEphi is %f, B1 is %f, "
            //     "rhs is %f\n",
            //     Eph1, Eph0, (Eph0 - Eph1), B[1][idx],
            //     rhs[idx]);
            // printf("ag33 is %f, ag13 is %f, sqgb is %f\n",
            //        ag_33(a, r, sth, cth), ag_13(a, r, sth, cth),
            //        sq_gamma_beta(a, r, sth, cth));
            // }
            // if (pos[1] == 2 && pos[0] == 200)
            //   printf("rhs is %f, D0 is %f, B1 is %f\n", rhs[idx], D[0][idx],
            //   B[1][idx]);

            value_t du_coef =
                prefactor * 0.5f * beta * sq_gamma_beta(a, r_sp, sth, cth);
            value_t dl_coef =
                -prefactor * 0.5f * beta * sq_gamma_beta(a, r_sm, sth, cth);

            if (pos[0] == grid.guard[0]) {
              rhs[idx_rhs] += dl_coef * B[1][idx.dec_x()];
              // rhs[idx_rhs] += dl_coef * B[1][idx];
            } else if (pos[0] == grid.dims[0] - grid.guard[0] - 1) {
              rhs[idx_rhs] += du_coef * B[1][idx.inc_x()];
              // rhs[idx_rhs] += du_coef * B[1][idx];
            }

            d[pos[0] - grid.guard[0]] = 1.0f - (du_coef + dl_coef);

            du[pos[0] - grid.guard[0]] = -du_coef;
            dl[pos[0] - grid.guard[0]] = -dl_coef;
          }
        }
      },
      B.get_const_ptrs(), B0.get_const_ptrs(), D.get_const_ptrs(),
      D0.get_const_ptrs(), m_tmp_rhs.dev_ndptr(), m_tri_d.dev_ptr(),
      m_tri_dl.dev_ptr(), m_tri_du.dev_ptr(), m_a, this->m_beta);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  solve_tridiagonal();

  // if (this->m_comm == nullptr || this->m_comm->domain_info().is_boundary[2])
  // {
  //   kernel_launch(
  //       [] __device__(auto f) {
  //         auto& grid = dev_grid<Conf::dim>();
  //         auto ext = grid.extent();
  //         for (auto n0 : grid_stride_range(0, grid.dims[0])) {
  //           int n1 = grid.guard[1];
  //           auto idx = Conf::idx({n0, n1}, ext);

  //           f[idx] = 0.0f;
  //         }
  //       },
  //       m_tmp_rhs.dev_ndptr());
  // }

  // B[1].copy_from(m_tmp_rhs);
  // select_dev(m_tmp_prev_field) = m_tmp_rhs * 0.5f + m_tmp_prev_field * 0.5f;
  kernel_launch(
      [] __device__(auto B, auto rhs, auto prev_field, auto beta) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        auto extl = grid.extent_less();
        for (auto idx : grid_stride_range(Conf::begin(extl), Conf::end(extl))) {
          auto pos = get_pos(idx, extl);
          auto fidx = Conf::idx(
              index(pos[0] + grid.guard[0], pos[1] + grid.guard[1]), ext);

          B[1][fidx] = rhs[idx];
          prev_field[fidx] = beta * rhs[idx] + (1.0f - beta) * prev_field[fidx];
          // prev_field[fidx] = rhs[idx];
        }
      },
      B.get_ptrs(), m_tmp_rhs.dev_ndptr(), m_tmp_prev_field.dev_ndptr(), this->m_beta);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::update_Bph(vector_field<Conf>& B,
                                        const vector_field<Conf>& B0,
                                        const vector_field<Conf>& D,
                                        const vector_field<Conf>& D0,
                                        value_t dt) {
  m_tmp_rhs.assign_dev(0.0f);

  // First assemble the right hand side and the diagonals of the tri-diagonal
  // equation
  kernel_launch(
      [dt] __device__(auto B, auto B0, auto D, auto D0, auto rhs, auto d,
                      auto dl, auto du, auto a, auto beta) {
        using namespace Metric_KS;
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        auto extl = grid.extent_less();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            value_t r =
                grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], false));
            value_t r_sp =
                grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0] + 1, true));
            value_t r_sm =
                grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], true));
            value_t dr = r_sp - r_sm;
            // value_t dr = grid.delta[0];

            value_t th = grid.template pos<1>(pos[1], false);
            value_t th_sp = grid.template pos<1>(pos[1] + 1, true);
            value_t th_sm = grid.template pos<1>(pos[1], true);
            value_t dth = th_sp - th_sm;
            if (th_sm < TINY) th_sm = 0.01f * grid.delta[1];

            value_t sth = math::sin(th);
            value_t cth = math::cos(th);
            value_t prefactor = dt / (sqrt_gamma(a, r, sth, cth) * dr * dth);

            auto Er1 = ag_11(a, r, th_sp) * D[0][idx.inc_y()] +
                       ag_13(a, r, th_sp) * 0.5f *
                           (D[2][idx.inc_y()] + D[2][idx.inc_y().inc_x()]);

            auto Er0 = ag_11(a, r, th_sm) * D[0][idx] +
                       ag_13(a, r, th_sm) * 0.5f *
                           (D[2][idx] + D[2][idx.inc_x()]);

            auto Eth1 = ag_22(a, r_sp, sth, cth) * D[1][idx.inc_x()] -
                        sq_gamma_beta(a, r_sp, sth, cth) * 0.5f * (1.0f - beta) *
                            (B[2][idx.inc_x()] + B[2][idx]);

            auto Eth0 = ag_22(a, r_sm, sth, cth) * D[1][idx] -
                        sq_gamma_beta(a, r_sm, sth, cth) * 0.5f * (1.0f - beta) *
                            (B[2][idx] + B[2][idx.dec_x()]);

            auto idx_rhs = Conf::idx(
                index(pos[0] - grid.guard[0], pos[1] - grid.guard[1]), extl);
            rhs[idx_rhs] = B[2][idx] -
                           prefactor * (dr * (Er0 - Er1) + dth * (Eth1 - Eth0));

            value_t du_coef =
                prefactor * dth * 0.5f * beta * sq_gamma_beta(a, r_sp, sth, cth);
            value_t dl_coef =
                -prefactor * dth * 0.5f * beta * sq_gamma_beta(a, r_sm, sth, cth);

            if (pos[0] == grid.guard[0]) {
              rhs[idx_rhs] += dl_coef * B[2][idx.dec_x()];
              // rhs[idx_rhs] += dl_coef * B[2][idx];
            } else if (pos[0] == grid.dims[0] - grid.guard[0] - 1) {
              rhs[idx_rhs] += du_coef * B[2][idx.inc_x()];
              // rhs[idx_rhs] += du_coef * B[2][idx];
            }

            d[pos[0] - grid.guard[0]] = 1.0f - (du_coef + dl_coef);
            du[pos[0] - grid.guard[0]] = -du_coef;
            dl[pos[0] - grid.guard[0]] = -dl_coef;
          }
        }
      },
      B.get_const_ptrs(), B0.get_const_ptrs(), D.get_const_ptrs(),
      D0.get_const_ptrs(), m_tmp_rhs.dev_ndptr(), m_tri_d.dev_ptr(),
      m_tri_dl.dev_ptr(), m_tri_du.dev_ptr(), m_a, this->m_beta);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  solve_tridiagonal();

  // B[2].copy_from(m_tmp_rhs);
  auto& grid = B.grid();
  select_dev(B[2], grid.guards(), grid.extent_less()) =
      select_dev_const(m_tmp_rhs);
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::update_Br(vector_field<Conf>& B,
                                       const vector_field<Conf>& B0,
                                       const vector_field<Conf>& D,
                                       const vector_field<Conf>& D0,
                                       value_t dt) {
  kernel_launch(
      [dt] __device__(auto B, auto B0, auto D, auto D0, auto tmp_field,
                      auto a) {
        using namespace Metric_KS;
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            value_t r =
                grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], true));

            value_t th = grid.template pos<1>(pos[1], false);
            value_t th_sp = grid.template pos<1>(pos[1] + 1, true);
            value_t th_sm = grid.template pos<1>(pos[1], true);
            value_t dth = th_sp - th_sm;
            if (th_sm < TINY) th_sm = 0.01f * grid.delta[1];

            value_t prefactor = dt / (sqrt_gamma(a, r, th) * dth);

            value_t sth = math::sin(th_sp);
            value_t cth = math::cos(th_sp);
            value_t Eph1 =
                ag_33(a, r, sth, cth) * D[2][idx.inc_y()] +
                ag_13(a, r, sth, cth) * 0.5f *
                    (D[0][idx.inc_y()] + D[0][idx.inc_y().dec_x()]) +
                sq_gamma_beta(a, r, sth, cth) * 0.5f *
                    (tmp_field[idx.inc_y()] + tmp_field[idx.inc_y().dec_x()]);

            sth = math::sin(th_sm);
            cth = math::cos(th_sm);
            value_t Eph0 =
                ag_33(a, r, sth, cth) * D[2][idx] +
                ag_13(a, r, sth, cth) * 0.5f * (D[0][idx] + D[0][idx.dec_x()]) +
                sq_gamma_beta(a, r, sth, cth) * 0.5f *
                    (tmp_field[idx] + tmp_field[idx.dec_x()]);
            // if (pos[0] == 6 && pos[1] == 200) {
            //   printf(
            //       "Eph1 is %f, Eph0 is %f, dEphi is %f, tmpf is %f, "
            //       "tmpf+ is %f\n",
            //       Eph1, Eph0, Eph1 - Eph0, tmp_field[idx],
            //       tmp_field[idx.inc_y()]);
            //   printf("ag33 is %f, ag13 is %f, sqgb is %f\n",
            //          ag_33(a, r, sth, cth), ag_13(a, r, sth, cth),
            //          sq_gamma_beta(a, r, sth, cth));
            // }

            B[0][idx] = B[0][idx] - prefactor * (Eph1 - Eph0);
          }
        }
      },
      B.get_ptrs(), B0.get_const_ptrs(), D.get_const_ptrs(),
      D0.get_const_ptrs(), m_tmp_prev_field.dev_ndptr_const(), m_a);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::update_Dth(vector_field<Conf>& D,
                                        const vector_field<Conf>& D0,
                                        const vector_field<Conf>& B,
                                        const vector_field<Conf>& B0,
                                        const vector_field<Conf>& J,
                                        value_t dt) {
  m_tmp_rhs.assign_dev(0.0f);
  m_tmp_prev_field.copy_from(D[1]);

  // First assemble the right hand side and the diagonals of the tri-diagonal
  // equation
  kernel_launch(
      [dt] __device__(auto D, auto D0, auto B, auto B0, auto J, auto rhs,
                      auto d, auto dl, auto du, auto a, auto beta) {
        using namespace Metric_KS;

        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        auto extl = grid.extent_less();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            value_t r =
                grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], true));
            value_t r_sp =
                grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], false));
            value_t r_sm = grid_ks_t<Conf>::radius(
                grid.template pos<0>(pos[0] - 1, false));
            value_t dr = r_sp - r_sm;
            // value_t dr = grid.delta[0];

            value_t th = grid.template pos<1>(pos[1], false);

            value_t sth = math::sin(th);
            value_t cth = math::cos(th);
            value_t prefactor = dt / (Metric_KS::sqrt_gamma(a, r, th) * dr);

            // if (pos[0] > 0 && pos[0] < grid.dims[0] - 1) {
            auto Hph1 = ag_33(a, r_sp, sth, cth) * B[2][idx] +
                        ag_13(a, r_sp, sth, cth) * 0.5f *
                            (B[0][idx.inc_x()] + B[0][idx]) -
                        sq_gamma_beta(a, r_sp, sth, cth) * 0.5f * (1.0f - beta) *
                            (D[1][idx.inc_x()] + D[1][idx]);

            auto Hph0 = ag_33(a, r_sm, sth, cth) * B[2][idx.dec_x()] +
                        ag_13(a, r_sm, sth, cth) * 0.5f *
                            (B[0][idx] + B[0][idx.dec_x()]) -
                        sq_gamma_beta(a, r_sm, sth, cth) * 0.5f * (1.0f - beta) *
                            (D[1][idx] + D[1][idx.dec_x()]);

            auto idx_rhs = Conf::idx(
                index(pos[0] - grid.guard[0], pos[1] - grid.guard[1]), extl);
            rhs[idx_rhs] =
                D[1][idx] - dt * J[1][idx] + prefactor * (Hph0 - Hph1);
            // }
            value_t du_coef =
                prefactor * 0.5f * beta * Metric_KS::sq_gamma_beta(a, r_sp, sth, cth);
            value_t dl_coef =
                -prefactor * 0.5f * beta * Metric_KS::sq_gamma_beta(a, r_sm, sth, cth);

            if (pos[0] == grid.guard[0]) {
              rhs[idx_rhs] += dl_coef * D[1][idx.dec_x()];
              // rhs[idx_rhs] += dl_coef * D[1][idx];
            } else if (pos[0] == grid.dims[0] - grid.guard[0] - 1) {
              rhs[idx_rhs] += du_coef * D[1][idx.inc_x()];
              // rhs[idx_rhs] += du_coef * D[1][idx];
            }

            d[pos[0] - grid.guard[0]] = 1.0f - (du_coef + dl_coef);
            du[pos[0] - grid.guard[0]] = -du_coef;
            dl[pos[0] - grid.guard[0]] = -dl_coef;
          }
        }
      },
      D.get_const_ptrs(), D0.get_const_ptrs(), B.get_const_ptrs(),
      B0.get_const_ptrs(), J.get_const_ptrs(), m_tmp_rhs.dev_ndptr(),
      m_tri_d.dev_ptr(), m_tri_dl.dev_ptr(), m_tri_du.dev_ptr(), m_a, this->m_beta);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  solve_tridiagonal();

  // if (this->m_comm == nullptr || this->m_comm->domain_info().is_boundary[2])
  // {
  //   kernel_launch(
  //       [] __device__(auto f) {
  //         auto& grid = dev_grid<Conf::dim>();
  //         auto ext = grid.extent();
  //         for (auto n0 : grid_stride_range(0, grid.dims[0])) {
  //           int n1 = grid.guard[1];
  //           auto idx = Conf::idx({n0, n1}, ext);

  //           f[idx] = 0.0f;
  //         }
  //       },
  //       m_tmp_rhs.dev_ndptr());
  // }

  // D[1].copy_from(m_tmp_rhs);
  // select_dev(m_tmp_prev_field) = m_tmp_rhs * 0.5f + m_tmp_prev_field * 0.5f;
  kernel_launch(
      [] __device__(auto D, auto rhs, auto prev_field, auto beta) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        auto extl = grid.extent_less();
        for (auto idx : grid_stride_range(Conf::begin(extl), Conf::end(extl))) {
          auto pos = get_pos(idx, extl);
          auto fidx = Conf::idx(
              index(pos[0] + grid.guard[0], pos[1] + grid.guard[1]), ext);

          D[1][fidx] = rhs[idx];
          prev_field[fidx] = beta * rhs[idx] + (1.0f - beta) * prev_field[fidx];
          // prev_field[fidx] = 1.0f * rhs[idx];
        }
      },
      D.get_ptrs(), m_tmp_rhs.dev_ndptr(), m_tmp_prev_field.dev_ndptr(), this->m_beta);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::update_Dph(vector_field<Conf>& D,
                                        const vector_field<Conf>& D0,
                                        const vector_field<Conf>& B,
                                        const vector_field<Conf>& B0,
                                        const vector_field<Conf>& J,
                                        value_t dt) {
  m_tmp_rhs.assign_dev(0.0f);

  // First assemble the right hand side and the diagonals of the tri-diagonal
  // equation
  kernel_launch(
      [dt] __device__(auto D, auto B, auto J, auto rhs, auto d, auto dl,
                      auto du, auto a, auto beta) {
        using namespace Metric_KS;

        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        auto extl = grid.extent_less();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            value_t r =
                grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], true));
            value_t r_sp =
                grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], false));
            value_t r_sm = grid_ks_t<Conf>::radius(
                grid.template pos<0>(pos[0] - 1, false));
            value_t dr = r_sp - r_sm;
            // value_t dr = grid.delta[0];

            value_t th = grid.template pos<1>(pos[1], true);
            value_t th_sp = grid.template pos<1>(pos[1], false);
            value_t th_sm = grid.template pos<1>(pos[1] - 1, false);
            value_t dth = th_sp - th_sm;
            if (th < TINY) th = 0.01f * grid.delta[1];

            value_t sth = math::sin(th);
            value_t cth = math::cos(th);
            value_t prefactor =
                dt / (Metric_KS::sqrt_gamma(a, r, th) * dr * dth);

            // if (pos[1] > 0 && pos[0] > 0 && pos[0] < grid.dims[0] - 1) {
            auto Hr0 = ag_11(a, r, th_sm) * B[0][idx.dec_y()] +
                       ag_13(a, r, th_sm) * 0.5f *
                           (B[2][idx.dec_y()] + B[2][idx.dec_y().dec_x()]);

            auto Hr1 =
                ag_11(a, r, th_sp) * B[0][idx] +
                ag_13(a, r, th_sp) * 0.5f * (B[2][idx] + B[2][idx.dec_x()]);

            auto Hth0 = ag_22(a, r_sm, sth, cth) * B[1][idx.dec_x()] +
                        sq_gamma_beta(a, r_sm, sth, cth) * 0.5f * (1.0f - beta) *
                            (D[2][idx] + D[2][idx.dec_x()]);

            auto Hth1 = ag_22(a, r_sp, sth, cth) * B[1][idx] +
                        sq_gamma_beta(a, r_sp, sth, cth) * 0.5f * (1.0f - beta) *
                            (D[2][idx.inc_x()] + D[2][idx]);

            auto idx_rhs = Conf::idx(
                index(pos[0] - grid.guard[0], pos[1] - grid.guard[1]), extl);
            rhs[idx_rhs] = D[2][idx] - dt * J[2][idx] +
                           prefactor * (dr * (Hr0 - Hr1) + dth * (Hth1 - Hth0));

            if (pos[0] == 100 && pos[1] == 250) {
              printf(
                  "Hr0 is %f, Hr1 is %f, Hth0 is %f, Hth1 is %f, dDphi is %f\n",
                  Hr0, Hr1, Hth0, Hth1,
                  prefactor * (dr * (Hr0 - Hr1) + dth * (Hth1 - Hth0)));
            }
            value_t du_coef = prefactor * dth * 0.5f * beta *
                              Metric_KS::sq_gamma_beta(a, r_sp, sth, cth);
            value_t dl_coef = -prefactor * dth * 0.5f * beta *
                              Metric_KS::sq_gamma_beta(a, r_sm, sth, cth);

            if (pos[0] == grid.guard[0]) {
              rhs[idx_rhs] += dl_coef * D[2][idx.dec_x()];
              // rhs[idx_rhs] += dl_coef * D[2][idx];
            } else if (pos[0] == grid.dims[0] - grid.guard[0] - 1) {
              rhs[idx_rhs] += du_coef * D[2][idx.inc_x()];
              // rhs[idx_rhs] += du_coef * D[2][idx];
            }

            d[pos[0] - grid.guard[0]] = 1.0f - (du_coef + dl_coef);
            du[pos[0] - grid.guard[0]] = -du_coef;
            dl[pos[0] - grid.guard[0]] = -dl_coef;
          }
        }
      },
      D.get_const_ptrs(), B.get_const_ptrs(), J.get_const_ptrs(),
      m_tmp_rhs.dev_ndptr(), m_tri_d.dev_ptr(), m_tri_dl.dev_ptr(),
      m_tri_du.dev_ptr(), m_a, this->m_beta);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  solve_tridiagonal();

  // D[2].copy_from(m_tmp_rhs);
  auto& grid = D.grid();
  select_dev(D[2], grid.guards(), grid.extent_less()) =
      select_dev_const(m_tmp_rhs);
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::update_Dr(vector_field<Conf>& D,
                                       const vector_field<Conf>& D0,
                                       const vector_field<Conf>& B,
                                       const vector_field<Conf>& B0,
                                       const vector_field<Conf>& J,
                                       value_t dt) {
  kernel_launch(
      [dt] __device__(auto D, auto B, auto J, auto tmp_field, auto a) {
        using namespace Metric_KS;

        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            value_t r =
                grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], false));

            value_t th = grid.template pos<1>(pos[1], true);
            value_t th_sp = grid.template pos<1>(pos[1], false);
            value_t th_sm = grid.template pos<1>(pos[1] - 1, false);
            value_t dth = th_sp - th_sm;
            if (th < TINY) th = 1.0e-5;

            value_t prefactor = dt / (Metric_KS::sqrt_gamma(a, r, th) * dth);

            value_t sth = math::sin(th_sp);
            value_t cth = math::cos(th_sp);
            auto Hph1 =
                ag_33(a, r, sth, cth) * B[2][idx] +
                ag_13(a, r, sth, cth) * 0.5f * (B[0][idx] + B[0][idx.inc_x()]) -
                sq_gamma_beta(a, r, sth, cth) * 0.5f *
                    (tmp_field[idx] + tmp_field[idx.inc_x()]);

            sth = math::sin(th_sm);
            cth = math::cos(th_sm);
            auto Hph0 =
                ag_33(a, r, sth, cth) * B[2][idx.dec_y()] +
                ag_13(a, r, sth, cth) * 0.5f *
                    (B[0][idx.dec_y()] + B[0][idx.dec_y().inc_x()]) -
                sq_gamma_beta(a, r, sth, cth) * 0.5f *
                    (tmp_field[idx.dec_y()] + tmp_field[idx.dec_y().inc_x()]);

            D[0][idx] = D[0][idx] - dt * J[0][idx] + prefactor * (Hph1 - Hph0);

            if (D[0][idx] != D[0][idx]) {
              printf(
                  "NaN detected in Dr update! B2 is %f, B0 is %f, tmp_field is "
                  "%f\n",
                  B[2][idx.dec_y()], B[0][idx.dec_y()], tmp_field[idx.dec_y()]);
              asm("trap;");
            }
            // prefactor *
            //     (H_ph<Conf>(B[2], B[0], tmp_field, idx, pos, grid, a) -
            //      H_ph<Conf>(B[2], B[0], tmp_field, idx.dec_y(1),
            //                 pos.dec_y(1), grid, a));
          }
        }
      },
      D.get_ptrs(), B.get_const_ptrs(), J.get_const_ptrs(),
      m_tmp_prev_field.dev_ndptr_const(), m_a);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::update(double dt, uint32_t step) {
  Logger::print_info("In GR KS solver! a is {}", m_a);

  if (this->m_update_b) {
    update_Bph(*(this->B), *(this->B0), *(this->E), *(this->E0), dt);
    update_Bth(*(this->B), *(this->B0), *(this->E), *(this->E0), dt);
    update_Br(*(this->B), *(this->B0), *(this->E), *(this->E0), dt);

    axis_boundary_b(*(this->B), m_ks_grid);
    // Communicate the new B values to guard cells
    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->B));
  }

  if (this->m_update_e) {
    update_Dph(*(this->E), *(this->E0), *(this->B), *(this->B0), *(this->J),
               dt);
    update_Dth(*(this->E), *(this->E0), *(this->B), *(this->B0), *(this->J),
               dt);
    update_Dr(*(this->E), *(this->E0), *(this->B), *(this->B0), *(this->J), dt);

    axis_boundary_e(*(this->E), m_ks_grid);
    // Communicate the new E values to guard cells
    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));
  }

  if (this->m_comm == nullptr || this->m_comm->domain_info().is_boundary[0]) {
    horizon_boundary(*(this->E), *(this->B), *(this->E0), *(this->B0),
                     m_ks_grid);
  }

  this->Etotal->copy_from(*(this->E));
  // this->Etotal->add_by(*(this->E0));

  this->Btotal->copy_from(*(this->B));
  // this->Btotal->add_by(*(this->B0));

  if (step % this->m_data_interval == 0) {
    compute_flux(*flux, *(this->Btotal), m_ks_grid);
  }

  CudaSafeCall(cudaDeviceSynchronize());
}

template class field_solver_gr_ks_cu<Config<2>>;

}  // namespace Aperture
