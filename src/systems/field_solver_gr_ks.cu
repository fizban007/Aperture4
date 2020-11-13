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
// buffer<char> sp_buffer;

template <typename Conf, typename ArrayType>
HD_INLINE typename Conf::value_t
H_r(const ArrayType& Br, const ArrayType& Bph, const typename Conf::idx_t& idx,
    const index_t<Conf::dim>& pos, const Grid<Conf::dim>& grid,
    typename Conf::value_t a) {
  using namespace Aperture::Metric_KS;
  auto r = grid_ks_t<Conf>::radius(grid.pos<0>(pos[0], true));
  auto th = grid_ks_t<Conf>::theta(grid.pos<1>(pos[1], false));

  return ag_11(a, r, th) * Br[idx] +
         ag_13(a, r, th) * 0.5f * (Bph[idx] + Bph[idx.dec_x(1)]);
}

template <typename Conf, typename ArrayType>
HD_INLINE typename Conf::value_t
H_th(const ArrayType& Bth, const ArrayType& Dph,
     const typename Conf::idx_t& idx, const index_t<Conf::dim>& pos,
     const Grid<Conf::dim>& grid, typename Conf::value_t a) {
  using namespace Aperture::Metric_KS;
  auto r = grid_ks_t<Conf>::radius(grid.pos<0>(pos[0], false));
  auto th = grid_ks_t<Conf>::theta(grid.pos<1>(pos[1], true));

  return ag_22(a, r, th) * Bth[idx] +
         sq_gamma_beta(a, r, th) * 0.5f * (Dph[idx] + Dph[idx.inc_x(1)]);
}

template <typename Conf, typename ArrayType>
HD_INLINE typename Conf::value_t
H_ph(const ArrayType& Bph, const ArrayType& Br, const ArrayType& Dth,
     const typename Conf::idx_t& idx, const index_t<Conf::dim>& pos,
     const Grid<Conf::dim>& grid, typename Conf::value_t a) {
  using namespace Aperture::Metric_KS;
  auto r = grid_ks_t<Conf>::radius(grid.pos<0>(pos[0], false));
  auto th = grid_ks_t<Conf>::theta(grid.pos<1>(pos[1], false));

  return ag_33(a, r, th) * Bph[idx] +
         ag_13(a, r, th) * 0.5f * (Br[idx] + Br[idx.inc_x(1)]) -
         sq_gamma_beta(a, r, th) * (Dth[idx] + Dth[idx.inc_x(1)]);
}

template <typename Conf, typename ArrayType>
HD_INLINE typename Conf::value_t
E_r(const ArrayType& Dr, const ArrayType& Dph, const typename Conf::idx_t& idx,
    const index_t<Conf::dim>& pos, const Grid<Conf::dim>& grid,
    typename Conf::value_t a) {
  using namespace Aperture::Metric_KS;
  auto r = grid_ks_t<Conf>::radius(grid.pos<0>(pos[0], false));
  auto th = grid_ks_t<Conf>::theta(grid.pos<1>(pos[1], true));

  return ag_11(a, r, th) * Dr[idx] +
         ag_13(a, r, th) * 0.5f * (Dph[idx] + Dph[idx.inc_x(1)]);
}

template <typename Conf, typename ArrayType>
HD_INLINE typename Conf::value_t
E_th(const ArrayType& Dth, const ArrayType& Bph,
     const typename Conf::idx_t& idx, const index_t<Conf::dim>& pos,
     const Grid<Conf::dim>& grid, typename Conf::value_t a) {
  using namespace Aperture::Metric_KS;
  auto r = grid_ks_t<Conf>::radius(grid.pos<0>(pos[0], true));
  auto th = grid_ks_t<Conf>::theta(grid.pos<1>(pos[1], false));

  return ag_22(a, r, th) * Dth[idx] -
         sq_gamma_beta(a, r, th) * 0.5f * (Bph[idx] + Bph[idx.dec_x(1)]);
}

template <typename Conf, typename ArrayType>
HD_INLINE typename Conf::value_t
E_ph(const ArrayType& Dph, const ArrayType& Dr, const ArrayType& Bth,
     const typename Conf::idx_t& idx, const index_t<Conf::dim>& pos,
     const Grid<Conf::dim>& grid, typename Conf::value_t a) {
  using namespace Aperture::Metric_KS;
  auto r = grid_ks_t<Conf>::radius(grid.pos<0>(pos[0], true));
  auto th = grid_ks_t<Conf>::theta(grid.pos<1>(pos[1], true));

  return ag_33(a, r, th) * Dph[idx] +
         ag_13(a, r, th) * 0.5f * (Dr[idx] + Dr[idx.dec_x(1)]) -
         sq_gamma_beta(a, r, th) * (Bth[idx] + Bth[idx.dec_x(1)]);
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
            Scalar th_p = grid_ks_t<Conf>::theta(grid.template pos<1>(n1 + 1, true));
            Scalar th_m = grid_ks_t<Conf>::theta(grid.template pos<1>(n1, true));
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

  cusparseCreate(&sp_handle);
  auto ext = this->m_grid.extent();

  m_tmp_rhs.set_memtype(MemType::device_only);
  m_tmp_rhs.resize(ext);

  m_tmp_prev_field.set_memtype(MemType::device_only);
  m_tmp_prev_field.resize(ext);

  m_tri_dl.set_memtype(MemType::device_only);
  m_tri_dl.resize(this->m_grid.dims[0]);
  m_tri_dl.assign_dev(0.0f);

  m_tri_d.set_memtype(MemType::device_only);
  m_tri_d.resize(this->m_grid.dims[0]);
  m_tri_d.assign_dev(0.0f);

  m_tri_du.set_memtype(MemType::device_only);
  m_tri_du.resize(this->m_grid.dims[0]);
  m_tri_du.assign_dev(0.0f);

  size_t buffer_size;
#if USE_DOUBLE
  cusparseDgtsv2_bufferSizeExt(sp_handle, ext[0], ext[1], m_tri_dl.dev_ptr(),
                               m_tri_d.dev_ptr(), m_tri_du.dev_ptr(),
                               m_tmp_rhs.dev_ptr(), ext[0], &buffer_size);
#else
  cusparseSgtsv2_bufferSizeExt(sp_handle, ext[0], ext[1], m_tri_dl.dev_ptr(),
                               m_tri_d.dev_ptr(), m_tri_du.dev_ptr(),
                               m_tmp_rhs.dev_ptr(), ext[0], &buffer_size);
#endif
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
  auto ext = this->m_grid.extent();
#if USE_DOUBLE
  status = cusparseDgtsv2(sp_handle, ext[0], ext[1], m_tri_dl.dev_ptr(),
                          m_tri_d.dev_ptr(), m_tri_du.dev_ptr(),
                          m_tmp_rhs.dev_ptr(), ext[0], sp_buffer.dev_ptr());
#else
  status = cusparseSgtsv2(sp_handle, ext[0], ext[1], m_tri_dl.dev_ptr(),
                          m_tri_d.dev_ptr(), m_tri_du.dev_ptr(),
                          m_tmp_rhs.dev_ptr(), ext[0], sp_buffer.dev_ptr());
#endif
  if (status != CUSPARSE_STATUS_SUCCESS) {
    Logger::print_err("cusparse failure at Bth update! Error code {}", status);
  }
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::update_Bth(vector_field<Conf>& B,
                                        const vector_field<Conf>& D,
                                        value_t dt) {
  m_tmp_rhs.assign_dev(0.0f);
  m_tmp_prev_field.copy_from(B[1]);

  // First assemble the right hand side and the diagonals of the tri-diagonal
  // equation
  kernel_launch(
      [dt] __device__(auto B, auto D, auto rhs, auto d, auto dl, auto du,
                      auto a) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
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
            value_t th = grid.template pos<1>(pos[1], true);

            value_t prefactor = dt / (Metric_KS::sqrt_gamma(a, r, th) * dr);
            rhs[idx] =
                B[1][idx] +
                prefactor * (E_ph<Conf>(D[2], D[0], B[1], idx.inc_x(1),
                                        pos.inc_x(1), grid, a) -
                             E_ph<Conf>(D[2], D[0], B[1], idx, pos, grid, a));

            value_t du_coef =
                prefactor * 0.5f * Metric_KS::sq_gamma_beta(a, r_sp, th);
            value_t dl_coef =
                -prefactor * 0.5f * Metric_KS::sq_gamma_beta(a, r_sm, th);
            d[pos[0]] = 1.0f - (du_coef + dl_coef);

            du[pos[0]] = du_coef;
            dl[pos[0]] = dl_coef;
          }
        }
      },
      B.get_const_ptrs(), D.get_const_ptrs(), m_tmp_rhs.dev_ndptr(),
      m_tri_d.dev_ptr(), m_tri_dl.dev_ptr(), m_tri_du.dev_ptr(), m_a);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  solve_tridiagonal();

  B[1].copy_from(m_tmp_rhs);
  select_dev(m_tmp_prev_field) = m_tmp_rhs * 0.5f + m_tmp_prev_field * 0.5f;
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::update_Bph(vector_field<Conf>& B,
                                        const vector_field<Conf>& D,
                                        value_t dt) {
  m_tmp_rhs.assign_dev(0.0f);

  // First assemble the right hand side and the diagonals of the tri-diagonal
  // equation
  kernel_launch(
      [dt] __device__(auto B, auto D, auto rhs, auto d, auto dl, auto du,
                      auto a) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
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

            value_t th = grid.template pos<1>(pos[1], false);
            value_t th_sp = grid.template pos<1>(pos[1] + 1, true);
            value_t th_sm = grid.template pos<1>(pos[1], true);
            value_t dth = th_sp - th_sm;

            value_t prefactor =
                dt / (Metric_KS::sqrt_gamma(a, r, th) * dr * dth);
            rhs[idx] =
                B[2][idx] +
                prefactor * (dr * (E_r<Conf>(D[0], D[2], idx.inc_y(1),
                                             pos.inc_y(1), grid, a) -
                                   E_r<Conf>(D[0], D[2], idx, pos, grid, a)) -
                             dth * (E_th<Conf>(D[1], B[2], idx.inc_x(1),
                                               pos.inc_x(1), grid, a) -
                                    E_th<Conf>(D[1], B[2], idx, pos, grid, a)));

            value_t du_coef =
                prefactor * dth * 0.5f * Metric_KS::sq_gamma_beta(a, r_sp, th);
            value_t dl_coef =
                -prefactor * dth * 0.5f * Metric_KS::sq_gamma_beta(a, r_sm, th);
            d[pos[0]] = 1.0f - (du_coef + dl_coef);

            du[pos[0]] = du_coef;
            dl[pos[0]] = dl_coef;
          }
        }
      },
      B.get_const_ptrs(), D.get_const_ptrs(), m_tmp_rhs.dev_ndptr(),
      m_tri_d.dev_ptr(), m_tri_dl.dev_ptr(), m_tri_du.dev_ptr(), m_a);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  solve_tridiagonal();

  B[2].copy_from(m_tmp_rhs);
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::update_Br(vector_field<Conf>& B,
                                       const vector_field<Conf>& D,
                                       value_t dt) {
  kernel_launch(
      [dt] __device__(auto B, auto D, auto tmp_field, auto a) {
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

            value_t prefactor = dt / (Metric_KS::sqrt_gamma(a, r, th) * dth);
            B[0][idx] =
                B[0][idx] +
                prefactor *
                    (E_ph<Conf>(D[2], D[0], tmp_field, idx, pos, grid, a) -
                     E_ph<Conf>(D[2], D[0], tmp_field, idx.inc_y(1),
                                pos.inc_y(1), grid, a));
          }
        }
      },
      B.get_ptrs(), D.get_const_ptrs(), m_tmp_prev_field.dev_ndptr_const(),
      m_a);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::update_Dth(vector_field<Conf>& D,
                                        const vector_field<Conf>& B,
                                        const vector_field<Conf>& J,
                                        value_t dt) {
  m_tmp_rhs.assign_dev(0.0f);
  m_tmp_prev_field.copy_from(D[1]);

  // First assemble the right hand side and the diagonals of the tri-diagonal
  // equation
  kernel_launch(
      [dt] __device__(auto D, auto B, auto J, auto rhs, auto d, auto dl,
                      auto du, auto a) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
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
            value_t th = grid.template pos<1>(pos[1], false);

            value_t prefactor = dt / (Metric_KS::sqrt_gamma(a, r, th) * dr);
            rhs[idx] =
                D[1][idx] - dt * J[1][idx] +
                prefactor * (H_ph<Conf>(B[2], B[0], D[1], idx.dec_x(1),
                                        pos.dec_x(1), grid, a) -
                             H_ph<Conf>(B[2], B[0], D[1], idx, pos, grid, a));

            value_t du_coef =
                prefactor * 0.5f * Metric_KS::sq_gamma_beta(a, r_sp, th);
            value_t dl_coef =
                -prefactor * 0.5f * Metric_KS::sq_gamma_beta(a, r_sm, th);
            d[pos[0]] = 1.0f - (du_coef + dl_coef);

            du[pos[0]] = du_coef;
            dl[pos[0]] = dl_coef;
          }
        }
      },
      D.get_const_ptrs(), B.get_const_ptrs(), J.get_const_ptrs(),
      m_tmp_rhs.dev_ndptr(), m_tri_d.dev_ptr(), m_tri_dl.dev_ptr(),
      m_tri_du.dev_ptr(), m_a);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  solve_tridiagonal();

  D[1].copy_from(m_tmp_rhs);
  select_dev(m_tmp_prev_field) = m_tmp_rhs * 0.5f + m_tmp_prev_field * 0.5f;
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::update_Dph(vector_field<Conf>& D,
                                        const vector_field<Conf>& B,
                                        const vector_field<Conf>& J,
                                        value_t dt) {
  m_tmp_rhs.assign_dev(0.0f);

  // First assemble the right hand side and the diagonals of the tri-diagonal
  // equation
  kernel_launch(
      [dt] __device__(auto D, auto B, auto J, auto rhs, auto d, auto dl,
                      auto du, auto a) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
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

            value_t th = grid.template pos<1>(pos[1], true);
            value_t th_sp = grid.template pos<1>(pos[1], false);
            value_t th_sm = grid.template pos<1>(pos[1] - 1, false);
            value_t dth = th_sp - th_sm;

            value_t prefactor =
                dt / (Metric_KS::sqrt_gamma(a, r, th) * dr * dth);
            rhs[idx] =
                D[2][idx] - dt * J[2][idx] +
                prefactor * (dr * (H_r<Conf>(B[0], B[2], idx.dec_y(1),
                                             pos.dec_y(1), grid, a) -
                                   H_r<Conf>(B[0], B[2], idx, pos, grid, a)) -
                             dth * (H_th<Conf>(B[1], D[2], idx.dec_x(1),
                                               pos.dec_x(1), grid, a) -
                                    H_th<Conf>(B[1], D[2], idx, pos, grid, a)));

            value_t du_coef =
                prefactor * dth * 0.5f * Metric_KS::sq_gamma_beta(a, r_sp, th);
            value_t dl_coef =
                -prefactor * dth * 0.5f * Metric_KS::sq_gamma_beta(a, r_sm, th);
            d[pos[0]] = 1.0f - (du_coef + dl_coef);

            du[pos[0]] = du_coef;
            dl[pos[0]] = dl_coef;
          }
        }
      },
      D.get_const_ptrs(), B.get_const_ptrs(), J.get_const_ptrs(),
      m_tmp_rhs.dev_ndptr(), m_tri_d.dev_ptr(), m_tri_dl.dev_ptr(),
      m_tri_du.dev_ptr(), m_a);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  solve_tridiagonal();

  D[2].copy_from(m_tmp_rhs);
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::update_Dr(vector_field<Conf>& D,
                                       const vector_field<Conf>& B,
                                       const vector_field<Conf>& J,
                                       value_t dt) {
  kernel_launch(
      [dt] __device__(auto D, auto B, auto J, auto tmp_field, auto a) {
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

            value_t prefactor = dt / (Metric_KS::sqrt_gamma(a, r, th) * dth);
            D[0][idx] =
                D[0][idx] - dt * J[0][idx] +
                prefactor *
                    (H_ph<Conf>(B[2], B[0], tmp_field, idx, pos, grid, a) -
                     H_ph<Conf>(B[2], B[0], tmp_field, idx.dec_y(1),
                                pos.dec_y(1), grid, a));
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
    update_Bth(*(this->B), *(this->E), dt);
    update_Bph(*(this->B), *(this->E), dt);
    update_Br(*(this->B), *(this->E), dt);

    // Communicate the new B values to guard cells
    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->B));
  }

  if (this->m_update_e) {
    update_Dth(*(this->E), *(this->B), *(this->J), dt);
    update_Dph(*(this->E), *(this->B), *(this->J), dt);
    update_Dr(*(this->E), *(this->B), *(this->J), dt);

    // Communicate the new E values to guard cells
    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));
  }

  this->Etotal->copy_from(*(this->E0));
  this->Etotal->add_by(*(this->E));

  this->Btotal->copy_from(*(this->B0));
  this->Btotal->add_by(*(this->B));

  if (step % this->m_data_interval == 0) {
    compute_flux(*flux, *(this->Btotal), m_ks_grid);
  }

  CudaSafeCall(cudaDeviceSynchronize());
}

template class field_solver_gr_ks_cu<Config<2>>;

}  // namespace Aperture
