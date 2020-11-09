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
#include "field_solver_gr_ks.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "utils/kernel_helper.hpp"
#include "utils/timer.h"
#include <cusparse.h>

namespace Aperture {

namespace {

cusparseHandle_t sp_handle;

template <typename Conf, typename ArrayType>
HD_INLINE typename Conf::value_t
H_r(const ArrayType& Br, const ArrayType& Bph, const typename Conf::idx_t& idx,
    const index_t<Conf::dim>& pos, const grid_ks_t<Conf>& grid) {
  using namespace Aperture::Metric_KS;
  auto r = grid_ks_t<Conf>::radius(grid.pos<0>(pos[0], true));
  auto th = grid_ks_t<Conf>::theta(grid.pos<1>(pos[1], false));

  return ag_11(grid.a, r, th) * Br[idx] +
         ag_13(grid.a, r, th) * 0.5f * (Bph[idx] + Bph[idx.dec_x(1)]);
}

template <typename Conf, typename ArrayType>
HD_INLINE typename Conf::value_t
H_th(const ArrayType& Bth, const ArrayType& Dph,
     const typename Conf::idx_t& idx, const index_t<Conf::dim>& pos,
     const grid_ks_t<Conf>& grid) {
  using namespace Aperture::Metric_KS;
  auto r = grid_ks_t<Conf>::radius(grid.pos<0>(pos[0], false));
  auto th = grid_ks_t<Conf>::theta(grid.pos<1>(pos[1], true));

  return ag_22(grid.a, r, th) * Bth[idx] +
         sq_gamma_beta(grid.a, r, th) * 0.5f * (Dph[idx] + Dph[idx.inc_x(1)]);
}

template <typename Conf, typename ArrayType>
HD_INLINE typename Conf::value_t
H_ph(const ArrayType& Bph, const ArrayType& Br, const ArrayType& Dth,
     const typename Conf::idx_t& idx, const index_t<Conf::dim>& pos,
     const grid_ks_t<Conf>& grid) {
  using namespace Aperture::Metric_KS;
  auto r = grid_ks_t<Conf>::radius(grid.pos<0>(pos[0], false));
  auto th = grid_ks_t<Conf>::theta(grid.pos<1>(pos[1], false));

  return ag_33(grid.a, r, th) * Bph[idx] +
         ag_13(grid.a, r, th) * 0.5f * (Br[idx] + Br[idx.inc_x(1)]) -
         sq_gamma_beta(grid.a, r, th) * (Dth[idx] + Dth[idx.inc_x(1)]);
}

template <typename Conf, typename ArrayType>
HD_INLINE typename Conf::value_t
E_r(const ArrayType& Dr, const ArrayType& Dph, const typename Conf::idx_t& idx,
    const index_t<Conf::dim>& pos, const grid_ks_t<Conf>& grid) {
  using namespace Aperture::Metric_KS;
  auto r = grid_ks_t<Conf>::radius(grid.pos<0>(pos[0], false));
  auto th = grid_ks_t<Conf>::theta(grid.pos<1>(pos[1], true));

  return ag_11(grid.a, r, th) * Dr[idx] +
         ag_13(grid.a, r, th) * 0.5f * (Dph[idx] + Dph[idx.inc_x(1)]);
}

template <typename Conf, typename ArrayType>
HD_INLINE typename Conf::value_t
E_th(const ArrayType& Dth, const ArrayType& Bph,
     const typename Conf::idx_t& idx, const index_t<Conf::dim>& pos,
     const grid_ks_t<Conf>& grid) {
  using namespace Aperture::Metric_KS;
  auto r = grid_ks_t<Conf>::radius(grid.pos<0>(pos[0], true));
  auto th = grid_ks_t<Conf>::theta(grid.pos<1>(pos[1], false));

  return ag_22(grid.a, r, th) * Dth[idx] -
         sq_gamma_beta(grid.a, r, th) * 0.5f * (Bph[idx] + Bph[idx.dec_x(1)]);
}

template <typename Conf, typename ArrayType>
HD_INLINE typename Conf::value_t
E_ph(const ArrayType& Dph, const ArrayType& Dr, const ArrayType& Bth,
     const typename Conf::idx_t& idx, const index_t<Conf::dim>& pos,
     const grid_ks_t<Conf>& grid) {
  using namespace Aperture::Metric_KS;
  auto r = grid_ks_t<Conf>::radius(grid.pos<0>(pos[0], true));
  auto th = grid_ks_t<Conf>::theta(grid.pos<1>(pos[1], true));

  return ag_33(grid.a, r, th) * Dph[idx] +
         ag_13(grid.a, r, th) * 0.5f * (Dr[idx] + Dr[idx.dec_x(1)]) -
         sq_gamma_beta(grid.a, r, th) * (Bth[idx] + Bth[idx.dec_x(1)]);
}

}  // namespace

template <typename Conf>
field_solver_gr_ks_cu<Conf>::~field_solver_gr_ks_cu() {
  cusparseDestroy(sp_handle);
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::init() {
  cusparseCreate(&sp_handle);

  m_tmp_rhs.set_memtype(MemType::device_only);
  m_tmp_rhs.resize(this->m_grid.extent());

  m_tri_dl.set_memtype(MemType::device_only);
  m_tri_dl.resize(this->m_grid.dims[0]);

  m_tri_d.set_memtype(MemType::device_only);
  m_tri_d.resize(this->m_grid.dims[0]);

  m_tri_du.set_memtype(MemType::device_only);
  m_tri_du.resize(this->m_grid.dims[0]);

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
field_solver_gr_ks_cu<Conf>::update(double dt, uint32_t step) {


}

template class field_solver_gr_ks_cu<Config<2>>;

}  // namespace Aperture
