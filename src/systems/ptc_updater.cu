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

#include "core/constant_mem_func.h"
#include "core/detail/multi_array_helpers.h"
#include "core/math.hpp"
#include "data/curand_states.h"
#include "framework/config.h"
#include "helpers/ptc_update_helper.hpp"
#include "ptc_updater.h"
#include "utils/double_buffer.h"
#include "utils/interpolation.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include "utils/util_functions.h"

namespace Aperture {

namespace {

template <typename Conf>
void
filter(typename Conf::multi_array_t& result, typename Conf::multi_array_t& f,
       vec_t<bool, 4> is_boundary) {
  kernel_launch(
      [] __device__(auto result, auto f, auto is_boundary) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          // auto idx = idx_t(n, ext);
          auto pos = idx.get_pos();
          if (grid.is_in_bound(pos)) {
            int dx_plus = 1, dx_minus = 1, dy_plus = 1, dy_minus = 1;
            if (is_boundary[0] && pos[0] == grid.skirt[0]) dx_minus = 0;
            if (is_boundary[1] && pos[0] == grid.dims[0] - grid.skirt[0] - 1)
              dx_plus = 0;
            if (is_boundary[2] && pos[1] == grid.skirt[1]) dy_minus = 0;
            if (is_boundary[3] && pos[1] == grid.dims[1] - grid.skirt[1] - 1)
              dy_plus = 0;
            result[idx] = 0.25f * f[idx];
            auto idx_px = idx.inc_x(dx_plus);
            auto idx_mx = idx.dec_x(dx_minus);
            auto idx_py = idx.inc_y(dy_plus);
            auto idx_my = idx.dec_y(dy_minus);
            result[idx] += 0.125f * f[idx_px];
            result[idx] += 0.125f * f[idx_mx];
            result[idx] += 0.125f * f[idx_py];
            result[idx] += 0.125f * f[idx_my];
            result[idx] += 0.0625f * f[idx_px.inc_y(dy_plus)];
            result[idx] += 0.0625f * f[idx_px.dec_y(dy_minus)];
            result[idx] += 0.0625f * f[idx_mx.inc_y(dy_plus)];
            result[idx] += 0.0625f * f[idx_mx.dec_y(dy_minus)];
          }
        }
      },
      result.dev_ndptr(), f.dev_ndptr_const(), vec_t<bool, 4>(is_boundary));
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
  f.dev_copy_from(result);
}

}  // namespace

template <typename Conf>
void
ptc_updater_cu<Conf>::init() {
  init_dev_charge_mass(this->m_charges.data(), this->m_masses.data());

  m_rho_ptrs.set_memtype(MemType::host_device);
  m_rho_ptrs.resize(this->m_num_species);
  for (int i = 0; i < this->m_num_species; i++) {
    m_rho_ptrs[i] = this->Rho[i]->dev_ndptr();
  }
  m_rho_ptrs.copy_to_device();

  // Allocate the tmp array for current filtering
  this->jtmp = std::make_unique<typename Conf::multi_array_t>(
      this->m_grid.extent(), MemType::host_device);

  this->m_env.get_data_optional("photons", &(this->ph));
  this->m_env.get_data_optional("Rho_ph", &(this->rho_ph));
}

template <typename Conf>
void
ptc_updater_cu<Conf>::register_data_components() {
  size_t max_ptc_num = 1000000;
  this->m_env.params().get_value("max_ptc_num", max_ptc_num);
  // Prefer device_only, but can take other possibilities if data is already
  // there
  this->ptc = this->m_env.template register_data<particle_data_t>(
      "particles", max_ptc_num, MemType::device_only);
  this->ptc->include_in_snapshot(true);

  this->E = this->m_env.template register_data<vector_field<Conf>>(
      "E", this->m_grid, field_type::edge_centered, MemType::host_device);
  this->B = this->m_env.template register_data<vector_field<Conf>>(
      "B", this->m_grid, field_type::face_centered, MemType::host_device);
  this->J = this->m_env.template register_data<vector_field<Conf>>(
      "J", this->m_grid, field_type::edge_centered, MemType::host_device);

  this->m_env.params().get_value("num_species", this->m_num_species);
  this->Rho.resize(this->m_num_species);
  for (int i = 0; i < this->m_num_species; i++) {
    this->Rho[i] = this->m_env.template register_data<scalar_field<Conf>>(
        std::string("Rho_") + ptc_type_name(i), this->m_grid,
        field_type::vert_centered, MemType::host_device);
  }

  int rand_seed = 1234;
  this->m_env.params().get_value("rand_seed", rand_seed);
  m_rand_states = this->m_env.template register_data<curand_states_t>(
      "rand_states", size_t(512 * 1024), rand_seed);
  m_rand_states->include_in_snapshot(true);
}

template <typename Conf>
void
ptc_updater_cu<Conf>::push_default(value_t dt) {
  // dispatch according to enum. This will also instantiate all the versions of
  // push
  if (this->m_pusher == Pusher::boris) {
    auto pusher = pusher_impl_t<boris_pusher>{};
    push(dt, pusher);
  } else if (this->m_pusher == Pusher::vay) {
    auto pusher = pusher_impl_t<vay_pusher>{};
    push(dt, pusher);
  } else if (this->m_pusher == Pusher::higuera) {
    auto pusher = pusher_impl_t<higuera_pusher>{};
    push(dt, pusher);
  }
}

template <typename Conf>
void
ptc_updater_cu<Conf>::move_deposit_1d(value_t dt, uint32_t step) {
  auto num = this->ptc->number();
  if (num > 0) {
    auto ext = this->m_grid.extent();

    kernel_launch(
        [ext, num, dt, step] __device__(auto ptc, auto J, auto Rho,
                                        auto rho_interval) {
          using spline_t = typename base_class::spline_t;
          auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
          for (auto n : grid_stride_range(0, num)) {
            uint32_t cell = ptc.cell[n];
            if (cell == empty_cell) continue;

            auto idx = J[0].idx_at(cell, ext);
            auto pos = idx.get_pos();

            // step 1: Move particles
            // auto x1 = ptc.x1[n], x2 = ptc.x2[n], x3 = ptc.x3[n];
            vec_t<value_t, 3> x(ptc.x1[n], ptc.x2[n], ptc.x3[n]);
            // value_t v1 = ptc.p1[n], v2 = ptc.p2[n], v3 = ptc.p3[n],
            //         gamma = ptc.E[n];
            vec_t<value_t, 3> v(ptc.p1[n], ptc.p2[n], ptc.p3[n]);
            value_t gamma = ptc.E[n];

            v /= gamma;

            auto new_x = x;
            int dc = 0;
            // auto new_x1 = x1 + (v1 * dt) * grid.inv_delta[0];
            new_x[0] = x[0] + (v[0] * dt) * grid.inv_delta[0];
            dc = std::floor(new_x[0]);
            pos[0] += dc;
            ptc.x1[n] = new_x[0] - (value_t)dc;
            ptc.x2[n] = x[1] + v[1] * dt;
            ptc.x3[n] = x[2] + v[2] * dt;

            ptc.cell[n] = J[0].get_idx(pos, ext).linear;

            // step 2: Deposit current
            auto flag = ptc.flag[n];
            int sp = get_ptc_type(flag);
            if (check_flag(flag, PtcFlag::ignore_current)) continue;
            value_t weight = dev_charges[sp] * ptc.weight[n];

            deposit_1d<spline_t>(x, new_x, dc, v, J, Rho, idx, weight, sp,
                                 step % rho_interval == 0);
            // int i_0 = (dc1 == -1 ? -spline_t::radius : 1 - spline_t::radius);
            // int i_1 = (dc1 == 1 ? spline_t::radius + 1 : spline_t::radius);
            // value_t djx = 0.0f;
            // for (int i = i_0; i <= i_1; i++) {
            //   value_t sx0 = interp(-x1 + i);
            //   value_t sx1 = interp(-new_x1 + i);

            //   // j1 is movement in x1
            //   int offset = i + pos[0] - dc1;
            //   djx += sx1 - sx0;
            //   atomicAdd(&J[0][offset], -weight * djx * grid.delta[0] / dt);
            //   // Logger::print_debug("J0 is {}", (*J)[0][offset]);

            //   // j2 is simply v2 times rho at center
            //   value_t val1 = 0.5f * (sx0 + sx1);
            //   atomicAdd(&J[1][offset], weight * v2 * val1);

            //   // j3 is simply v3 times rho at center
            //   atomicAdd(&J[2][offset], weight * v3 * val1);

            //   // rho is deposited at the final position
            //   if (step % rho_interval == 0) {
            //     atomicAdd(&Rho[sp][offset], weight * sx1);
            //   }
            // }
          }
        },
        this->ptc->dev_ptrs(), this->J->get_ptrs(), m_rho_ptrs.dev_ptr(),
        this->m_rho_interval);

    // Modify J with prefactor
    kernel_launch([dt] __device__(auto J) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          J[0][idx] *= grid.delta[0] / dt;
        }
      }, this->J->get_ptrs());
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();
  }
}

template <typename Conf>
void
ptc_updater_cu<Conf>::move_deposit_2d(value_t dt, uint32_t step) {
  this->J->init();
  for (auto rho : this->Rho) rho->init();

  auto num = this->ptc->number();
  if (num > 0) {
    kernel_launch(
        [num, dt, step] __device__(auto ptc, auto J, auto Rho,
                                   auto rho_interval) {
          using spline_t = typename Conf::spline_t;
          auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
          auto ext = grid.extent();
          // Obtain a local pointer to the shared array
          // extern __shared__ char shared_array[];
          // value_t* djy = (value_t*)&shared_array[threadIdx.x * sizeof(value_t) *
          //                                        (2 * spline_t::radius + 1)];

          for (auto n : grid_stride_range(0, num)) {
            uint32_t cell = ptc.cell[n];
            if (cell == empty_cell) continue;

            auto idx = typename Conf::idx_t(cell, ext);
            auto pos = idx.get_pos();

            // step 1: Move particles
            // auto x1 = ptc.x1[n], x2 = ptc.x2[n], x3 = ptc.x3[n];
            vec_t<value_t, 3> x(ptc.x1[n], ptc.x2[n], ptc.x3[n]);
            // value_t v1 = ptc.p1[n], v2 = ptc.p2[n], v3 = ptc.p3[n],
            //         gamma = ptc.E[n];
            vec_t<value_t, 3> v(ptc.p1[n], ptc.p2[n], ptc.p3[n]);
            value_t gamma = ptc.E[n];

            v /= gamma;

            auto new_x = x;
            vec_t<int, 2> dc = 0;

            new_x[0] = x[0] + (v[0] * dt) * grid.inv_delta[0];
            new_x[1] = x[1] + (v[1] * dt) * grid.inv_delta[1];
            dc[0] = std::floor(new_x[0]);
            dc[1] = std::floor(new_x[1]);

            // if (dc[0] > 1 || dc[0] < -1 || dc[1] > 1 || dc[1] < -1) {
            //   printf("----------------- Error: moved more than 1 cell, n is %lu!\n", n);
            // }

            pos[0] += dc[0];
            pos[1] += dc[1];

            ptc.x1[n] = new_x[0] - (value_t)dc[0];
            ptc.x2[n] = new_x[1] - (value_t)dc[1];
            ptc.x3[n] = x[2] + v[2] * dt;

            ptc.cell[n] = Conf::idx(pos, ext).linear;

            // step 2: Deposit current
            auto flag = ptc.flag[n];
            auto sp = get_ptc_type(flag);
            // auto interp = spline_t{};
            if (check_flag(flag, PtcFlag::ignore_current)) continue;
            value_t weight = dev_charges[sp] * ptc.weight[n];

            deposit_2d<spline_t>(x, new_x, dc, v[2], J, Rho, idx, weight, sp,
                                 step % rho_interval == 0);
//             int j_0 = (dc2 == -1 ? -spline_t::radius : 1 - spline_t::radius);
//             int j_1 = (dc2 == 1 ? spline_t::radius + 1 : spline_t::radius);
//             int i_0 = (dc1 == -1 ? -spline_t::radius : 1 - spline_t::radius);
//             int i_1 = (dc1 == 1 ? spline_t::radius + 1 : spline_t::radius);

//         // Reset djy since it could be nonzero from previous particle
// #pragma unroll
//             for (int j = 0; j < 2 * spline_t::radius + 1; j++) {
//               djy[j] = 0.0;
//             }

//             // value_t djy[2 * spline_t::radius + 1] = {};
//             for (int j = j_0; j <= j_1; j++) {
//               value_t sy0 = interp(-x2 + j);
//               value_t sy1 = interp(-new_x2 + j);

//               value_t djx = 0.0f;
//               for (int i = i_0; i <= i_1; i++) {
//                 value_t sx0 = interp(-x1 + i);
//                 value_t sx1 = interp(-new_x1 + i);

//                 // j1 is movement in x1
//                 auto offset = idx.inc_x(i).inc_y(j);
//                 djx += movement2d(sy0, sy1, sx0, sx1);
//                 if (math::abs(djx) > TINY)
//                   atomicAdd(&J[0][offset], -weight * djx * grid.delta[0] / dt);
//                 // Logger::print_debug("J0 is {}", (*J)[0][offset]);

//                 // j2 is movement in x2
//                 djy[i - i_0] += movement2d(sx0, sx1, sy0, sy1);
//                 if (math::abs(djy[i - i_0]) > TINY)
//                   atomicAdd(&J[1][offset],
//                             -weight * djy[i - i_0] * grid.delta[1] / dt);

//                 // j3 is simply v3 times rho at center
//                 atomicAdd(&J[2][offset],
//                           weight * v3 * center2d(sx0, sx1, sy0, sy1));

//                 // rho is deposited at the final position
//                 if (step % rho_interval == 0) {
//                   if (math::abs(sx1 * sy1) > TINY) {
//                     atomicAdd(&Rho[sp][offset], weight * sx1 * sy1);
//                   }
//                 }
//               }
//             }
          }
        },
        this->ptc->dev_ptrs(), this->J->get_ptrs(), m_rho_ptrs.dev_ptr(),
        this->m_rho_interval);

    // Modify J with prefactor
    kernel_launch([dt] __device__(auto J) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          J[0][idx] *= grid.delta[0] / dt;
          J[1][idx] *= grid.delta[1] / dt;
        }
      }, this->J->get_ptrs());
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();
  }
}

template <typename Conf>
void
ptc_updater_cu<Conf>::move_deposit_3d(value_t dt, uint32_t step) {
  auto num = this->ptc->number();
  if (num > 0) {
    auto ext = this->m_grid.extent();

    kernel_launch(
        [ext, num, dt, step] __device__(auto ptc, auto J, auto Rho,
                                        auto rho_interval) {
          using spline_t = typename base_class::spline_t;
          auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
          for (auto n : grid_stride_range(0, num)) {
            uint32_t cell = ptc.cell[n];
            if (cell == empty_cell) continue;

            auto idx = typename Conf::idx_t(cell, ext);
            auto pos = idx.get_pos();

            // step 1: Move particles
            // auto x1 = ptc.x1[n], x2 = ptc.x2[n], x3 = ptc.x3[n];
            vec_t<value_t, 3> x(ptc.x1[n], ptc.x2[n], ptc.x3[n]);
            // value_t v1 = ptc.p1[n], v2 = ptc.p2[n], v3 = ptc.p3[n],
            //         gamma = ptc.E[n];
            vec_t<value_t, 3> v(ptc.p1[n], ptc.p2[n], ptc.p3[n]);
            value_t gamma = ptc.E[n];

            v /= gamma;
            // v1 /= gamma;
            // v2 /= gamma;
            // v3 /= gamma;

            auto new_x = x;
            new_x[0] = x[0] + v[0] * dt * grid.inv_delta[0];
            new_x[1] = x[1] + v[1] * dt * grid.inv_delta[1];
            new_x[2] = x[2] + v[2] * dt * grid.inv_delta[2];

            vec_t<int, 3> dc;
            // auto new_x1 = x1 + (v1 * dt) * grid.inv_delta[0];
            dc[0] = std::floor(new_x[0]);
            pos[0] += dc[0];
            ptc.x1[n] = new_x[0] - (value_t)dc[0];

            // auto new_x2 = x2 + (v2 * dt) * grid.inv_delta[1];
            dc[1] = std::floor(new_x[1]);
            pos[1] += dc[1];
            ptc.x2[n] = new_x[1] - (value_t)dc[1];

            // auto new_x3 = x3 + (v3 * dt) * grid.inv_delta[2];
            dc[2] = std::floor(new_x[2]);
            pos[2] += dc[2];
            ptc.x3[n] = new_x[2] - (value_t)dc[2];

            ptc.cell[n] = J[0].get_idx(pos, ext).linear;

            // step 2: Deposit current
            auto flag = ptc.flag[n];
            auto sp = get_ptc_type(flag);
            // auto interp = spline_t{};
            if (check_flag(flag, PtcFlag::ignore_current)) continue;
            value_t weight = dev_charges[sp] * ptc.weight[n];

            deposit_3d<spline_t>(x, new_x, dc, v, J, Rho, idx, weight, sp,
                                 step % rho_interval == 0);
            // int k_0 = (dc3 == -1 ? -spline_t::radius : 1 - spline_t::radius);
            // int k_1 = (dc3 == 1 ? spline_t::radius + 1 : spline_t::radius);
            // int j_0 = (dc2 == -1 ? -spline_t::radius : 1 - spline_t::radius);
            // int j_1 = (dc2 == 1 ? spline_t::radius + 1 : spline_t::radius);
            // int i_0 = (dc1 == -1 ? -spline_t::radius : 1 - spline_t::radius);
            // int i_1 = (dc1 == 1 ? spline_t::radius + 1 : spline_t::radius);

            // value_t djz[2 * spline_t::radius + 1][2 * spline_t::radius + 1] =
            //     {};
            // for (int k = k_0; k <= k_1; k++) {
            //   value_t sz0 = interp(-x3 + k);
            //   value_t sz1 = interp(-new_x3 + k);

            //   value_t djy[2 * spline_t::radius + 1] = {};
            //   for (int j = j_0; j <= j_1; j++) {
            //     value_t sy0 = interp(-x2 + j);
            //     value_t sy1 = interp(-new_x2 + j);

            //     value_t djx = 0.0f;
            //     for (int i = i_0; i <= i_1; i++) {
            //       value_t sx0 = interp(-x1 + i);
            //       value_t sx1 = interp(-new_x1 + i);

            //       // j1 is movement in x1
            //       auto offset = idx.inc_x(i).inc_y(j).inc_z(k);
            //       djx += movement3d(sy0, sy1, sz0, sz1, sx0, sx1);
            //       if (math::abs(djx) > TINY)
            //         atomicAdd(&J[0][offset],
            //                   -weight * djx * grid.delta[0] / dt);
            //       // Logger::print_debug("J0 is {}", (*J)[0][offset]);

            //       // j2 is movement in x2
            //       djy[i - i_0] += movement3d(sz0, sz1, sx0, sx1, sy0, sy1);
            //       if (math::abs(djy[i - i_0]) > TINY)
            //         atomicAdd(&J[1][offset],
            //                   -weight * djy[i - i_0] * grid.delta[1] / dt);

            //       // j3 is movement in x3
            //       djz[j - j_0][i - i_0] +=
            //           movement3d(sx0, sx1, sy0, sy1, sz0, sz1);
            //       if (math::abs(djz[j - j_0][i - i_0]) > TINY)
            //         atomicAdd(&J[2][offset], -weight * djz[j - j_0][i - i_0] *
            //                                      grid.delta[2] / dt);

            //       // rho is deposited at the final position
            //       if (step % rho_interval == 0) {
            //         atomicAdd(&Rho[sp][offset], weight * sx1 * sy1 * sz1);
            //       }
            //     }
            //   }
            // }
          }
        },
        this->ptc->dev_ptrs(), this->J->get_ptrs(), m_rho_ptrs.dev_ptr(),
        this->m_rho_interval);

    // Modify J with prefactor
    kernel_launch([dt] __device__(auto J) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          J[0][idx] *= grid.delta[0] / dt;
          J[1][idx] *= grid.delta[1] / dt;
          J[2][idx] *= grid.delta[2] / dt;
        }
      }, this->J->get_ptrs());
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();
  }
}

template <typename Conf>
void
ptc_updater_cu<Conf>::move_photons_1d(value_t dt, uint32_t step) {
  auto ph_num = this->ph->number();
  if (ph_num > 0) {
  }
}

template <typename Conf>
void
ptc_updater_cu<Conf>::move_photons_2d(value_t dt, uint32_t step) {
  auto ph_num = this->ph->number();
  if (ph_num > 0) {
  }
}

template <typename Conf>
void
ptc_updater_cu<Conf>::move_photons_3d(value_t dt, uint32_t step) {
  auto ph_num = this->ph->number();
  if (ph_num > 0) {
  }
}

template <typename Conf>
void
ptc_updater_cu<Conf>::clear_guard_cells() {
  auto ext = this->m_grid.extent();
  auto num = this->ptc->number();

  auto clear_guard_cell_knl = [ext] __device__(auto ptc, auto num) {
    auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
    for (auto n : grid_stride_range(0, num)) {
      uint32_t cell = ptc.cell[n];
      if (cell == empty_cell) continue;
      auto idx = typename Conf::idx_t(cell, ext);
      auto pos = idx.get_pos();

      if (!grid.is_in_bound(pos)) ptc.cell[n] = empty_cell;
    }
  };
  kernel_launch(clear_guard_cell_knl, this->ptc->dev_ptrs(), num);
  CudaSafeCall(cudaDeviceSynchronize());

  if (this->ph != nullptr) {
    num = this->ph->number();
    kernel_launch(clear_guard_cell_knl, this->ph->dev_ptrs(), num);
    CudaSafeCall(cudaDeviceSynchronize());
  }
  CudaCheckError();
}

template <typename Conf>
void
ptc_updater_cu<Conf>::sort_particles() {
  this->ptc->sort_by_cell_dev(this->m_grid.extent().size());
  if (this->ph != nullptr) {
    this->ph->sort_by_cell_dev(this->m_grid.extent().size());
  }
}

template <typename Conf>
void
ptc_updater_cu<Conf>::fill_multiplicity(int mult,
                                        typename Conf::value_t weight) {
  auto num = this->ptc->number();
  using idx_t = typename Conf::idx_t;

  kernel_launch(
      [num, mult, weight] __device__(auto ptc, auto states) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        cuda_rng_t rng(&states[id]);
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = idx_t(n, ext);
          auto pos = idx.get_pos();
          if (grid.is_in_bound(pos)) {
            for (int i = 0; i < mult; i++) {
              uint32_t offset = num + idx.linear * mult * 2 + i * 2;

              ptc.x1[offset] = ptc.x1[offset + 1] = rng();
              ptc.x2[offset] = ptc.x2[offset + 1] = rng();
              ptc.x3[offset] = ptc.x3[offset + 1] = rng();
              ptc.p1[offset] = ptc.p1[offset + 1] = 0.0;
              ptc.p2[offset] = ptc.p2[offset + 1] = 0.0;
              ptc.p3[offset] = ptc.p3[offset + 1] = 0.0;
              ptc.E[offset] = ptc.E[offset + 1] = 1.0;
              ptc.cell[offset] = ptc.cell[offset + 1] = idx.linear;
              ptc.weight[offset] = ptc.weight[offset + 1] = weight;
              ptc.flag[offset] = set_ptc_type_flag(flag_or(PtcFlag::primary),
                                                   PtcType::electron);
              ptc.flag[offset + 1] = set_ptc_type_flag(
                  flag_or(PtcFlag::primary), PtcType::positron);
            }
          }
        }
      },
      this->ptc->dev_ptrs(), m_rand_states->states());
  CudaSafeCall(cudaDeviceSynchronize());
  this->ptc->set_num(num + mult * 2 * this->m_grid.extent().size());
}

template <typename Conf>
void
ptc_updater_cu<Conf>::filter_field(vector_field<Conf>& f, int comp) {
  if (this->m_comm != nullptr) {
    filter<Conf>(*(this->jtmp), f[comp],
                 this->m_comm->domain_info().is_boundary);
  } else {
    // bool is_boundary[4] = {true, true, true, true};
    vec_t<bool, 4> is_boundary(true, true, true, true);
    filter<Conf>(*(this->jtmp), f[comp], is_boundary);
  }
}

template <typename Conf>
void
ptc_updater_cu<Conf>::filter_field(scalar_field<Conf>& f) {
  if (this->m_comm != nullptr) {
    filter<Conf>(*(this->jtmp), f[0], this->m_comm->domain_info().is_boundary);
  } else {
    // bool is_boundary[4] = {true, true, true, true};
    vec_t<bool, 4> is_boundary(true, true, true, true);
    filter<Conf>(*(this->jtmp), f[0], is_boundary);
  }
}

#include "ptc_updater_cu_impl.hpp"

INSTANTIATE_WITH_CONFIG(ptc_updater_cu);

}  // namespace Aperture
