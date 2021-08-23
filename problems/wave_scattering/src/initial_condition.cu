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

#include "core/math.hpp"
#include "core/random.h"
#include "data/curand_states.h"
#include "data/fields.h"
#include "data/particle_data.h"
#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "utils/kernel_helper.hpp"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace {

using namespace Aperture;

template <typename value_t>
struct fast_wave_solution {
  value_t sinth = 0.0;
  value_t costh;
  value_t lambda = 100;
  value_t omega = 1.0 / lambda;
  value_t a0 = 30;
  value_t B0 = 10 * omega;

  value_t x0 = 0.0;
  value_t y0 = 0.0;
  value_t length = 4.0f;
  value_t smooth_width = 0.1f;

  HD_INLINE value_t xi(value_t x, value_t y) const {
    return x * sinth + y * costh;
  }

  HD_INLINE value_t eta(value_t x, value_t y) const {
    return x * costh - y * sinth;
  }

  HOST_DEVICE fast_wave_solution(value_t sinth_, value_t lambda_,
                                 value_t x0_, value_t y0_, value_t length_,
                                 value_t a0_)
      : sinth(sinth_),
        lambda(lambda_),
        x0(x0_),
        y0(y0_),
        length(length_),
        a0(a0_) {
    costh = math::sqrt(1.0f - sinth * sinth);
    omega = 1.0 / lambda;
    // delta_eta = delta_y * sinth;
    // eta0 = eta(0.0, y0);
  }

  HD_INLINE value_t wave_arg(value_t t, value_t x, value_t y) const {
    return 2.0 * M_PI * (x - t) /
           lambda;
  }

  HD_INLINE value_t wave_arg_clamped(value_t t, value_t x, value_t y) const {
    return 2.0 * M_PI * clamp<value_t>((x - t) / lambda, 0.0f, length);
  }

  HD_INLINE value_t wave_profile(value_t x) const {
    // Convert x into a number between 0 and 1
    value_t arg = clamp<value_t>(x / (2.0 * M_PI) / length, 0.0f, 1.0f);
    value_t prof = 0.0f;
    if (arg < smooth_width) {
      prof = square(math::sin(arg * M_PI / (smooth_width * 2.0f)));
    } else if (arg > (1.0f - smooth_width)) {
      prof = square(
          math::sin((arg - 1.0f + smooth_width) * M_PI / (smooth_width * 2.0f) +
                    0.5f * M_PI));
    } else {
      prof = 1.0f;
    }
    return math::sin(x) * prof;
    // return math::sin(x) * square(math::sin(0.5 * x / length));
  }

  HD_INLINE value_t Bz(value_t t, value_t x, value_t y) const {
    return a0 * omega * wave_profile(wave_arg_clamped(t, x, y));
  }

  HD_INLINE value_t Ey(value_t t, value_t x, value_t y) const {
    return Bz(t, x, y);
  }

};

}  // namespace

namespace Aperture {

template <typename Conf>
void
initial_condition_wave(vector_field<Conf> &B,
                       vector_field<Conf> &E, vector_field<Conf> &B0,
                       particle_data_t &ptc, rng_states_t &states, int mult,
                       Scalar weight) {
  Scalar weight_enhance_factor = 1.0f;
  Scalar sinth = sim_env().params().get_as<double>("muB", 0.1);
  Scalar Bp = sim_env().params().get_as<double>("Bp", 5000.0);
  Scalar q_e = sim_env().params().get_as<double>("q_e", 1.0);
  q_e *= weight_enhance_factor;
  Scalar Bwave_factor = sim_env().params().get_as<double>("waveB", 0.1);
  Scalar Bwave = Bwave_factor * Bp;
  int mult_wave = 1;

  fast_wave_solution<Scalar> wave(sinth, 1.0, 0.05, 4.0, 2.0, Bwave);

  B0.set_values(
      0, [Bp, sinth](Scalar x, Scalar y, Scalar z) { return Bp * sinth; });
  B0.set_values(1, [Bp, sinth](Scalar x, Scalar y, Scalar z) {
    return Bp * math::sqrt(1.0 - sinth * sinth);
  });
  B.set_values(
      2, [wave](Scalar x, Scalar y, Scalar z) { return wave.Bz(0.0, x, y); });
  E.set_values(
      1, [wave](Scalar x, Scalar y, Scalar z) { return wave.Ey(0.0, x, y); });

  auto num = ptc.number();

  // Compute injection number per cell
  auto ext = B.grid().extent();
  multi_array<int, Conf::dim> num_per_cell(ext, MemType::host_device);
  multi_array<int, Conf::dim> cum_num_per_cell(ext, MemType::host_device);

  num_per_cell.assign_dev(0);
  cum_num_per_cell.assign_dev(0);

  // compute_ptc_per_cell<Conf>(wave, num_per_cell, q_e, mult, mult_wave);

  thrust::device_ptr<int> p_num_per_cell(num_per_cell.dev_ptr());
  thrust::device_ptr<int> p_cum_num_per_cell(cum_num_per_cell.dev_ptr());

  thrust::exclusive_scan(p_num_per_cell, p_num_per_cell + ext.size(),
                         p_cum_num_per_cell);
  CudaCheckError();
  num_per_cell.copy_to_host();
  cum_num_per_cell.copy_to_host();
  int new_particles =
      (cum_num_per_cell[ext.size() - 1] + num_per_cell[ext.size() - 1]);
  Logger::print_info("Initializing {} particles", new_particles);

  // Initialize the particles
  num = ptc.number();
  kernel_launch(
      [mult, mult_wave, num, q_e, wave, Bp, weight_enhance_factor] __device__(
          auto ptc, auto states, auto w, auto num_per_cell,
          auto cum_num_per_cell) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        // int id = threadIdx.x + blockIdx.x * blockDim.x;
        // cuda_rng_t rng(&states[id]);
        rng_t rng(states);
        for (auto cell : grid_stride_range(0, ext.size())) {
          auto idx = Conf::idx(cell, ext);
          // auto pos = idx.get_pos();
          auto pos = get_pos(idx, ext);
          // auto idx_row = idx_row_major_t<Conf::dim>(pos, ext);
          if (grid.is_in_bound(pos)) {
            for (int i = 0; i < num_per_cell[idx]; i++) {
              uint32_t offset = num + cum_num_per_cell[idx] + i;
              // uint32_t offset = num + idx_row.linear * mult * 2 + i * 2;
              ptc.x1[offset] = rng.uniform<float>();
              ptc.x2[offset] = rng.uniform<float>();
              ptc.x3[offset] = 0.0f;

              ptc.cell[offset] = cell;
              Scalar x = grid.template pos<0>(pos[0], ptc.x1[offset]);

              if (i < mult * 2) {
                // if (x < 0.4 * grid.sizes[0]) {
                if (x < 1.0 * grid.sizes[0]) {
                  // Scalar weight = w * (1.0f + 29.0f * (1.0f - x /
                  // grid.sizes[0]));
                  Scalar weight = w;

                  // if (x > 0.4 * grid.sizes[0]) weight *= 0.02;
                  ptc.p1[offset] = 0.0f;
                  ptc.p2[offset] = 0.0f;
                  ptc.p3[offset] = 0.0f;
                  ptc.E[offset] = 1.0f;
                  ptc.weight[offset] = weight;
                  ptc.flag[offset] = set_ptc_type_flag(
                      flag_or(PtcFlag::primary),
                      ((i % 2 == 0) ? PtcType::electron : PtcType::positron));
                }
              } else {
                Scalar x = grid.template pos<0>(pos, 0.0f);
                Scalar y = grid.template pos<1>(pos, 0.0f);
                // auto width_arg = wave.width_arg(x, y);
                auto wave_arg = wave.wave_arg(0.0f, x, y);
                auto rho = wave.Rho(0.0f, x, y);
                int num = floor(math::abs(rho) / q_e);

                Scalar v;
                if (i < mult * 2 + mult_wave * num) {
                  v = 0.0f;
                } else {
                  v = 1.0f / (mult_wave + 1);
                }
                auto v_d = wave.Bz(0.0f, x, y) /
                           math::sqrt(square(wave.Bz(0.0f, x, y)) + Bp * Bp);
                auto gamma = 1.0f / math::sqrt(1.0f - v * v - v_d * v_d);
                ptc.p1[offset] = v * wave.sinth * gamma;
                ptc.p2[offset] = v * wave.costh * gamma;
                ptc.p3[offset] = v_d * gamma;
                ptc.E[offset] = gamma;
                ptc.weight[offset] =
                    weight_enhance_factor * math::abs(rho) / num / q_e;
                if (i < mult * 2 + mult_wave * num) {
                  ptc.flag[offset] = set_ptc_type_flag(
                      flag_or(PtcFlag::primary, PtcFlag::initial),
                      ((rho < 0.0f) ? PtcType::positron : PtcType::electron));
                } else {
                  ptc.flag[offset] = set_ptc_type_flag(
                      flag_or(PtcFlag::primary, PtcFlag::initial),
                      ((rho < 0.0f) ? PtcType::electron : PtcType::positron));
                }
              }
            }
          }
        }
      },
      ptc.dev_ptrs(), states.states().dev_ptr(), weight, num_per_cell.dev_ndptr(),
      cum_num_per_cell.dev_ndptr());
  CudaSafeCall(cudaDeviceSynchronize());
  ptc.set_num(num + new_particles);
}

template void initial_condition_wave<Config<2>>(
    vector_field<Config<2>> &B,
    vector_field<Config<2>> &E, vector_field<Config<2>> &B0,
    particle_data_t &ptc, rng_states_t &states, int mult, Scalar weight);

}  // namespace Aperture
