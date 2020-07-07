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

#ifndef _RADIATIVE_TRANSFER_CU_IMPL_H_
#define _RADIATIVE_TRANSFER_CU_IMPL_H_

#include "core/constant_mem.h"
#include "framework/environment.h"
#include "radiative_transfer_impl.hpp"
#include "utils/kernel_helper.hpp"
#include <exception>

namespace Aperture {

template <typename Conf, typename RadImpl>
radiative_transfer_cu<Conf, RadImpl>::radiative_transfer_cu(
    sim_environment& env, const grid_t<Conf>& grid,
    const domain_comm<Conf>* comm)
    : radiative_transfer_common<Conf>(env, grid, comm) {
  m_rad = std::make_unique<RadImpl>(env);
 
  size_t max_ptc_num, max_ph_num;
  this->m_env.params().get_value("max_ptc_num", max_ptc_num);
  this->m_env.params().get_value("max_ph_num", max_ph_num);

  m_num_per_block.resize(m_blocks_per_grid);
  m_cum_num_per_block.resize(m_blocks_per_grid);
  m_pos_in_block.resize(std::max(max_ptc_num, max_ph_num));
}

template <typename Conf, typename RadImpl>
radiative_transfer_cu<Conf, RadImpl>::~radiative_transfer_cu() {}

template <typename Conf, typename RadImpl>
void
radiative_transfer_cu<Conf, RadImpl>::init() {
  this->m_env.get_data("rand_states", &m_rand_states);
  this->m_env.get_data("particles", &(this->ptc));
}

template <typename Conf, typename RadImpl>
void
radiative_transfer_cu<Conf, RadImpl>::register_data_components() {
  size_t max_ph_num = 10000;
  this->m_env.params().get_value("max_ph_num", max_ph_num);

  this->ph = this->m_env.template register_data<photon_data_t>(
      "photons", max_ph_num, MemType::device_only);
  this->rho_ph = this->m_env.template register_data<scalar_field<Conf>>(
      "Rho_ph", this->m_grid, field_type::vert_centered, MemType::host_device);
  this->photon_produced =
      this->m_env.template register_data<scalar_field<Conf>>(
          "photon_produced", this->m_grid, field_type::vert_centered,
          MemType::host_device);
  this->pair_produced = this->m_env.template register_data<scalar_field<Conf>>(
      "pair_produced", this->m_grid, field_type::vert_centered,
      MemType::host_device);
  this->photon_produced->reset_after_output();
  this->pair_produced->reset_after_output();
}

template <typename Conf, typename RadImpl>
void
radiative_transfer_cu<Conf, RadImpl>::emit_photons(double dt) {
  auto ptc_num = this->ptc->number();
  if (ptc_num == 0) return;

  m_pos_in_block.assign_dev(0, ptc_num, 0);
  m_num_per_block.assign_dev(0);
  m_cum_num_per_block.assign_dev(0);

  Logger::print_debug("Before kernel launch!");
  // First count number of photons produced
  kernel_launch({m_blocks_per_grid, m_threads_per_block},
      [ptc_num] __device__(auto ptc, auto ph_count, auto ph_pos,
                           auto ph_produced, auto states, auto rad) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        cuda_rng_t rng(&states[id]);

        __shared__ int photon_produced;
        if (threadIdx.x == 0) photon_produced = 0;
        __syncthreads();

        for (auto n : grid_stride_range(0, ptc_num)) {
          uint32_t cell = ptc.cell[n];
          // Skip empty particles
          if (cell == empty_cell) continue;
          auto idx = typename Conf::idx_t(cell, ext);
          auto pos = idx.get_pos();

          if (!grid.is_in_bound(pos)) continue;
          auto flag = ptc.flag[n];
          int sp = get_ptc_type(flag);
          if (sp == (int)PtcType::ion) continue;

          if (rad.check_emit_photon(ptc, n, rng)) {
            auto w = ptc.weight[n];

            ph_pos[n] = atomicAdd(&photon_produced, 1) + 1;
            atomicAdd(&ph_produced[idx], w);
          }
        }

        // Record the number of photons produced this block to global array
        if (threadIdx.x == 0) {
          ph_count[blockIdx.x] = photon_produced;
        }
      },
      this->ptc->dev_ptrs(), m_num_per_block.dev_ptr(),
      m_pos_in_block.dev_ptr(), this->photon_produced->dev_ndptr(),
      m_rand_states->states(), *m_rad);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  Logger::print_info("Before photon scan");
  thrust::device_ptr<int> ptrNumPerBlock(m_num_per_block.dev_ptr());
  thrust::device_ptr<int> ptrCumNum(m_cum_num_per_block.dev_ptr());

  // Scan the number of photons produced per block. The result gives
  // the offset for each block
  thrust::exclusive_scan(ptrNumPerBlock, ptrNumPerBlock + m_blocks_per_grid,
                         ptrCumNum);
  CudaCheckError();
  // Logger::print_debug("Scan finished");
  m_cum_num_per_block.copy_to_host();
  m_num_per_block.copy_to_host();
  int new_photons = m_cum_num_per_block[m_blocks_per_grid - 1] +
      m_num_per_block[m_blocks_per_grid - 1];
  this->ph->add_num(new_photons);
  Logger::print_info("{} photons are produced!", new_photons);

  // Then emit the number of photons computed
  auto ph_num = this->ph->number();
  kernel_launch({m_blocks_per_grid, m_threads_per_block},
      [ptc_num, ph_num] __device__(auto ptc, auto ph, auto ph_pos,
                                   auto ph_count, auto ph_cum, auto ph_id, auto states,
                                   auto rad, auto tracked_frac) {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        cuda_rng_t rng(&states[id]);
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();

        for (auto n : grid_stride_range(0, ptc_num)) {
          int pos_in_block = ph_pos[n] - 1;
          uint32_t cell = ptc.cell[n];
          if (pos_in_block > -1 && cell != empty_cell) {
            auto idx = typename Conf::idx_t(cell, ext);
            auto pos = idx.get_pos();
            if (!grid.is_in_bound(pos)) continue;
            size_t start_pos = ph_cum[blockIdx.x];
            size_t offset = ph_num + start_pos + pos_in_block;

            rad.emit_photon(ptc, n, ph, offset, rng);

            float u = rng();
            if (u < tracked_frac) {
              ph.flag[offset] = flag_or(PhFlag::tracked);
              // ph.id[offset] = dev_rank + atomicAdd(&dev_ph_id, 1);
              ph.id[offset] = dev_rank + atomicAdd(ph_id, 1);
            }
          }
        }
      },
      this->ptc->dev_ptrs(), this->ph->dev_ptrs(), m_pos_in_block.dev_ptr(),
      m_num_per_block.dev_ptr(), m_cum_num_per_block.dev_ptr(),
      this->ph->ptc_id().dev_ptr(),
      m_rand_states->states(), *m_rad, this->m_tracked_fraction);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf, typename RadImpl>
void
radiative_transfer_cu<Conf, RadImpl>::produce_pairs(double dt) {
  auto ph_num = this->ph->number();
  if (ph_num == 0) return;

  m_pos_in_block.assign_dev(0, ph_num, 0);
  m_num_per_block.assign_dev(0);
  m_cum_num_per_block.assign_dev(0);

  // First count number of pairs produced
  kernel_launch({m_blocks_per_grid, m_threads_per_block},
      [ph_num] __device__(auto ph, auto pair_count, auto pair_pos,
                           auto pair_produced, auto states, auto rad) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        cuda_rng_t rng(&states[id]);

        __shared__ int pair_produced_this_block;
        if (threadIdx.x == 0) pair_produced_this_block = 0;
        __syncthreads();

        for (auto n : grid_stride_range(0, ph_num)) {
          uint32_t cell = ph.cell[n];
          // Skip empty particles
          if (cell == empty_cell) continue;
          auto idx = typename Conf::idx_t(cell, ext);
          auto pos = idx.get_pos();

          if (!grid.is_in_bound(pos)) continue;

          if (rad.check_produce_pair(ph, n, rng)) {
            auto w = ph.weight[n];

            pair_pos[n] = atomicAdd(&pair_produced_this_block, 1) + 1;
            atomicAdd(&pair_produced[idx], w);
          }
        }

        // Record the number of photons produced this block to global array
        if (threadIdx.x == 0) {
          pair_count[blockIdx.x] = pair_produced_this_block;
        }
      },
      this->ph->dev_ptrs(), m_num_per_block.dev_ptr(),
      m_pos_in_block.dev_ptr(), this->pair_produced->dev_ndptr(),
      m_rand_states->states(), *m_rad);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  Logger::print_debug("Before pairs scan");
  thrust::device_ptr<int> ptrNumPerBlock(m_num_per_block.dev_ptr());
  thrust::device_ptr<int> ptrCumNum(m_cum_num_per_block.dev_ptr());

  // Scan the number of photons produced per block. The result gives
  // the offset for each block
  thrust::exclusive_scan(ptrNumPerBlock, ptrNumPerBlock + m_blocks_per_grid,
                         ptrCumNum);
  CudaCheckError();
  // Logger::print_debug("Scan finished");
  m_cum_num_per_block.copy_to_host();
  m_num_per_block.copy_to_host();
  int new_pairs = m_cum_num_per_block[m_blocks_per_grid - 1] +
      m_num_per_block[m_blocks_per_grid - 1];
  this->ptc->add_num(new_pairs * 2);
  Logger::print_info("{} pairs are produced!", new_pairs);

  // Then emit the number of photons computed
  auto ptc_num = this->ptc->number();
  kernel_launch({m_blocks_per_grid, m_threads_per_block},
      [ptc_num, ph_num] __device__(auto ph, auto ptc, auto pair_pos,
                                   auto pair_count, auto pair_cum, auto ptc_id,
                                   auto states, auto rad, auto tracked_frac) {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        cuda_rng_t rng(&states[id]);
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();

        for (auto n : grid_stride_range(0, ph_num)) {
          int pos_in_block = pair_pos[n] - 1;
          uint32_t cell = ph.cell[n];
          if (pos_in_block > -1 && cell != empty_cell) {
            auto idx = typename Conf::idx_t(cell, ext);
            auto pos = idx.get_pos();
            if (!grid.is_in_bound(pos)) continue;
            size_t start_pos = pair_cum[blockIdx.x];
            size_t offset = ptc_num + (start_pos + pos_in_block) * 2;

            // rad.emit_photon(ptc, n, ph, offset, rng);
            rad.produce_pair(ph, n, ptc, offset, rng);

            float u = rng();
            if (u < tracked_frac) {
              set_flag(ptc.flag[offset], PtcFlag::tracked);
              set_flag(ptc.flag[offset + 1], PtcFlag::tracked);
              // ptc.id[offset] = dev_rank + atomicAdd(&dev_ptc_id, 1);
              // ptc.id[offset + 1] = dev_rank + atomicAdd(&dev_ptc_id, 1);
              ptc.id[offset] = dev_rank + atomicAdd(ptc_id, 1);
              ptc.id[offset + 1] = dev_rank + atomicAdd(ptc_id, 1);
            }
          }
        }
      },
      this->ph->dev_ptrs(), this->ptc->dev_ptrs(), m_pos_in_block.dev_ptr(),
      m_num_per_block.dev_ptr(), m_cum_num_per_block.dev_ptr(),
      this->ptc->ptc_id().dev_ptr(),
      m_rand_states->states(), *m_rad, this->m_tracked_fraction);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

}

}  // namespace Aperture


#endif  // _RADIATIVE_TRANSFER_CU_IMPL_H_
