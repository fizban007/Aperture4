#include "framework/config.h"
#include "framework/environment.h"
#include "radiative_transfer.h"
#include "utils/kernel_helper.hpp"
#include <exception>

namespace Aperture {

struct check_emit_photon_t {
  float gamma_thr;

  check_emit_photon_t(float g) : gamma_thr(g) {}

  template <typename Ptc>
  __device__ bool operator()(Ptc& ptc, uint32_t tid, cuda_rng_t& rng) {
    auto gamma = ptc.E[tid];
    return gamma > gamma_thr;
  }
};

template <typename Conf>
struct emit_photon_t {
  float E_secondary;
  float photon_path;
  float r_cutoff;

  emit_photon_t(float Es, float lph, float r_cut) :
      E_secondary(Es), photon_path(lph), r_cutoff(r_cut) {}

  template <typename Ptc, typename Ph>
  __device__ void operator()(Ptc& ptc, uint32_t tid, Ph& ph, uint32_t offset,
                            cuda_rng_t& rng) {
    Scalar p1 = ptc.p1[tid];
    Scalar p2 = ptc.p2[tid];
    Scalar p3 = ptc.p3[tid];
    //   // Scalar gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
    Scalar gamma = ptc.E[tid];
    Scalar pi = std::sqrt(gamma * gamma - 1.0f);

    Scalar u = rng();
    Scalar Eph = 2.5f + u * (E_secondary - 1.0f) * 2.0f;
    Scalar pf = std::sqrt(square(gamma - Eph) - 1.0f);

    ptc.p1[tid] = p1 * pf / pi;
    ptc.p2[tid] = p2 * pf / pi;
    ptc.p3[tid] = p3 * pf / pi;
    ptc.E[tid] = gamma - Eph;

    auto c = ptc.cell[tid];
    auto& grid = dev_grid<Conf::dim>();
    auto idx = typename Conf::idx_t(c, grid.extent());
    auto pos = idx.get_pos();
    Scalar theta = grid.pos<1>(pos[0], ptc.x2[tid]);
    Scalar lph = min(
        10.0f, (1.0f / std::sin(theta) - 1.0f) * photon_path);
    // If photon energy is too low, do not track it, but still
    // subtract its energy as done above
    // if (std::abs(Eph) < dev_params.E_ph_min) continue;
    if (theta < 0.005f || theta > M_PI - 0.005f) return;

    u = rng();
    // Add the new photo
    Scalar path = lph * (0.5f + 0.5f * u);
    // if (path > dev_params.r_cutoff) return;
    // printf("Eph is %f, path is %f\n", Eph, path);
    ph.x1[offset] = ptc.x1[tid];
    ph.x2[offset] = ptc.x2[tid];
    ph.x3[offset] = ptc.x3[tid];
    ph.p1[offset] = Eph * p1 / pi;
    ph.p2[offset] = Eph * p2 / pi;
    ph.p3[offset] = Eph * p3 / pi;
    ph.weight[offset] = ptc.weight[tid];
    ph.path_left[offset] = path;
    ph.cell[offset] = ptc.cell[tid];
  }
};

template <typename Conf>
radiative_transfer_cu<Conf>::radiative_transfer_cu(
    sim_environment& env, const grid_t<Conf>& grid,
    const domain_comm<Conf>* comm)
    : radiative_transfer<Conf>(env, grid, comm) {
  size_t max_ptc_num, max_ph_num;
  this->m_env.params().get_value("max_ptc_num", max_ptc_num);
  this->m_env.params().get_value("max_ph_num", max_ph_num);

  m_num_per_block.resize(m_blocks_per_grid);
  m_cum_num_per_block.resize(m_blocks_per_grid);
  m_pos_in_block.resize(std::max(max_ptc_num, max_ph_num));
}

template <typename Conf>
void
radiative_transfer_cu<Conf>::init() {
  this->m_env.get_data("rand_states", &m_rand_states);
  this->m_env.get_data("particles", &(this->ptc));
}

template <typename Conf>
void
radiative_transfer_cu<Conf>::register_dependencies() {
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
}

template <typename Conf>
void
radiative_transfer_cu<Conf>::emit_photons(double dt) {
  auto ptc_num = this->ptc->number();
  m_pos_in_block.assign_dev(0, ptc_num, 0);
  m_num_per_block.assign_dev(0);
  m_cum_num_per_block.assign_dev(0);

  // Initialize custom checker
  double gamma_thr =
      this->m_env.params().template get_as<double>("gamma_thr", 30.0);
  check_emit_photon_t check_photon_f(gamma_thr);

  // First count number of photons produced
  kernel_launch(
      [ptc_num] __device__(auto ptc, auto ph_count, auto ph_pos,
                           auto ph_produced, auto states, auto check_photon_f) {
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

          if (check_photon_f(ptc, n, rng)) {
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
      m_pos_in_block.dev_ptr(), this->photon_produced->get_ptr(),
      m_rand_states->states(), check_photon_f);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

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
  Logger::print_info("{} photons are produced!", new_photons);

  // Initialize custom checker
  double Es = this->m_env.params().template get_as<double>("E_s", 2.5);
  double lph = this->m_env.params().template get_as<double>("photon_path", 1.0);
  emit_photon_t<Conf> emit_photon_f(Es, lph, 1.0);

  // Then emit the number of photons computed
  auto ph_num = this->ph->number();
  kernel_launch(
      [ptc_num, ph_num] __device__(auto ptc, auto ph, auto ph_pos,
                                   auto ph_count, auto ph_cum, auto states,
                                   auto emit_photon_f) {
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

            emit_photon_f(ptc, n, ph, offset, rng);

            // float u = rng();
            // if (u < dev_params.track_percent) {
            //   data.photons.flag[offset] = bit_or(PhotonFlag::tracked);
            //   data.photons.id[offset] = dev_rank + atomicAdd(&dev_ph_id, 1);
            // }
          }
        }
      },
      this->ptc->dev_ptrs(), this->ph->dev_ptrs(), m_pos_in_block.dev_ptr(),
      m_num_per_block.dev_ptr(), m_cum_num_per_block.dev_ptr(),
      m_rand_states->states(), emit_photon_f);
}

template <typename Conf>
void
radiative_transfer_cu<Conf>::produce_pairs(double dt) {
  m_pos_in_block.assign_dev(0, this->ph->number(), 0);
  m_num_per_block.assign_dev(0);
  m_cum_num_per_block.assign_dev(0);
}

template class radiative_transfer_cu<Config<1>>;
template class radiative_transfer_cu<Config<2>>;
template class radiative_transfer_cu<Config<3>>;

}  // namespace Aperture
