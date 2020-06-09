#include "compute_lorentz_factor.h"
#include "framework/environment.h"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include <memory>

namespace Aperture {

template <typename Conf>
void
compute_lorentz_factor_cu<Conf>::register_data_components() {
  this->register_data_impl(MemType::host_device);
}

template <typename Conf>
void
compute_lorentz_factor_cu<Conf>::init() {
  compute_lorentz_factor<Conf>::init();

  m_nums.resize(this->gamma.size());
  for (auto& p : m_nums) {
    p = std::make_unique<scalar_field<Conf>>(
        this->m_grid, field_type::cell_centered, MemType::device_only);
  }

  // Initialize the gamma and particle number pointers
  m_gamma_ptrs.set_memtype(MemType::host_device);
  m_nums_ptrs.set_memtype(MemType::host_device);
  m_gamma_ptrs.resize(this->gamma.size());
  m_nums_ptrs.resize(this->gamma.size());
  for (int i = 0; i < this->gamma.size(); i++) {
    m_gamma_ptrs[i] = this->gamma[i]->get_ptr();
    m_nums_ptrs[i] = m_nums[i]->get_ptr();
  }
  m_gamma_ptrs.copy_to_device();
  m_nums_ptrs.copy_to_device();
}

template <typename Conf>
void
compute_lorentz_factor_cu<Conf>::update(double dt, uint32_t step) {
  if (step % this->m_data_interval != 0) return;

  // Compute average Lorentz factors of all particles in every cell
  for (auto g : this->gamma) g->init();
  for (auto& p : this->m_nums) p->init();

  auto num = this->ptc->number();
  if (num > 0) {
    kernel_launch([num] __device__(auto ptc, auto gammas, auto nums) {
          auto& grid = dev_grid<Conf::dim>();
          auto ext = grid.extent();
          for (auto n : grid_stride_range(0, num)) {
            uint32_t cell = ptc.cell[n];
            if (cell == empty_cell) return;

            auto idx = typename Conf::idx_t(cell, ext);

            auto g = ptc.E[n];
            auto weight = ptc.weight[n];
            auto flag = ptc.flag[n];
            auto sp = get_ptc_type(flag);

            atomicAdd(&gammas[sp][idx], weight * g);
            atomicAdd(&nums[sp][idx], weight);
          }
      }, this->ptc->dev_ptrs(), this->m_gamma_ptrs.dev_ptr(),
      m_nums_ptrs.dev_ptr());
    CudaCheckError();

    int num_species = this->m_num_species;
    kernel_launch([num_species] __device__(auto gammas, auto nums) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          for (int i = 0; i < num_species; i++) {
            if (nums[i][idx] > TINY) {
              gammas[i][idx] /= nums[i][idx];
            } else {
              gammas[i][idx] = 0.0f;
            }
          }
        }
      }, this->m_gamma_ptrs.dev_ptr(), m_nums_ptrs.dev_ptrs());
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();
  }
}

}  // namespace Aperture
