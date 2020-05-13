#include "framework/config.h"
#include "framework/environment.h"
#include "ptc_injector_mult.h"
#include "systems/grid_curv.h"
#include "systems/grid_sph.h"
#include "utils/interpolation.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace Aperture {

template <typename Conf>
void
compute_inject_num(multi_array<int, Conf::dim>& num_per_cell,
                   const vector_field<Conf>& J,
                   const buffer<typename Conf::ndptr_const_t>& rho_ptrs,
                   const grid_t<Conf>& grid, float target_mult,
                   curandState* states) {
  using value_t = typename Conf::value_t;
  auto ext = grid.extent();
  num_per_cell.assign_dev(0);

  int num_species = rho_ptrs.size();

  // Then compute sigma = B^2/(gamma * n) using the results computed above.
  // Afterwards, compute the number of particles to inject at each cell using
  // sigma and target_sigma
  kernel_launch(
      [target_mult, num_species] __device__(auto j, auto rho,
                                            auto num_per_cell, auto states) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        auto interp = lerp<Conf::dim>{};
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        cuda_rng_t rng(&states[id]);
        for (auto idx :
             grid_stride_range(Conf::idx(0, ext), Conf::idx(ext.size(), ext))) {
          auto pos = idx.get_pos();

          if (grid.is_in_bound(pos)) {
            value_t J1 = interp(j[0], idx, stagger_t(0b110), stagger_t(0b111));
            value_t J2 = interp(j[1], idx, stagger_t(0b101), stagger_t(0b111));
            value_t J3 = interp(j[2], idx, stagger_t(0b011), stagger_t(0b111));
            value_t J = math::sqrt(J1 * J1 + J2 * J2 + J3 * J3);
            value_t n = 0.0f;
            for (int i = 0; i < num_species; i++) {
              n += math::abs(rho[i][idx]);
            }

            value_t mult = n / J;

            if (mult < target_mult) {
              if (rng() < 0.1f) num_per_cell[idx] = 1;
            }
          } else {
            num_per_cell[idx] = 0;
          }
        }
      },
      J.get_ptrs(), rho_ptrs.dev_ptr(), num_per_cell.get_ptr(),
      states);
  CudaCheckError();
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Conf>
void
inject_pairs(const multi_array<int, Conf::dim>& num_per_cell,
             const multi_array<int, Conf::dim>& cum_num_per_cell,
             particle_data_t& ptc, const grid_t<Conf>& grid,
             curandState* states) {
  auto ptc_num = ptc.number();
  kernel_launch(
      [ptc_num] __device__(auto ptc, auto num_per_cell, auto cum_num,
                           auto states) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        cuda_rng_t rng(&states[id]);
        for (auto cell : grid_stride_range(0, ext.size())) {
          auto idx = typename Conf::idx_t(cell, ext);
          auto pos = idx.get_pos();
          for (int i = 0; i < num_per_cell[cell]; i++) {
            int offset = ptc_num + cum_num[cell] * 2 + i * 2;
            ptc.x1[offset] = ptc.x1[offset + 1] = rng();
            ptc.x2[offset] = ptc.x2[offset + 1] = rng();
            ptc.x3[offset] = ptc.x3[offset + 1] = 0.0f;
            Scalar th = grid.template pos<1>(pos[1], ptc.x2[offset]);
            ptc.p1[offset] = ptc.p1[offset + 1] = 0.0f;
            ptc.p2[offset] = ptc.p2[offset + 1] = 0.0f;
            ptc.p3[offset] = ptc.p3[offset + 1] = 0.0f;
            ptc.E[offset] = ptc.E[offset + 1] = 1.0f;
            ptc.cell[offset] = ptc.cell[offset + 1] = cell;
            // ptc.weight[offset] = ptc.weight[offset + 1] = max(0.02,
            //     abs(2.0f * square(cos(th)) - square(sin(th))) * sin(th));
            ptc.weight[offset] = ptc.weight[offset + 1] = sin(th);
            // ptc.weight[offset] = ptc.weight[offset + 1] = 1.0f;
            ptc.flag[offset] = set_ptc_type_flag(0, PtcType::electron);
            ptc.flag[offset + 1] = set_ptc_type_flag(0, PtcType::positron);
          }
        }
      },
      ptc.get_dev_ptrs(), num_per_cell.get_const_ptr(),
      cum_num_per_cell.get_const_ptr(), states);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
ptc_injector_mult<Conf>::init() {
  ptc_injector<Conf>::init();

  this->m_env.get_data("rand_states", &m_rand_states);
  this->m_env.get_data("J", &J);
  int num_species = 2;
  this->m_env.params().get_value("num_species", num_species);
  Rho.resize(num_species);
  for (int i = 0; i < num_species; i++) {
    this->m_env.get_data(std::string("Rho_") + ptc_type_name(i), &Rho[i]);
  }
  m_rho_ptrs.set_memtype(MemType::host_device);
  m_rho_ptrs.resize(num_species);
  for (int i = 0; i < num_species; i++) {
    m_rho_ptrs[i] = Rho[i]->get_ptr();
  }
  m_rho_ptrs.copy_to_device();
}

template <typename Conf>
void
ptc_injector_mult<Conf>::register_dependencies() {
  ptc_injector<Conf>::register_dependencies();

  m_num_per_cell.set_memtype(MemType::host_device);
  m_cum_num_per_cell.set_memtype(MemType::host_device);
  // m_pos_in_array.set_memtype(MemType::host_device);

  m_num_per_cell.resize(this->m_grid.extent());
  m_cum_num_per_cell.resize(this->m_grid.extent());
  // m_posInBlock.resize()
}

template <typename Conf>
void
ptc_injector_mult<Conf>::update(double dt, uint32_t step) {
  // Compute multiplicity and number of pairs to inject per cells
  compute_inject_num<Conf>(m_num_per_cell, *J, m_rho_ptrs, this->m_grid,
                           4.0f, m_rand_states->states());

  size_t grid_size = this->m_grid.extent().size();
  thrust::device_ptr<int> p_num_per_block(m_num_per_cell.dev_ptr());
  thrust::device_ptr<int> p_cum_num_per_block(m_cum_num_per_cell.dev_ptr());

  thrust::exclusive_scan(p_num_per_block, p_num_per_block + grid_size,
                         p_cum_num_per_block);
  CudaCheckError();
  m_num_per_cell.copy_to_host();
  m_cum_num_per_cell.copy_to_host();
  int new_pairs =
      2 * (m_cum_num_per_cell[grid_size - 1] + m_num_per_cell[grid_size - 1]);
  Logger::print_info("{} new pairs are injected in the box!", new_pairs);

  // Use the num_per_cell and cum_num info to inject actual pairs
  inject_pairs(m_num_per_cell, m_cum_num_per_cell, *(this->ptc), this->m_grid,
               m_rand_states->states());
  this->ptc->add_num(new_pairs);
}

template class ptc_injector_mult<Config<2>>;

}  // namespace Aperture
