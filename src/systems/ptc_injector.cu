#include "framework/config.h"
#include "framework/environment.h"
#include "ptc_injector.h"
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
compute_sigma(scalar_field<Conf>& sigma,
              multi_array<int, Conf::dim>& num_per_cell, particle_data_t& ptc,
              const vector_field<Conf>& B, const grid_t<Conf>& grid,
              typename Conf::value_t target_sigma,
              curandState* states) {
  using value_t = typename Conf::value_t;
  auto ext = grid.extent();
  auto num = ptc.number();
  sigma.init();
  num_per_cell.assign_dev(0);

  // Solve the case for curvilinear grid
  const grid_curv_t<Conf>* grid_curv =
      dynamic_cast<const grid_curv_t<Conf>*>(&grid);
  typename Conf::ndptr_const_t dv_ptr;

  if (grid_curv != nullptr) {
    dv_ptr = grid_curv->get_grid_ptrs().dV;
  }
  // First deposit gamma * n onto the grid
  kernel_launch(
      [ext, num] __device__(auto sigma, auto ptc) {
        // Loop over particles and deposit their (relativistic) density onto the
        // grid
        for (auto n : grid_stride_range(0, num)) {
          auto cell = ptc.cell[n];
          if (cell == empty_cell) continue;

          value_t gamma = ptc.E[n];
          auto flag = ptc.flag[n];
          auto sp = get_ptc_type(flag);
          value_t weight = square(dev_charges[sp]) * ptc.weight[n] / dev_masses[sp];

          atomicAdd(&sigma[cell], gamma * weight);
        }
      },
      sigma.get_ptr(), ptc.get_dev_ptrs());
  CudaCheckError();

  // Then compute sigma = B^2/(gamma * n) using the results computed above.
  // Afterwards, compute the number of particles to inject at each cell using
  // sigma and target_sigma
  kernel_launch(
      [target_sigma] __device__(auto sigma, auto b, auto num_per_cell,
                                auto dv_ptr, auto states) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        auto interp = lerp<Conf::dim>{};
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        cuda_rng_t rng(&states[id]);
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = typename Conf::idx_t(n, ext);
          auto pos = idx.get_pos();

          if (grid.is_in_bound(pos)) {
            value_t B1 = interp(b[0], idx, stagger_t(0b001), stagger_t(0b000));
            value_t B2 = interp(b[1], idx, stagger_t(0b010), stagger_t(0b000));
            value_t B3 = interp(b[2], idx, stagger_t(0b100), stagger_t(0b000));
            value_t B_sqr = B1 * B1 + B2 * B2 + B3 * B3;
            value_t dv = 1.0f;
            if (dv_ptr != nullptr)
              dv = interp(dv_ptr, idx, stagger_t(0b111), stagger_t(0b000));

            value_t s = sigma[idx];
            if (s < TINY) s = TINY;
            s = B_sqr * dv / s;
            sigma[idx] = s;

            value_t r = grid_sph_t<Conf>::radius(grid.template pos<0>(pos[0], false));
            // if (pos[0] == 5 && pos[1] == 256)
            //   printf("B_sqr is %f, s is %f\n", B_sqr, s);
            if (s > target_sigma / cube(r)) {
              // value_t th = grid_sph_t<Conf>::theta(grid.template pos<1>(pos[1], false));
              // value_t ds = B_sqr * (cube(r) / target_sigma - 1.0f / s) * dv /
              //     std::abs(dev_charges[0]) / sin(th);
              if (rng() < 0.01f / r)
                num_per_cell[idx] = 1;
            }
          } else {
            num_per_cell[idx] = 0;
          }
        }
      },
      sigma.get_ptr(), B.get_ptrs(), num_per_cell.get_ptr(), dv_ptr, states);
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
  kernel_launch([ptc_num] __device__ (auto ptc, auto num_per_cell, auto cum_num,
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
          ptc.weight[offset] = ptc.weight[offset + 1] = sin(th),
          // ptc.weight[offset] = ptc.weight[offset + 1] = 1.0f;
          ptc.flag[offset] = set_ptc_type_flag(0, PtcType::electron);
          ptc.flag[offset + 1] = set_ptc_type_flag(0, PtcType::positron);
        }
      }
    }, ptc.get_dev_ptrs(), num_per_cell.get_const_ptr(), cum_num_per_cell.get_const_ptr(),
                states);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

// template <typename Conf>
// void
// count_ptc_injected(buffer<int>& num_per_cell, buffer<int>& cum_num_per_cell,
//                    const scalar_field<Conf>& sigma,
//                    typename Conf::value_t target_sigma,
//                    const grid_t<Conf>& grid) {
//   num_per_cell.assign_dev(0);
//   cum_num_per_cell.assign_dev(0);
//   auto ext = grid.extent();
//   const grid_curv_t<Conf>* grid_curv =
//       dynamic_cast<const grid_curv_t<Conf>*>(&grid);
//   typename Conf::ndptr_const_t dv_ptr;

//   if (grid_curv != nullptr) {
//     dv_ptr = grid_curv->get_grid_ptrs().dV;
//   }

//   kernel_launch(
//       [ext, target_sigma] __device__(auto num_per_cell, auto sigma, auto dv) {
//         auto& grid = dev_grid<Conf::dim>();
//         for (auto n : grid_stride_range(0, ext.size())) {
//           auto idx = typename Conf::idx_t(n, ext);
//           auto pos = idx.get_pos();
//           if (grid.is_in_bound(pos)) {
//             if (sigma[idx] > target_sigma) {
//             }
//           }
//         }
//       },
//       num_per_cell.dev_ptr(), sigma.get_ptr(), dv_ptr);
// }

template <typename Conf>
void
ptc_injector_cu<Conf>::init() {
  ptc_injector<Conf>::init();
}

template <typename Conf>
void
ptc_injector_cu<Conf>::register_dependencies() {
  ptc_injector<Conf>::register_dependencies();

  this->m_env.get_data("rand_states", &m_rand_states);

  m_num_per_cell.set_memtype(MemType::host_device);
  m_cum_num_per_cell.set_memtype(MemType::host_device);
  // m_pos_in_array.set_memtype(MemType::host_device);

  m_num_per_cell.resize(this->m_grid.extent());
  m_cum_num_per_cell.resize(this->m_grid.extent());
  // m_posInBlock.resize()
  m_sigma = this->m_env.template register_data<scalar_field<Conf>>(
      "sigma", this->m_grid, field_type::cell_centered, MemType::host_device);
}

template <typename Conf>
void
ptc_injector_cu<Conf>::update(double dt, uint32_t step) {
  // Compute sigma and number of pairs to inject per cells
  compute_sigma(*m_sigma, m_num_per_cell, *(this->ptc), *(this->B),
                this->m_grid, this->m_target_sigma, m_rand_states->states());

  size_t grid_size = this->m_grid.extent().size();
  thrust::device_ptr<int> p_num_per_block(m_num_per_cell.dev_ptr());
  thrust::device_ptr<int> p_cum_num_per_block(m_cum_num_per_cell.dev_ptr());

  thrust::exclusive_scan(p_num_per_block, p_num_per_block + grid_size,
                         p_cum_num_per_block);
  CudaCheckError();
  m_num_per_cell.copy_to_host();
  m_cum_num_per_cell.copy_to_host();
  int new_pairs = 2 * (m_cum_num_per_cell[grid_size - 1] +
                       m_num_per_cell[grid_size - 1]);
  Logger::print_info("{} new pairs are injected in the box!", new_pairs);

  // Use the num_per_cell and cum_num info to inject actual pairs
  inject_pairs(m_num_per_cell, m_cum_num_per_cell, *(this->ptc),
               this->m_grid, m_rand_states->states());
  this->ptc->add_num(new_pairs);
}

template class ptc_injector_cu<Config<2>>;

}  // namespace Aperture
