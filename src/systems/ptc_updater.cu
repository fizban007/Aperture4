#include "core/constant_mem_func.h"
#include "core/detail/multi_array_helpers.h"
#include "core/math.hpp"
#include "framework/config.h"
#include "helpers/ptc_update_helper.hpp"
#include "ptc_updater.h"
#include "utils/double_buffer.h"
#include "utils/interpolation.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
void
ptc_updater_cu<Conf>::init() {
  init_dev_charge_mass(this->m_charges, this->m_masses);

  m_rho_ptrs.set_memtype(MemType::host_device);
  m_rho_ptrs.resize(this->m_num_species);
  for (int i = 0; i < this->m_num_species; i++) {
    m_rho_ptrs[i] = this->Rho[i]->get_ptr();
  }
  m_rho_ptrs.copy_to_device();

  // Allocate the tmp array for current filtering
  this->jtmp = std::make_unique<typename Conf::multi_array_t>
      (this->m_grid.extent(), MemType::host_device);

  this->m_env.get_data_optional("photons", &(this->ph));
  this->m_env.get_data_optional("rho_ph", &(this->rho_ph));
}

template <typename Conf>
void
ptc_updater_cu<Conf>::register_dependencies() {
  size_t max_ptc_num = 1000000;
  this->m_env.params().get_value("max_ptc_num", max_ptc_num);
  // Prefer device_only, but can take other possibilities if data is already
  // there
  this->ptc = this->m_env.template register_data<particle_data_t>(
      "particles", max_ptc_num, MemType::device_only);

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
}

// template <typename Conf>
// void
// ptc_updater_cu<Conf>::update(double dt, uint32_t step) {
//   if (this->m_pusher == Pusher::boris) {
//     push<boris_pusher>(dt, true);
//   } else if (this->m_pusher == Pusher::vay) {
//     push<vay_pusher>(dt, true);
//   } else if (this->m_pusher == Pusher::higuera) {
//     push<higuera_pusher>(dt, true);
//   }

// }

template <typename Conf>
void
ptc_updater_cu<Conf>::push_default(double dt) {
  // dispatch according to enum
  if (this->m_pusher == Pusher::boris) {
    push<boris_pusher>(dt);
  } else if (this->m_pusher == Pusher::vay) {
    push<vay_pusher>(dt);
  } else if (this->m_pusher == Pusher::higuera) {
    push<higuera_pusher>(dt);
  }
}

template <typename Conf>
template <typename P>
void
ptc_updater_cu<Conf>::push(double dt) {
  auto num = this->ptc->number();
  auto ext = this->m_grid.extent();
  P pusher;

  auto pusher_kernel = [dt, num, ext] __device__(auto ptrs, auto E, auto B,
                                                 auto pusher) {
    using value_t = typename Conf::value_t;
    for (auto n : grid_stride_range(0, num)) {
      uint32_t cell = ptrs.cell[n];
      if (cell == empty_cell) continue;
      auto idx = E[0].idx_at(cell, ext);
      // auto pos = idx.get_pos();

      auto interp = interpolator<typename Conf::spline_t, Conf::dim>{};
      auto flag = ptrs.flag[n];
      int sp = get_ptc_type(flag);

      value_t qdt_over_2m = (value_t)dt * 0.5f * dev_charges[sp] / dev_masses[sp];

      auto x = vec_t<Pos_t, 3>(ptrs.x1[n], ptrs.x2[n], ptrs.x3[n]);
      //  Grab E & M fields at the particle position
      value_t E1 = interp(E[0], x, idx, stagger_t(0b110));
      value_t E2 = interp(E[1], x, idx, stagger_t(0b101));
      value_t E3 = interp(E[2], x, idx, stagger_t(0b011));
      value_t B1 = interp(B[0], x, idx, stagger_t(0b001));
      value_t B2 = interp(B[1], x, idx, stagger_t(0b010));
      value_t B3 = interp(B[2], x, idx, stagger_t(0b100));

      //  Push particles
      value_t p1 = ptrs.p1[n], p2 = ptrs.p2[n], p3 = ptrs.p3[n],
             gamma = ptrs.E[n];
      if (p1 != p1 || p2 != p2 || p3 != p3) {
        printf(
            "NaN detected in push! p1 is %f, p2 is %f, p3 is %f, gamma "
            "is %f\n",
            p1, p2, p3, gamma);
        asm("trap;");
      }

      if (!check_flag(flag, PtcFlag::ignore_EM)) {
        pusher(p1, p2, p3, gamma, E1, E2, E3, B1, B2, B3, qdt_over_2m,
               (value_t)dt);
      }

      // if (dev_params.rad_cooling_on && sp != (int)ParticleType::ion) {
      //   sync_kill_perp(p1, p2, p3, gamma, B1, B2, B3, E1, E2, E3,
      //                  q_over_m);
      // }
      ptrs.p1[n] = p1;
      ptrs.p2[n] = p2;
      ptrs.p3[n] = p3;
      ptrs.E[n] = gamma;
    }
  };

  if (num > 0) {
    // exec_policy p;
    // configure_grid(p, pusher_kernel, this->ptc->dev_ptrs(), this->E->get_ptrs(),
    //               this->B->get_ptrs(), pusher);
    // Logger::print_info(
    //     "pusher kernel: block_size: {}, grid_size: {}, shared_mem: {}",
    //     p.get_block_size(), p.get_grid_size(), p.get_shared_mem_bytes());

    kernel_launch(pusher_kernel, this->ptc->dev_ptrs(), this->E->get_ptrs(),
                  this->B->get_ptrs(), pusher);
  }
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
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
          for (auto n : grid_stride_range(0, num)) {
            uint32_t cell = ptc.cell[n];
            if (cell == empty_cell) continue;

            auto idx = J[0].idx_at(cell, ext);
            auto pos = idx.get_pos();

            // step 1: Move particles
            auto x1 = ptc.x1[n], x2 = ptc.x2[n], x3 = ptc.x3[n];
            value_t v1 = ptc.p1[n], v2 = ptc.p2[n], v3 = ptc.p3[n],
                   gamma = ptc.E[n];

            v1 /= gamma;
            v2 /= gamma;
            v3 /= gamma;

            auto new_x1 = x1 + (v1 * dt) * dev_grid_1d.inv_delta[0];
            int dc1 = std::floor(new_x1);
            pos[0] += dc1;
            ptc.x1[n] = new_x1 - (Pos_t)dc1;
            ptc.x2[n] = x2 + v2 * dt;
            ptc.x3[n] = x3 + v3 * dt;

            ptc.cell[n] = J[0].get_idx(pos, ext).linear;

            // step 2: Deposit current
            auto flag = ptc.flag[n];
            auto sp = get_ptc_type(flag);
            auto interp = spline_t{};
            if (check_flag(flag, PtcFlag::ignore_current)) continue;
            auto weight = dev_charges[sp] * ptc.weight[n];

            int i_0 = (dc1 == -1 ? -spline_t::radius : 1 - spline_t::radius);
            int i_1 = (dc1 == 1 ? spline_t::radius + 1 : spline_t::radius);
            value_t djx = 0.0f;
            for (int i = i_0; i <= i_1; i++) {
              value_t sx0 = interp(-x1 + i);
              value_t sx1 = interp(-new_x1 + i);

              // j1 is movement in x1
              int offset = i + pos[0] - dc1;
              djx += sx1 - sx0;
              atomicAdd(&J[0][offset], -weight * djx);
              // Logger::print_debug("J0 is {}", (*J)[0][offset]);

              // j2 is simply v2 times rho at center
              value_t val1 = 0.5f * (sx0 + sx1);
              atomicAdd(&J[1][offset], weight * v2 * val1);

              // j3 is simply v3 times rho at center
              atomicAdd(&J[2][offset], weight * v3 * val1);

              // rho is deposited at the final position
              if (step % rho_interval == 0) {
                atomicAdd(&Rho[sp][offset], weight * sx1);
              }
            }
          }
        },
        this->ptc->dev_ptrs(), this->J->get_ptrs(), m_rho_ptrs.dev_ptr(),
        this->m_rho_interval);
  }
}

template <typename Conf>
void
ptc_updater_cu<Conf>::move_deposit_2d(value_t dt, uint32_t step) {
  this->J->init();
  for (auto rho : this->Rho) rho->init();

  auto num = this->ptc->number();
  if (num > 0) {
    auto ext = this->m_grid.extent();

    kernel_launch(
        [ext, num, dt, step] __device__(auto ptc, auto J, auto Rho,
                                        auto rho_interval) {
          using spline_t = typename base_class::spline_t;
          auto& grid = dev_grid<Conf::dim>();
          // Obtain a local pointer to the shared array
          extern __shared__ char shared_array[];
          value_t* djy = (value_t*)&shared_array[threadIdx.x * sizeof(value_t) *
                                                 (2 * spline_t::radius + 1)];
#pragma unroll
          for (int j = 0; j < 2 * spline_t::radius + 1; j++) {
            djy[j] = 0.0f;
          }

          for (auto n : grid_stride_range(0, num)) {
            uint32_t cell = ptc.cell[n];
            if (cell == empty_cell) continue;

            auto idx = typename Conf::idx_t(cell, ext);
            auto pos = idx.get_pos();

            // step 1: Move particles
            auto x1 = ptc.x1[n], x2 = ptc.x2[n], x3 = ptc.x3[n];
            value_t v1 = ptc.p1[n], v2 = ptc.p2[n], v3 = ptc.p3[n],
                   gamma = ptc.E[n];

            v1 /= gamma;
            v2 /= gamma;
            v3 /= gamma;

            auto new_x1 = x1 + (v1 * dt) * grid.inv_delta[0];
            int dc1 = std::floor(new_x1);
            pos[0] += dc1;
            ptc.x1[n] = new_x1 - (Pos_t)dc1;

            auto new_x2 = x2 + (v2 * dt) * grid.inv_delta[1];
            int dc2 = std::floor(new_x2);
            pos[1] += dc2;
            ptc.x2[n] = new_x2 - (Pos_t)dc2;

            ptc.x3[n] = x3 + v3 * dt;

            ptc.cell[n] = J[0].get_idx(pos, ext).linear;

            // step 2: Deposit current
            auto flag = ptc.flag[n];
            auto sp = get_ptc_type(flag);
            auto interp = spline_t{};
            if (check_flag(flag, PtcFlag::ignore_current)) continue;
            auto weight = dev_charges[sp] * ptc.weight[n];

            int j_0 = (dc2 == -1 ? -spline_t::radius : 1 - spline_t::radius);
            int j_1 = (dc2 == 1 ? spline_t::radius + 1 : spline_t::radius);
            int i_0 = (dc1 == -1 ? -spline_t::radius : 1 - spline_t::radius);
            int i_1 = (dc1 == 1 ? spline_t::radius + 1 : spline_t::radius);

            // value_t djy[2 * spline_t::radius + 1] = {};
            for (int j = j_0; j <= j_1; j++) {
              value_t sy0 = interp(-x2 + j);
              value_t sy1 = interp(-new_x2 + j);

              value_t djx = 0.0f;
              for (int i = i_0; i <= i_1; i++) {
                value_t sx0 = interp(-x1 + i);
                value_t sx1 = interp(-new_x1 + i);

                // j1 is movement in x1
                auto offset = idx.inc_x(i).inc_y(j);
                djx += movement2d(sy0, sy1, sx0, sx1);
                if (math::abs(djx) > TINY)
                  atomicAdd(&J[0][offset], -weight * djx);
                // Logger::print_debug("J0 is {}", (*J)[0][offset]);

                // j2 is movement in x2
                djy[i - i_0] += movement2d(sx0, sx1, sy0, sy1);
                if (math::abs(djy[i - i_0]) > TINY)
                  atomicAdd(&J[1][offset], -weight * djy[i - i_0]);

                // j3 is simply v3 times rho at center
                atomicAdd(&J[2][offset],
                          weight * v3 * center2d(sx0, sx1, sy0, sy1));

                // rho is deposited at the final position
                if (step % rho_interval == 0) {
                  atomicAdd(&Rho[sp][offset], weight * sx1 * sy1);
                }
              }
            }
          }
        },
        this->ptc->dev_ptrs(), this->J->get_ptrs(), m_rho_ptrs.dev_ptr(),
        this->m_rho_interval);
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
          auto& grid = dev_grid<Conf::dim>();
          for (auto n : grid_stride_range(0, num)) {
            uint32_t cell = ptc.cell[n];
            if (cell == empty_cell) continue;

            auto idx = typename Conf::idx_t(cell, ext);
            auto pos = idx.get_pos();

            // step 1: Move particles
            auto x1 = ptc.x1[n], x2 = ptc.x2[n], x3 = ptc.x3[n];
            value_t v1 = ptc.p1[n], v2 = ptc.p2[n], v3 = ptc.p3[n],
                   gamma = ptc.E[n];

            v1 /= gamma;
            v2 /= gamma;
            v3 /= gamma;

            auto new_x1 = x1 + (v1 * dt) * grid.inv_delta[0];
            int dc1 = std::floor(new_x1);
            pos[0] += dc1;
            ptc.x1[n] = new_x1 - (Pos_t)dc1;

            auto new_x2 = x2 + (v2 * dt) * grid.inv_delta[1];
            int dc2 = std::floor(new_x2);
            pos[1] += dc2;
            ptc.x2[n] = new_x2 - (Pos_t)dc2;

            auto new_x3 = x3 + (v3 * dt) * grid.inv_delta[2];
            int dc3 = std::floor(new_x3);
            pos[2] += dc3;
            ptc.x3[n] = new_x3 - (Pos_t)dc3;

            ptc.cell[n] = J[0].get_idx(pos, ext).linear;

            // step 2: Deposit current
            auto flag = ptc.flag[n];
            auto sp = get_ptc_type(flag);
            auto interp = spline_t{};
            if (check_flag(flag, PtcFlag::ignore_current)) continue;
            auto weight = dev_charges[sp] * ptc.weight[n];

            int k_0 = (dc3 == -1 ? -spline_t::radius : 1 - spline_t::radius);
            int k_1 = (dc3 == 1 ? spline_t::radius + 1 : spline_t::radius);
            int j_0 = (dc2 == -1 ? -spline_t::radius : 1 - spline_t::radius);
            int j_1 = (dc2 == 1 ? spline_t::radius + 1 : spline_t::radius);
            int i_0 = (dc1 == -1 ? -spline_t::radius : 1 - spline_t::radius);
            int i_1 = (dc1 == 1 ? spline_t::radius + 1 : spline_t::radius);

            value_t djz[2 * spline_t::radius + 1][2 * spline_t::radius + 1] = {};
            for (int k = k_0; k <= k_1; k++) {
              value_t sz0 = interp(-x3 + k);
              value_t sz1 = interp(-new_x3 + k);

              value_t djy[2 * spline_t::radius + 1] = {};
              for (int j = j_0; j <= j_1; j++) {
                value_t sy0 = interp(-x2 + j);
                value_t sy1 = interp(-new_x2 + j);

                value_t djx = 0.0f;
                for (int i = i_0; i <= i_1; i++) {
                  value_t sx0 = interp(-x1 + i);
                  value_t sx1 = interp(-new_x1 + i);

                  // j1 is movement in x1
                  auto offset = idx.inc_x(i).inc_y(j).inc_z(k);
                  djx += movement3d(sy0, sy1, sz0, sz1, sx0, sx1);
                  if (math::abs(djx) > TINY)
                    atomicAdd(&J[0][offset], -weight * djx);
                  // Logger::print_debug("J0 is {}", (*J)[0][offset]);

                  // j2 is movement in x2
                  djy[i - i_0] += movement3d(sz0, sz1, sx0, sx1, sy0, sy1);
                  if (math::abs(djy[i - i_0]) > TINY)
                    atomicAdd(&J[1][offset], -weight * djy[i - i_0]);

                  // j3 is movement in x3
                  djz[j - j_0][i - i_0] +=
                      movement3d(sx0, sx1, sy0, sy1, sz0, sz1);
                  if (math::abs(djz[j - j_0][i - i_0]) > TINY)
                    atomicAdd(&J[2][offset], -weight * djz[j - j_0][i - i_0]);

                  // rho is deposited at the final position
                  if (step % rho_interval == 0) {
                    atomicAdd(&Rho[sp][offset], weight * sx1 * sy1 * sz1);
                  }
                }
              }
            }
          }
        },
        this->ptc->dev_ptrs(), this->J->get_ptrs(), m_rho_ptrs.dev_ptr(),
        this->m_rho_interval);
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();
  }
}

template <typename Conf>
void
ptc_updater_cu<Conf>::move_photons_1d(value_t dt, uint32_t step) {}

template <typename Conf>
void
ptc_updater_cu<Conf>::move_photons_2d(value_t dt, uint32_t step) {}

template <typename Conf>
void
ptc_updater_cu<Conf>::move_photons_3d(value_t dt, uint32_t step) {}

template <typename Conf>
void
ptc_updater_cu<Conf>::clear_guard_cells() {
  auto ext = this->m_grid.extent();
  auto num = this->ptc->number();

  auto clear_guard_cell_knl = [ext] __device__(auto ptc, auto num) {
    auto& grid = dev_grid<Conf::dim>();
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
        auto& grid = dev_grid<Conf::dim>();
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
              ptc.flag[offset] = set_ptc_type_flag(bit_or(PtcFlag::primary),
                                                   PtcType::electron);
              ptc.flag[offset + 1] = set_ptc_type_flag(bit_or(PtcFlag::primary),
                                                       PtcType::positron);
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
ptc_updater_cu<Conf>::filter_field(vector_field<Conf>& f, int comp) {}

template <typename Conf>
void
ptc_updater_cu<Conf>::filter_field(scalar_field<Conf>& f) {}

template class ptc_updater_cu<Config<1>>;
template class ptc_updater_cu<Config<2>>;
template class ptc_updater_cu<Config<3>>;

}  // namespace Aperture
