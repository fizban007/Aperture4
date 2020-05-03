#include "framework/config.h"
#include "helpers/ptc_update_helper.hpp"
#include "ptc_updater_sph.h"
#include "utils/kernel_helper.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
void
process_j_rho(vector_field<Conf>& j,
              typename ptc_updater_cu<Conf>::rho_ptrs_t& rho_ptrs,
              int num_species, const grid_sph_t<Conf>& grid,
              typename Conf::value_t dt) {
  auto ext = grid.extent();
  kernel_launch(
      [dt, num_species, ext] __device__(auto j, auto rho, auto gp) {
        auto& grid = dev_grid<Conf::dim>();
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = typename Conf::idx_t(n, ext);
          auto pos = idx.get_pos();
          if (grid.is_in_bound(pos)) {
            auto w = grid.delta[0] * grid.delta[1] / dt;
            j[0][idx] *= w / gp.Ae[0][idx];
            j[1][idx] *= w / gp.Ae[1][idx];
            j[2][idx] /= gp.dV[idx];
            for (int n = 0; n < num_species; n++) {
              rho[n][idx] /= gp.dV[idx];
            }
          }
          typename Conf::value_t theta = grid.template pos<1>(pos[1], true);
          if (std::abs(theta) < 0.1 * grid.delta[1]) {
            // j[1][idx] = 0.0;
            j[2][idx] = 0.0;
          }
        }
      },
      j.get_ptrs(), rho_ptrs.dev_ptr(), grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
filter(typename Conf::multi_array_t& result, typename Conf::multi_array_t& f,
       const typename Conf::ndptr_const_t& geom_factor,
       const bool is_boundary[4]) {
  kernel_launch(
      [] __device__(auto result, auto f, auto A, auto is_boundary) {
        typedef typename Conf::idx_t idx_t;
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = idx_t(n, ext);
          auto pos = idx.get_pos();
          if (grid.is_in_bound(pos)) {
            int dx_plus = 1, dx_minus = 1, dy_plus = 1, dy_minus = 1;
            if (is_boundary[0] && pos[0] == grid.skirt[0]) dx_minus = 0;
            if (is_boundary[1] && pos[0] == grid.dims[0] - grid.skirt[0] - 1)
              dx_plus = 0;
            if (is_boundary[2] && pos[1] == grid.skirt[1]) dy_minus = 0;
            if (is_boundary[3] && pos[1] == grid.dims[1] - grid.skirt[1] - 1)
              dy_plus = 0;
            result[idx] = 0.25f * f[idx] * A[idx];
            auto idx_px = idx.inc_x(dx_plus);
            auto idx_mx = idx.dec_x(dx_minus);
            auto idx_py = idx.inc_y(dy_plus);
            auto idx_my = idx.dec_y(dy_minus);
            result[idx] += 0.125f * f[idx_px] * A[idx_px];
            result[idx] += 0.125f * f[idx_mx] * A[idx_mx];
            result[idx] += 0.125f * f[idx_py] * A[idx_py];
            result[idx] += 0.125f * f[idx_my] * A[idx_my];
            result[idx] +=
                0.0625f * f[idx_px.inc_y(dy_plus)] * A[idx_px.inc_y(dy_plus)];
            result[idx] +=
                0.0625f * f[idx_px.dec_y(dy_minus)] * A[idx_px.dec_y(dy_minus)];
            result[idx] +=
                0.0625f * f[idx_mx.inc_y(dy_plus)] * A[idx_mx.inc_y(dy_plus)];
            result[idx] +=
                0.0625f * f[idx_mx.dec_y(dy_minus)] * A[idx_mx.dec_y(dy_minus)];
            result[idx] /= A[idx];
          }
        }
      },
      result.get_ptr(), f.get_const_ptr(), geom_factor,
      vec_t<bool, 4>(is_boundary));
  CudaSafeCall(cudaDeviceSynchronize());
  f.copy_from(result);
}

template <typename Conf>
void
ptc_outflow(particle_data_t& ptc, const grid_sph_t<Conf>& grid, int damping_length) {
  auto ptc_num = ptc.number();
  kernel_launch([ptc_num, damping_length] __device__(auto ptc, auto gp) {
      auto& grid = dev_grid<Conf::dim>();
      for (auto n : grid_stride_range(0, ptc_num)) {
        auto c = ptc.cell[n];
        if (c == empty_cell) continue;

        auto idx = typename Conf::idx_t(c, grid.extent());
        auto pos = idx.get_pos();
        auto flag = ptc.flag[n];
        if (check_flag(flag, PtcFlag::ignore_EM)) continue;
        if (pos[0] > grid.dims[0] - damping_length + 2) {
          flag |= bit_or(PtcFlag::ignore_EM);
          ptc.flag[n] = flag;
        }
      }
    }, ptc.get_dev_ptrs(), grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
ptc_updater_sph_cu<Conf>::init() {
  ptc_updater_cu<Conf>::init();

  this->m_env.params().get_value("compactness", m_compactness);
  this->m_env.params().get_value("omega", m_omega);
  this->m_env.params().get_value("damping_length", m_damping_length);
}

template <typename Conf>
void
ptc_updater_sph_cu<Conf>::register_dependencies() {
  ptc_updater_cu<Conf>::register_dependencies();
}

template <typename Conf>
void
ptc_updater_sph_cu<Conf>::move_deposit_2d(double dt, uint32_t step) {
  this->J->init();
  for (auto rho : this->Rho) rho->init();

  auto num = this->ptc->number();
  if (num > 0) {
    auto ext = this->m_grid.extent();

    auto deposit_kernel = [ext, num, dt, step] __device__(
                              auto ptc, auto J, auto Rho, auto data_interval) {
      using spline_t = typename ptc_updater<Conf>::spline_t;
      using idx_t = typename Conf::idx_t;
      using value_t = typename Conf::value_t;
      auto& grid = dev_grid<Conf::dim>();
      // Obtain a local pointer to the shared array
      extern __shared__ char shared_array[];
      value_t* djy = (value_t*)&shared_array[threadIdx.x * sizeof(value_t) *
                                             (2 * spline_t::radius + 1)];
#pragma unroll
      for (int j = 0; j < 2 * spline_t::radius + 1; j++) {
        djy[j] = 0.0;
      }

      for (auto n : grid_stride_range(0, num)) {
        uint32_t cell = ptc.cell[n];
        if (cell == empty_cell) continue;

        auto idx = idx_t(cell, ext);
        auto pos = idx.get_pos();

        // Move particles
        auto x1 = ptc.x1[n], x2 = ptc.x2[n], x3 = ptc.x3[n];
        value_t v1 = ptc.p1[n], v2 = ptc.p2[n], v3 = ptc.p3[n],
                gamma = ptc.E[n];

        value_t r1 = grid.template pos<0>(pos[0], x1);
        // value_t exp_r1 = std::exp(r1);
        value_t radius = grid_sph_t<Conf>::radius(r1);
        value_t r2 = grid.template pos<1>(pos[1], x2);
        value_t theta = grid_sph_t<Conf>::theta(r2);

        // printf("Particle p1: %f, p2: %f, p3: %f, gamma: %f\n", v1, v2, v3,
        // gamma);

        v1 /= gamma;
        v2 /= gamma;
        v3 /= gamma;
        // value_t v3_gr = v3 - beta_phi(exp_r1, r2);
        //
        // step 1: Compute particle movement and update position
        value_t x = radius * std::sin(theta) * std::cos(x3);
        value_t y = radius * std::sin(theta) * std::sin(x3);
        value_t z = radius * std::cos(theta);

        // logsph2cart(v1, v2, v3_gr, r1, r2, old_x3);
        sph2cart(v1, v2, v3, radius, theta, x3);
        x += v1 * dt;
        y += v2 * dt;
        z += v3 * dt;
        // z += alpha_gr(exp_r1) * v3_gr * dt;
        value_t r1p = sqrt(x * x + y * y + z * z);
        value_t r2p = acos(z / r1p);
        value_t r3p = atan2(y, x);

        cart2sph(v1, v2, v3, r1p, r2p, r3p);
        r1p = grid_sph_t<Conf>::from_radius(r1p);
        r2p = grid_sph_t<Conf>::from_theta(r2p);

        ptc.p1[n] = v1 * gamma;
        ptc.p2[n] = v2 * gamma;
        // ptc.p3[n] = (v3_gr + beta_phi(exp_r1p, r2p)) * gamma;
        ptc.p3[n] = v3 * gamma;

        auto new_x1 = x1 + (r1p - r1) * grid.inv_delta[0];
        auto new_x2 = x2 + (r2p - r2) * grid.inv_delta[1];
        int dc1 = std::floor(new_x1);
        int dc2 = std::floor(new_x2);
#ifndef NDEBUG
        if (dc1 > 1 || dc1 < -1 || dc2 > 1 || dc2 < -1)
          printf("----------------- Error: moved more than 1 cell!");
#endif
        // reflect around the axis
        if (pos[1] <= grid.guard[1] ||
            pos[1] >= grid.dims[1] - grid.guard[1] - 1) {
          theta = grid_sph_t<Conf>::theta(r2p);
          if (theta < 0.0f) {
            dc2 += 1;
            new_x2 = 1.0f - new_x2;
            ptc.p2[n] *= -1.0f;
            ptc.p3[n] *= -1.0f;
          }
          if (theta >= M_PI) {
            dc2 -= 1;
            new_x2 = 1.0f - new_x2;
            ptc.p2[n] *= -1.0f;
            ptc.p3[n] *= -1.0f;
          }
        }
        pos[0] += dc1;
        pos[1] += dc2;

        ptc.x1[n] = new_x1 - (Pos_t)dc1;
        ptc.x2[n] = new_x2 - (Pos_t)dc2;
        ptc.x3[n] = r3p;

        ptc.cell[n] = idx_t(pos, ext).linear;

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

//         // Reset djy since it could be nonzero from previous particle
// #pragma unroll
//         for (int j = 0; j < 2 * spline_t::radius + 1; j++) {
//           djy[j] = 0.0;
//         }

        // Scalar djy[2 * spline_t::radius + 1] = {};
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
            atomicAdd(&J[0][offset], -weight * djx);

            // j2 is movement in x2
            djy[i - i_0] += movement2d(sx0, sx1, sy0, sy1);
            atomicAdd(&J[1][offset], -weight * djy[i - i_0]);

            // j3 is simply v3 times rho at center
            atomicAdd(&J[2][offset],
                      weight * v3 * center2d(sx0, sx1, sy0, sy1));

            // rho is deposited at the final position
            if ((step + 1) % data_interval == 0) {
              atomicAdd(&Rho[sp][offset], weight * sx1 * sy1);
            }
          }
        }
      }
    };
    // exec_policy p;
    // configure_grid(p, deposit_kernel, this->ptc->dev_ptrs(),
    //                this->J->get_ptrs(), this->m_rho_ptrs.dev_ptr(),
    //                this->m_data_interval);

    // Logger::print_info(
    //     "deposit kernel: block_size: {}, grid_size: {}, shared_mem: {}",
    //     p.get_block_size(), p.get_grid_size(), p.get_shared_mem_bytes());

    kernel_launch(
        deposit_kernel, this->ptc->dev_ptrs(), this->J->get_ptrs(),
        this->m_rho_ptrs.dev_ptr(), this->m_data_interval);

    auto& grid = dynamic_cast<const grid_sph_t<Conf>&>(this->m_grid);
    process_j_rho(*(this->J), this->m_rho_ptrs, this->m_num_species, grid, dt);

    ptc_outflow(*(this->ptc), grid, m_damping_length);
  }
}

template <typename Conf>
void
ptc_updater_sph_cu<Conf>::filter_field(vector_field<Conf>& f, int comp) {
  auto& grid = dynamic_cast<const grid_sph_t<Conf>&>(this->m_grid);
  if (this->m_comm != nullptr) {
    filter<Conf>(*(this->jtmp), f.at(comp), grid.get_grid_ptrs().Ae[comp],
                 this->m_comm->domain_info().is_boundary);
  } else {
    bool is_boundary[4] = {true, true, true, true};
    filter<Conf>(*(this->jtmp), f.at(comp), grid.get_grid_ptrs().Ae[comp],
                 is_boundary);
  }
}

template <typename Conf>
void
ptc_updater_sph_cu<Conf>::filter_field(scalar_field<Conf>& f) {
  auto& grid = dynamic_cast<const grid_sph_t<Conf>&>(this->m_grid);
  if (this->m_comm != nullptr) {
    filter<Conf>(*(this->jtmp), f.at(0), grid.get_grid_ptrs().dV,
                 this->m_comm->domain_info().is_boundary);
  } else {
    bool is_boundary[4] = {true, true, true, true};
    filter<Conf>(*(this->jtmp), f.at(0), grid.get_grid_ptrs().dV, is_boundary);
  }
}

template class ptc_updater_sph_cu<Config<2>>;

}  // namespace Aperture
