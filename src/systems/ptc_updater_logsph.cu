#include "framework/config.h"
#include "helpers/ptc_update_helper.hpp"
#include "ptc_updater_logsph.h"
#include "utils/kernel_helper.hpp"

namespace Aperture {

template <typename Conf>
void
process_j_rho(vector_field<Conf>& j,
              typename ptc_updater_cu<Conf>::rho_ptrs_t& rho_ptrs,
              int num_species, const grid_logsph_t<Conf>& grid,
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
          Scalar theta = grid.template pos<1>(pos[1], true);
          if (std::abs(theta) < 0.1 * grid.delta[1]) {
            j[1][idx] = 0.0;
            j[2][idx] = 0.0;
          }
        }
      },
      j.get_ptrs(), rho_ptrs.dev_ptr(), grid.get_grid_ptrs());
}

template <typename Conf>
void
ptc_updater_logsph_cu<Conf>::init() {
  ptc_updater_cu<Conf>::init();

  this->m_env.params().get_value("compactness", m_compactness);
  this->m_env.params().get_value("omega", m_omega);
}

template <typename Conf>
void
ptc_updater_logsph_cu<Conf>::register_dependencies() {
  ptc_updater_cu<Conf>::register_dependencies();
}

template <typename Conf>
void
ptc_updater_logsph_cu<Conf>::move_deposit_2d(double dt, uint32_t step) {
  this->J->init();
  for (auto rho : this->Rho)
    rho->init();

  auto num = this->ptc->number();
  if (num > 0) {
    auto ext = this->m_grid.extent();

    kernel_launch(
        [ext, num, dt, step] __device__(auto ptc, auto J, auto Rho,
                                        auto data_interval) {
          using spline_t = typename ptc_updater<Conf>::spline_t;
          using idx_t = typename Conf::idx_t;
          using value_t = typename Conf::value_t;
          auto& grid = dev_grid<Conf::dim>();

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
            value_t exp_r1 = std::exp(r1);
            value_t r2 = grid.template pos<1>(pos[1], x2);

            // printf("Particle p1: %f, p2: %f, p3: %f, gamma: %f\n", v1, v2, v3, gamma);

            v1 /= gamma;
            v2 /= gamma;
            v3 /= gamma;
            // value_t v3_gr = v3 - beta_phi(exp_r1, r2);
            //
            // step 1: Compute particle movement and update position
            value_t x = exp_r1 * std::sin(r2) * std::cos(x3);
            value_t y = exp_r1 * std::sin(r2) * std::sin(x3);
            value_t z = exp_r1 * std::cos(r2);

            // logsph2cart(v1, v2, v3_gr, r1, r2, old_x3);
            logsph2cart(v1, v2, v3, r1, r2, x3);
            x += v1 * dt;
            y += v2 * dt;
            z += v3 * dt;
            // z += alpha_gr(exp_r1) * v3_gr * dt;
            value_t r1p = sqrt(x * x + y * y + z * z);
            value_t r2p = acos(z / r1p);
            // value_t exp_r1p = r1p;
            r1p = log(r1p);
            value_t r3p = atan2(y, x);

            cart2logsph(v1, v2, v3, r1p, r2p, r3p);
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
              auto theta = grid.template pos<1>(pos[1] + dc2, new_x2);
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

            Scalar djy[2 * spline_t::radius + 1] = {};
            for (int j = j_0; j <= j_1; j++) {
              Scalar sy0 = interp(-x2 + j);
              Scalar sy1 = interp(-new_x2 + j);

              Scalar djx = 0.0f;
              for (int i = i_0; i <= i_1; i++) {
                Scalar sx0 = interp(-x1 + i);
                Scalar sx1 = interp(-new_x1 + i);

                // j1 is movement in x1
                auto offset = idx.inc_x(i).inc_y(j);
                djx += movement2d(sy0, sy1, sx0, sx1);
                atomicAdd(&J[0][offset], -weight * djx);
                // printf("J0 is %f, djx is %f, weight is %f\n", J[0][offset], djx, weight);

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
        },
        this->ptc->dev_ptrs(), this->J->get_ptrs(), this->m_rho_ptrs.dev_ptr(),
        this->m_data_interval);

    auto& grid = dynamic_cast<const grid_logsph_t<Conf>&>(this->m_grid);
    process_j_rho(*(this->J), this->m_rho_ptrs, this->m_num_species, grid, dt);
    CudaSafeCall(cudaDeviceSynchronize());
  }
}

template class ptc_updater_logsph_cu<Config<2>>;

}  // namespace Aperture
