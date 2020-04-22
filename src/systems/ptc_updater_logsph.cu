#include "ptc_updater_logsph.h"
#include "helpers/ptc_update_helper.hpp"

namespace Aperture {

template <typename Conf>
void
ptc_updater_logsph_cu<Conf>::init() {
  ptc_updater_cu<Conf>::init();
}

template <typename Conf>
void
ptc_updater_logsph_cu<Conf>::register_dependencies() {
  ptc_updater_cu<Conf>::register_dependencies();
}

template <typename Conf>
void
ptc_updater_logsph_cu<Conf>::move_deposit_2d(double dt, uint32_t step) {
  auto num = this->ptc->number();
  if (num > 0) {
    auto ext = this->m_grid.extent();

    kernel_launch(
        [ext, num, dt, step] __device__(auto ptc, auto J, auto Rho,
                                        auto data_interval) {
          using spline_t = typename ptc_updater<Conf>::spline_t;
          for (auto n : grid_stride_range(0, num)) {
            uint32_t cell = ptc.cell[n];
            if (cell == empty_cell) continue;

            auto idx = J[0].idx_at(cell, ext);
            auto pos = idx.get_pos();

            // step 1: Move particles
            auto x1 = ptc.x1[n], x2 = ptc.x2[n], x3 = ptc.x3[n];
            Scalar v1 = ptc.p1[n], v2 = ptc.p2[n], v3 = ptc.p3[n],
                   gamma = ptc.E[n];

            v1 /= gamma;
            v2 /= gamma;
            v3 /= gamma;

            auto new_x1 = x1 + (v1 * dt) * dev_grid_2d.inv_delta[0];
            int dc1 = std::floor(new_x1);
            pos[0] += dc1;
            ptc.x1[n] = new_x1 - (Pos_t)dc1;

            auto new_x2 = x2 + (v2 * dt) * dev_grid_2d.inv_delta[1];
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
                // Logger::print_debug("J0 is {}", (*J)[0][offset]);

                // j2 is movement in x2
                djy[i - i_0] += movement2d(sx0, sx1, sy0, sy1);
                atomicAdd(&J[1][offset], -weight * djy[i - i_0]);

                // j3 is simply v3 times rho at center
                atomicAdd(&J[2][offset], weight * v3 *
                          center2d(sx0, sx1, sy0, sy1));

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
  }
}

}
