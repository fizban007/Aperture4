#include "core/constant_mem.h"
#include "core/cuda_control.h"
#include "core/typedefs_and_constants.h"
#include "field_solver_default.h"
#include "framework/config.h"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include "utils/timer.h"

namespace Aperture {

template <typename Conf>
void
field_solver_default<Conf>::update(double dt, uint32_t step) {
  timer::stamp("field_update");
  if (step == 0) {
    update_e(0.5 * dt);
    m_comm.send_guard_cells(*E);
  }

  update_b(dt);

  // Communicate the new B values to guard cells
  m_comm.send_guard_cells(*B);

  update_e(dt);
  // Communicate the new E values to guard cells
  m_comm.send_guard_cells(*E);

  compute_divs();

  CudaSafeCall(cudaDeviceSynchronize());
  timer::show_duration_since_stamp("Field update", "ms",
                                   "field_update");
}

template <>
void
field_solver_default<Config<1>>::update_e(double dt) {
  auto ext = m_grid.extent();

  kernel_launch(
      [dt, ext] __device__(auto E, auto B, auto Bbg, auto J) {
        for (auto n : grid_stride_range(0u, ext.size())) {
          if (dev_grid_1d.is_in_bound(n)) {
            E[0][n] -= dt * J[0][n];

            E[1][n] +=
                dt *
                ((B[2][n - 1] - Bbg[2][n - 1] - B[2][n] + Bbg[2][n]) *
                     dev_grid_1d.inv_delta[0] -
                 J[1][n]);

            E[2][n] +=
                dt *
                ((B[1][n] - Bbg[1][n] - B[1][n - 1] + Bbg[1][n - 1]) *
                     dev_grid_1d.inv_delta[0] -
                 J[2][n]);
          }
        }
      },
      E->get_ptrs(), B->get_ptrs(), B0->get_ptrs(), J->get_ptrs());
}

template <>
void
field_solver_default<Config<2>>::update_e(double dt) {
  auto ext = m_grid.extent();

  kernel_launch(
      [dt, ext] __device__(auto E, auto B, auto Bbg, auto J) {
        for (auto n : grid_stride_range(0u, ext.size())) {
          auto idx = E[0].idx_at(n, ext);
          auto pos = idx.get_pos();
          if (dev_grid_2d.is_in_bound(pos)) {
            // (Curl u)_1 = d2u3 - d3u2
            E[0][idx] +=
                dt * ((B[2][idx] - Bbg[2][idx] - B[2][idx.dec_y()] +
                       Bbg[2][idx.dec_y()]) *
                          dev_grid_2d.inv_delta[1] -
                      J[0][idx]);

            // (Curl u)_2 = d3u1 - d1u3
            E[1][idx] +=
                dt * ((B[2][idx.dec_x()] - Bbg[2][idx.dec_x()] -
                       B[2][idx] + Bbg[2][idx]) *
                          dev_grid_2d.inv_delta[0] -
                      J[1][idx]);

            // (Curl u)_3 = d1u2 - d2u1
            E[2][idx] +=
                dt * ((B[1][idx] - Bbg[1][idx] - B[1][idx.dec_x()] +
                       Bbg[1][idx.dec_x()]) *
                          dev_grid_2d.inv_delta[0] +
                      (B[0][idx.dec_y()] - Bbg[0][idx.dec_y()] -
                       B[0][idx] + Bbg[0][idx]) *
                          dev_grid_2d.inv_delta[1] -
                      J[2][idx]);
          }
        }
      },
      E->get_ptrs(), B->get_ptrs(), B0->get_ptrs(), J->get_ptrs());
}

template <>
void
field_solver_default<Config<3>>::update_e(double dt) {
  auto ext = m_grid.extent();

  kernel_launch(
      [dt, ext] __device__(auto E, auto B, auto Bbg, auto J) {
        for (auto n : grid_stride_range(0u, ext.size())) {
          auto idx = E[0].idx_at(n, ext);
          auto pos = idx.get_pos();
          if (dev_grid_3d.is_in_bound(pos)) {
            // (Curl u)_1 = d2u3 - d3u2
            E[0][idx] +=
                dt * ((B[2][idx] - Bbg[2][idx] - B[2][idx.dec_y()] +
                       Bbg[2][idx.dec_y()]) *
                          dev_grid_3d.inv_delta[1] +
                      (B[1][idx.dec_z()] - Bbg[1][idx.dec_z()] -
                       B[1][idx] + Bbg[1][idx]) *
                          dev_grid_3d.inv_delta[2] -
                      J[0][idx]);

            // (Curl u)_2 = d3u1 - d1u3
            E[1][idx] +=
                dt * ((B[2][idx.dec_x()] - Bbg[2][idx.dec_x()] -
                       B[2][idx] + Bbg[2][idx]) *
                          dev_grid_3d.inv_delta[0] +
                      (B[0][idx] - Bbg[0][idx] - B[0][idx.dec_z()] +
                       Bbg[0][idx.dec_z()]) *
                          dev_grid_3d.inv_delta[2] -
                      J[1][idx]);

            // (Curl u)_3 = d1u2 - d2u1
            E[2][idx] +=
                dt * ((B[1][idx] - Bbg[1][idx] - B[1][idx.dec_x()] +
                       Bbg[1][idx.dec_x()]) *
                          dev_grid_3d.inv_delta[0] +
                      (B[0][idx.dec_y()] - Bbg[0][idx.dec_y()] -
                       B[0][idx] + Bbg[0][idx]) *
                          dev_grid_3d.inv_delta[1] -
                      J[2][idx]);
          }
        }
      },
      E->get_ptrs(), B->get_ptrs(), B0->get_ptrs(), J->get_ptrs());
}

template <>
void
field_solver_default<Config<1>>::update_b(double dt) {
  auto ext = m_grid.extent();

  kernel_launch(
      [dt, ext] __device__(auto B, auto E, auto Ebg) {
        for (auto n : grid_stride_range(0u, ext.size())) {
          if (dev_grid_1d.is_in_bound(n)) {
            // (Curl u)_1 = d2u3 - d3u2
            // b1 does not change in 1d

            // (Curl u)_2 = d3u1 - d1u3
            B[1][n] -=
                dt *
                (E[2][n] - E[2][n + 1] - Ebg[2][n] + Ebg[2][n + 1]) *
                dev_grid_1d.inv_delta[0];

            // (Curl u)_3 = d1u2 - d2u1
            B[2][n] -=
                dt *
                (E[1][n + 1] - E[1][n] - Ebg[1][n + 1] + Ebg[1][n]) *
                dev_grid_1d.inv_delta[0];
          }
        }
      },
      B->get_ptrs(), E->get_ptrs(), E0->get_ptrs());
}

template <>
void
field_solver_default<Config<2>>::update_b(double dt) {
  auto ext = m_grid.extent();

  kernel_launch(
      [dt, ext] __device__(auto B, auto E, auto Ebg) {
        for (auto n : grid_stride_range(0u, ext.size())) {
          auto idx = B[0].idx_at(n, ext);
          auto pos = idx.get_pos();
          if (dev_grid_2d.is_in_bound(pos)) {
            // (Curl u)_1 = d2u3 - d3u2
            B[0][idx] -= dt *
                         (E[2][idx.inc_y()] - E[2][idx] -
                          Ebg[2][idx.inc_y()] + Ebg[2][idx]) *
                         dev_grid_2d.inv_delta[1];

            // (Curl u)_2 = d3u1 - d1u3
            B[1][n] -= dt *
                       (E[2][idx] - E[2][idx.inc_x()] - Ebg[2][idx] +
                        Ebg[2][idx.inc_x()]) *
                       dev_grid_2d.inv_delta[0];

            // (Curl u)_3 = d1u2 - d2u1
            B[2][n] -= dt * ((E[1][idx.inc_x()] - E[1][idx] -
                              Ebg[1][idx.inc_x()] + Ebg[1][idx]) *
                                 dev_grid_2d.inv_delta[0] +
                             (E[0][idx] - E[0][idx.inc_y()] -
                              Ebg[0][idx] + Ebg[0][idx.inc_y()]) *
                                 dev_grid_2d.inv_delta[1]);
          }
        }
      },
      B->get_ptrs(), E->get_ptrs(), E0->get_ptrs());
}

template <>
void
field_solver_default<Config<3>>::update_b(double dt) {
  auto ext = m_grid.extent();

  kernel_launch(
      [dt, ext] __device__(auto B, auto E, auto Ebg) {
        for (auto n : grid_stride_range(0u, ext.size())) {
          auto idx = B[0].idx_at(n, ext);
          auto pos = idx.get_pos();
          if (dev_grid_3d.is_in_bound(pos)) {
            // (Curl u)_1 = d2u3 - d3u2
            B[0][idx] -= dt * ((E[2][idx.inc_y()] - E[2][idx] -
                                Ebg[2][idx.inc_y()] + Ebg[2][idx]) *
                                   dev_grid_3d.inv_delta[1] +
                               (E[1][idx] - E[1][idx.inc_z()] -
                                Ebg[1][idx] + Ebg[1][idx.inc_z()]) *
                                   dev_grid_3d.inv_delta[2]);

            // (Curl u)_2 = d3u1 - d1u3
            B[1][n] -= dt * ((E[2][idx] - E[2][idx.inc_x()] -
                              Ebg[2][idx] + Ebg[2][idx.inc_x()]) *
                                 dev_grid_3d.inv_delta[0] +
                             (E[0][idx.inc_z()] - E[0][idx] -
                              Ebg[0][idx.inc_z()] + Ebg[0][idx]) *
                                 dev_grid_3d.inv_delta[2]);

            // (Curl u)_3 = d1u2 - d2u1
            B[2][n] -= dt * ((E[1][idx.inc_x()] - E[1][idx] -
                              Ebg[1][idx.inc_x()] + Ebg[1][idx]) *
                                 dev_grid_3d.inv_delta[0] +
                             (E[0][idx] - E[0][idx.inc_y()] -
                              Ebg[0][idx] + Ebg[0][idx.inc_y()]) *
                                 dev_grid_3d.inv_delta[1]);
          }
        }
      },
      B->get_ptrs(), E->get_ptrs(), E0->get_ptrs());
}

template <>
void
field_solver_default<Config<1>>::compute_divs() {
  auto ext = m_grid.extent();

  kernel_launch(
      [ext] __device__(auto divE, auto divB, auto E, auto B, auto Ebg,
                       auto Bbg) {
        for (auto n : grid_stride_range(0u, ext.size())) {
          if (dev_grid_1d.is_in_bound(n)) {
            divE[n] =
                (E[0][n] - E[0][n - 1] - Ebg[0][n] + Ebg[0][n - 1]) *
                dev_grid_1d.inv_delta[0];

            divB[n] =
                (B[0][n + 1] - B[0][n] - Bbg[0][n + 1] + Bbg[0][n]) *
                dev_grid_1d.inv_delta[0];
          }
        }
      },
      (*divE)[0].get_ptr(), (*divB)[0].get_ptr(), E->get_ptrs(),
      B->get_ptrs(), E0->get_ptrs(), B0->get_ptrs());
}

template <>
void
field_solver_default<Config<2>>::compute_divs() {
  auto ext = m_grid.extent();

  kernel_launch(
      [ext] __device__(auto divE, auto divB, auto E, auto B, auto Ebg,
                       auto Bbg) {
        for (auto n : grid_stride_range(0u, ext.size())) {
          auto idx = divE.idx_at(n, ext);
          auto pos = idx.get_pos();

          if (dev_grid_2d.is_in_bound(pos)) {
            divE[n] = (E[0][idx] - E[0][idx.dec_x()] - Ebg[0][idx] +
                       Ebg[0][idx.dec_x()]) *
                          dev_grid_2d.inv_delta[0] +
                      (E[1][idx] - E[1][idx.dec_y()] - Ebg[1][idx] +
                       Ebg[1][idx.dec_y()]) *
                          dev_grid_2d.inv_delta[1];

            divB[n] = (B[0][idx.inc_x()] - B[0][idx] -
                       Bbg[0][idx.inc_x()] + Bbg[0][idx]) *
                          dev_grid_2d.inv_delta[0] +
                      (B[1][idx.inc_y()] - B[1][idx] -
                       Bbg[1][idx.inc_y()] + Bbg[1][idx]) *
                          dev_grid_2d.inv_delta[1];
          }
        }
      },
      (*divE)[0].get_ptr(), (*divB)[0].get_ptr(), E->get_ptrs(),
      B->get_ptrs(), E0->get_ptrs(), B0->get_ptrs());
}

template <>
void
field_solver_default<Config<3>>::compute_divs() {
  auto ext = m_grid.extent();

  kernel_launch(
      [ext] __device__(auto divE, auto divB, auto E, auto B, auto Ebg,
                       auto Bbg) {
        for (auto n : grid_stride_range(0u, ext.size())) {
          auto idx = divE.idx_at(n, ext);
          auto pos = idx.get_pos();

          if (dev_grid_3d.is_in_bound(pos)) {
            divE[n] = (E[0][idx] - Ebg[0][idx] - E[0][idx.dec_x()] +
                       Ebg[0][idx.dec_x()]) *
                          dev_grid_3d.inv_delta[0] +
                      (E[1][idx] - Ebg[1][idx] - E[1][idx.dec_y()] +
                       Ebg[1][idx.dec_y()]) *
                          dev_grid_3d.inv_delta[1] +
                      (E[2][idx] - Ebg[2][idx] - E[2][idx.dec_z()] +
                       Ebg[2][idx.dec_z()]) *
                          dev_grid_3d.inv_delta[2];

            divB[n] = (B[0][idx.inc_x()] - Bbg[0][idx.inc_x()] -
                       B[0][idx] + Bbg[0][idx]) *
                          dev_grid_3d.inv_delta[0] +
                      (B[1][idx.inc_y()] - Bbg[1][idx.inc_y()] -
                       B[1][idx] + Bbg[1][idx]) *
                          dev_grid_3d.inv_delta[1] +
                      (B[2][idx.inc_z()] - Bbg[2][idx.inc_z()] -
                       B[2][idx] + Bbg[2][idx]) *
                          dev_grid_3d.inv_delta[2];
          }
        }
      },
      (*divE)[0].get_ptr(), (*divB)[0].get_ptr(), E->get_ptrs(),
      B->get_ptrs(), E0->get_ptrs(), B0->get_ptrs());
}

template class field_solver_default<Config<1>>;
template class field_solver_default<Config<2>>;
template class field_solver_default<Config<3>>;

}  // namespace Aperture
