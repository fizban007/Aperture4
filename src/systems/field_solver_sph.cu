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

#include "core/multi_array_exp.hpp"
#include "core/ndsubset_dev.hpp"
#include "field_solver_sph.h"
#include "framework/config.h"
#include "framework/environment.h"
// #include "helpers/field_solver_helper_cu.hpp"
#include "utils/double_buffer.h"
#include "utils/kernel_helper.hpp"
#include "utils/timer.h"

namespace Aperture {

namespace {

// template <typename PtrType, typename Idx_t>
template <typename VecType, typename Idx_t>
HD_INLINE Scalar
circ0(const VecType& f, const VecType& dl, const Idx_t& idx,
      const Idx_t& idx_py) {
  return f[2][idx_py] * dl[2][idx_py] - f[2][idx] * dl[2][idx];
}

template <typename VecType, typename Idx_t>
HD_INLINE Scalar
circ1(const VecType& f, const VecType& dl, const Idx_t& idx,
      const Idx_t& idx_px) {
  return f[2][idx] * dl[2][idx] - f[2][idx_px] * dl[2][idx_px];
}

template <typename VecType, typename Idx_t>
HD_INLINE Scalar
circ2(const VecType& f, const VecType& dl, const Idx_t& idx_mx,
      const Idx_t& idx_my, const Idx_t& idx_px, const Idx_t& idx_py) {
  return f[1][idx_px] * dl[1][idx_px] - f[1][idx_mx] * dl[1][idx_mx] +
         f[0][idx_my] * dl[0][idx_my] - f[0][idx_py] * dl[0][idx_py];
}

template <typename Conf>
void
compute_double_circ(vector_field<Conf>& result, const vector_field<Conf>& b,
                    const grid_curv_t<Conf>& grid,
                    typename Conf::value_t coef) {
  auto ext = grid.extent();
  kernel_launch(
      [coef, ext] __device__(auto result, auto b, auto gp) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = typename Conf::idx_t(n, ext);
          // auto pos = idx.get_pos();
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            auto idx_mx = idx.dec_x();
            auto idx_my = idx.dec_y();
            auto idx_px = idx.inc_x();
            auto idx_py = idx.inc_y();
            auto idx_pymx = idx.inc_y().dec_x();
            auto idx_pxmy = idx.inc_x().dec_y();
            result[0][idx] =
                coef *
                (gp.le[2][idx_py] *
                     circ2(b, gp.lb, idx_pymx, idx, idx_py, idx_py) /
                     gp.Ae[2][idx_py] -
                 gp.le[2][idx] * circ2(b, gp.lb, idx_mx, idx_my, idx, idx) /
                     gp.Ae[2][idx]) /
                gp.Ab[0][idx];

            result[1][idx] =
                coef *
                (gp.le[2][idx] * circ2(b, gp.lb, idx_mx, idx_my, idx, idx) /
                     gp.Ae[2][idx] -
                 gp.le[2][idx_px] *
                     circ2(b, gp.lb, idx, idx_pxmy, idx_px, idx_px) /
                     gp.Ae[2][idx_px]) /
                gp.Ab[1][idx];
            // Take care of axis boundary
            Scalar theta = grid.template pos<1>(pos[1], true);
            if (abs(theta) < 0.1 * grid.delta[1]) {
              result[1][idx] = 0.0f;
            }

            result[2][idx] =
                coef *
                (gp.le[0][idx] * circ0(b, gp.lb, idx_my, idx) /
                     gp.Ae[0][idx_py] -
                 gp.le[0][idx_py] * circ0(b, gp.lb, idx, idx_py) /
                     gp.Ae[0][idx] +
                 gp.le[1][idx_px] * circ1(b, gp.lb, idx, idx_px) /
                     gp.Ae[1][idx_px] -
                 gp.le[1][idx] * circ1(b, gp.lb, idx_mx, idx) / gp.Ae[1][idx]) /
                gp.Ab[2][idx];
          }
        }
      },
      result.get_ptrs(), b.get_ptrs(), grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Conf>
void
compute_implicit_rhs(vector_field<Conf>& result, const vector_field<Conf>& e,
                     const vector_field<Conf>& j, const grid_curv_t<Conf>& grid,
                     typename Conf::value_t alpha, typename Conf::value_t beta,
                     typename Conf::value_t dt) {
  auto ext = grid.extent();
  kernel_launch(
      [alpha, beta, dt, ext] __device__(auto result, auto e, auto j, auto gp) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        // gp is short for grid_ptrs
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = result[0].idx_at(n, ext);
          // auto pos = idx.get_pos();
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            auto idx_py = idx.inc_y();
            result[0][idx] += -dt *
                              (circ0(e, gp.le, idx, idx_py) -
                               dt * beta * circ0(j, gp.le, idx, idx_py)) /
                              gp.Ab[0][idx];

            auto idx_px = idx.inc_x();
            result[1][idx] += -dt *
                              (circ1(e, gp.le, idx, idx_px) -
                               dt * beta * circ1(j, gp.le, idx, idx_px)) /
                              gp.Ab[1][idx];

            // Take care of axis boundary
            Scalar theta = grid.template pos<1>(pos[1], true);
            if (abs(theta) < 0.1 * grid.delta[1]) {
              result[1][idx] = 0.0f;
            }

            result[2][idx] +=
                -dt *
                (circ2(e, gp.le, idx, idx, idx_px, idx_py) -
                 dt * beta * circ2(j, gp.le, idx, idx, idx_px, idx_py)) /
                gp.Ab[2][idx];
          }
        }
      },
      result.get_ptrs(), e.get_ptrs(), j.get_ptrs(), grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Conf>
void
compute_e_update_explicit(vector_field<Conf>& result,
                          const vector_field<Conf>& b,
                          const vector_field<Conf>& j,
                          const grid_curv_t<Conf>& grid,
                          typename Conf::value_t dt) {
  auto ext = grid.extent();
  kernel_launch(
      [dt, ext] __device__(auto result, auto b, auto j, auto gp) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        // gp is short for grid_ptrs
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = result[0].idx_at(n, ext);
          // auto pos = idx.get_pos();
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            auto idx_my = idx.dec_y();
            result[0][idx] +=
                dt * (circ0(b, gp.lb, idx_my, idx) / gp.Ae[0][idx] - j[0][idx]);

            auto idx_mx = idx.dec_x();
            result[1][idx] +=
                dt * (circ1(b, gp.lb, idx_mx, idx) / gp.Ae[1][idx] - j[1][idx]);

            result[2][idx] += dt * (circ2(b, gp.lb, idx_mx, idx_my, idx, idx) /
                                        gp.Ae[2][idx] -
                                    j[2][idx]);
            // Take care of axis boundary
            Scalar theta = grid.template pos<1>(pos[1], true);
            if (abs(theta) < 0.1f * grid.delta[1]) {
              result[2][idx] = 0.0f;
            }
            // if (pos[0] == 3 && pos[1] == 256) {
            //   printf("Br is %f, %f, %f; Btheta is %f, %f, %f\n",
            //   b[0][idx_my],
            //          b[0][idx], b[0][idx.inc_y()], b[1][idx_mx], b[1][idx],
            //          b[1][idx.inc_x()]);
            //   printf("Ephi is %f, circ2_b is %f, jphi is %f\n",
            //   result[2][idx],
            //          circ2(b, gp.lb, idx_mx, idx_my, idx, idx), j[2][idx]);
            // }
          }
          // extra work for the theta = pi axis
          auto theta = grid.template pos<1>(pos[1], true);
          if (std::abs(theta - M_PI) < 0.1f * grid.delta[1]) {
            auto idx_my = idx.dec_y();
            result[0][idx] +=
                dt * (circ0(b, gp.lb, idx_my, idx) / gp.Ae[0][idx] - j[0][idx]);
          }
        }
      },
      result.get_ptrs(), b.get_ptrs(), j.get_ptrs(), grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Conf>
void
compute_b_update_explicit(vector_field<Conf>& result,
                          const vector_field<Conf>& e,
                          const grid_curv_t<Conf>& grid,
                          typename Conf::value_t dt) {
  auto ext = grid.extent();
  kernel_launch(
      [dt, ext] __device__(auto result, auto e, auto gp) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        // gp is short for grid_ptrs
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = typename Conf::idx_t(n, ext);
          // auto pos = idx.get_pos();
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            auto idx_py = idx.inc_y();
            result[0][idx] -= dt * circ0(e, gp.le, idx, idx_py) / gp.Ab[0][idx];

            auto idx_px = idx.inc_x();
            result[1][idx] -= dt * circ1(e, gp.le, idx, idx_px) / gp.Ab[1][idx];
            // Take care of axis boundary
            Scalar theta = grid.template pos<1>(pos[1], true);
            if (abs(theta) < 0.1f * grid.delta[1]) {
              result[1][idx] = 0.0f;
            }

            result[2][idx] -=
                dt * circ2(e, gp.le, idx, idx, idx_px, idx_py) / gp.Ab[2][idx];
          }
        }
      },
      result.get_ptrs(), e.get_ptrs(), grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Conf>
void
axis_boundary_e(vector_field<Conf>& e, const grid_curv_t<Conf>& grid) {
  auto ext = grid.extent();
  typedef typename Conf::idx_t idx_t;
  kernel_launch(
      [ext] __device__(auto e) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        for (auto n0 : grid_stride_range(0, grid.dims[0])) {
          auto n1_0 = grid.guard[1];
          auto n1_pi = grid.dims[1] - grid.guard[1];
          if (abs(grid.template pos<1>(n1_0, true)) < 0.1f * grid.delta[1]) {
            // At the theta = 0 axis

            // Set E_phi and B_theta to zero
            auto idx = idx_t(index_t<2>(n0, n1_0), ext);
            e[2][idx] = 0.0f;
            // e[1][idx] = 0.0;
            e[1][idx.dec_y()] = e[1][idx];
            // e[0][idx] = 0.0f;
          }
          // printf("boundary pi at %f\n", grid.template pos<1>(n1_pi, true));
          if (abs(grid.template pos<1>(n1_pi, true) - M_PI) <
              0.1f * grid.delta[1]) {
            // At the theta = pi axis
            auto idx = idx_t(index_t<2>(n0, n1_pi), ext);
            e[2][idx] = 0.0f;
            // e[1][idx] = 0.0;
            e[1][idx] = e[1][idx.dec_y()];
            // e[0][idx] = 0.0f;
          }
        }
      },
      e.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Conf>
void
axis_boundary_b(vector_field<Conf>& b, const grid_curv_t<Conf>& grid) {
  auto ext = grid.extent();
  typedef typename Conf::idx_t idx_t;
  kernel_launch(
      [ext] __device__(auto b) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        for (auto n0 : grid_stride_range(0, grid.dims[0])) {
          auto n1_0 = grid.guard[1];
          auto n1_pi = grid.dims[1] - grid.guard[1];
          if (abs(grid.template pos<1>(n1_0, true)) < 0.1f * grid.delta[1]) {
            // At the theta = 0 axis

            // Set E_phi and B_theta to zero
            auto idx = idx_t(index_t<2>(n0, n1_0), ext);
            b[1][idx] = 0.0;
            b[2][idx.dec_y()] = b[2][idx];
            // b[2][idx.dec_y()] = 0.0;
            b[0][idx.dec_y()] = b[0][idx];
          }
          // printf("boundary pi at %f\n", grid.template pos<1>(n1_pi, true));
          if (abs(grid.template pos<1>(n1_pi, true) - M_PI) <
              0.1f * grid.delta[1]) {
            // At the theta = pi axis
            auto idx = idx_t(index_t<2>(n0, n1_pi), ext);
            b[1][idx] = 0.0;
            b[2][idx] = b[2][idx.dec_y()];
            // b[2][idx] = 0.0;
            b[0][idx] = b[0][idx.dec_y()];
          }
        }
      },
      b.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Conf>
void
damping_boundary(vector_field<Conf>& e, vector_field<Conf>& b,
                 int damping_length, typename Conf::value_t damping_coef) {
  auto ext = e.grid().extent();
  typedef typename Conf::idx_t idx_t;
  typedef typename Conf::value_t value_t;
  kernel_launch(
      [ext, damping_length, damping_coef] __device__(auto e, auto b) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        for (auto n1 : grid_stride_range(0, grid.dims[1])) {
          // for (int i = 0; i < damping_length - grid.guard[0] - 1; i++) {
          for (int i = 0; i < damping_length - 1; i++) {
            int n0 = grid.dims[0] - damping_length + i;
            auto idx = idx_t(index_t<2>(n0, n1), ext);
            value_t lambda =
                1.0f - damping_coef * cube((value_t)i / (damping_length - 1));
            e[0][idx] *= lambda;
            e[1][idx] *= lambda;
            e[2][idx] *= lambda;
            b[1][idx] *= lambda;
            b[2][idx] *= lambda;
          }
        }
      },
      e.get_ptrs(), b.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Conf>
void
compute_divs_cu(scalar_field<Conf>& divE, scalar_field<Conf>& divB,
                const vector_field<Conf>& e, const vector_field<Conf>& b,
                const grid_curv_t<Conf>& grid,
                const bool is_boundary[Conf::dim * 2]) {
  // const vec_t<bool, Conf::dim * 2> is_boundary) {
  vec_t<bool, Conf::dim * 2> boundary(is_boundary);
  auto ext = grid.extent();
  kernel_launch(
      [ext] __device__(auto divE, auto divB, auto e, auto b, auto gp,
                       auto is_boundary) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        // gp is short for grid_ptrs
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = typename Conf::idx_t(n, ext);
          // auto pos = idx.get_pos();
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            auto idx_mx = idx.dec_x();
            auto idx_my = idx.dec_y();

            divE[idx] =
                (e[0][idx] * gp.Ae[0][idx] - e[0][idx_mx] * gp.Ae[0][idx_mx] +
                 e[1][idx] * gp.Ae[1][idx] - e[1][idx_my] * gp.Ae[1][idx_my]) /
                (gp.dV[idx] * grid.delta[0] * grid.delta[1]);
            auto idx_px = idx.inc_x();
            auto idx_py = idx.inc_y();

            divB[idx] =
                (b[0][idx_px] * gp.Ab[0][idx_px] - b[0][idx] * gp.Ab[0][idx] +
                 b[1][idx_py] * gp.Ab[1][idx_py] - b[1][idx] * gp.Ab[1][idx]) /
                (gp.dV[idx] * grid.delta[0] * grid.delta[1]);

            if (is_boundary[0] && pos[0] == grid.guard[0])
              divE[idx] = divB[idx] = 0.0f;
            if (is_boundary[1] && pos[0] == grid.dims[0] - grid.guard[0] - 1)
              divE[idx] = divB[idx] = 0.0f;
            if (is_boundary[2] && pos[1] == grid.guard[1])
              divE[idx] = divB[idx] = 0.0f;
            if (is_boundary[3] && pos[1] == grid.dims[1] - grid.guard[1] - 1)
              divE[idx] = divB[idx] = 0.0f;
          }
        }
      },
      divE.dev_ndptr(), divB.dev_ndptr(), e.get_ptrs(), b.get_ptrs(),
      grid.get_grid_ptrs(), boundary);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_flux(scalar_field<Conf>& flux, const vector_field<Conf>& b,
             const grid_curv_t<Conf>& grid) {
  flux.init();
  // auto ext = grid.extent();
  kernel_launch(
      [] __device__(auto flux, auto b, auto gp) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto n0 : grid_stride_range(0, grid.dims[0])) {
          for (int n1 = grid.guard[1]; n1 < grid.dims[1] - grid.guard[1];
               n1++) {
            auto pos = index_t<Conf::dim>(n0, n1);
            auto idx = typename Conf::idx_t(pos, ext);
            flux[idx] = flux[idx.dec_y()] + b[0][idx] * gp.Ab[0][idx];
          }
        }
      },
      flux.dev_ndptr(), b.get_ptrs(), grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

}

template <typename Conf>
void
field_solver_sph_cu<Conf>::init() {
  field_solver_cu<Conf>::init();

  sim_env().params().get_value("damping_length", m_damping_length);
  sim_env().params().get_value("damping_coef", m_damping_coef);
}

template <typename Conf>
void
field_solver_sph_cu<Conf>::register_data_components() {
  field_solver_cu<Conf>::register_data_components();

}

template <typename Conf>
void
field_solver_sph_cu<Conf>::update(double dt, uint32_t step) {
  field_solver_cu<Conf>::update(dt, step);

  if (step % this->m_data_interval == 0) {
    auto& grid = dynamic_cast<const grid_curv_t<Conf>&>(this->m_grid);
    compute_flux(*(this->flux), *(this->Btotal), grid);
  }
}

template <typename Conf>
void
field_solver_sph_cu<Conf>::update_explicit(double dt, double time) {
  auto& grid = dynamic_cast<const grid_curv_t<Conf>&>(this->m_grid);
  if (time < TINY)
    compute_b_update_explicit(*(this->B), *(this->E), grid, 0.5 * dt);
  else
    compute_b_update_explicit(*(this->B), *(this->E), grid, dt);

  // Communicate B guard cells
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->B));
  axis_boundary_b(*(this->B), grid);

  compute_e_update_explicit(*(this->E), *(this->B), *(this->J), grid, dt);
  axis_boundary_e(*(this->E), grid);

  // Communicate E guard cells
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));

  // apply coordinate boundary condition
  if (this->m_comm == nullptr || this->m_comm->domain_info().is_boundary[1]) {
    damping_boundary(*(this->E), *(this->B), m_damping_length, m_damping_coef);
  }

  if (this->m_comm != nullptr) {
    compute_divs_cu(*(this->divE), *(this->divB), *(this->E), *(this->B), grid,
                    this->m_comm->domain_info().is_boundary);
  } else {
    // vec_t<bool, 4> is_boundary(true, true, true, true);
    bool is_boundary[4] = {true, true, true, true};
    compute_divs_cu(*(this->divE), *(this->divB), *(this->E), *(this->B), grid,
                    is_boundary);
  }
}

template <typename Conf>
void
field_solver_sph_cu<Conf>::update_semi_implicit(double dt, double alpha,
                                                double beta, double time) {
  Logger::print_info("updating sph fields implicitly");
  // set m_tmp_b1 to B
  this->m_tmp_b1->copy_from(*(this->B));

  // Assemble the RHS
  auto& grid = dynamic_cast<const grid_curv_t<Conf>&>(this->m_grid);
  // compute_double_circ(*m_tmp_b2, *m_tmp_b1, *(this->B0), grid,
  //                     -alpha * beta * dt * dt);
  compute_double_circ(*(this->m_tmp_b2), *(this->m_tmp_b1), grid,
                      -alpha * beta * dt * dt);
  this->m_tmp_b1->add_by(*(this->m_tmp_b2));

  // Send guard cells for m_tmp_b1
  if (this->m_comm != nullptr)
    this->m_comm->send_guard_cells(*(this->m_tmp_b1));

  compute_implicit_rhs(*(this->m_tmp_b1), *(this->E), *(this->J), grid, alpha,
                       beta, dt);
  axis_boundary_b(*(this->m_tmp_b1), grid);
  // m_tmp_b1->add_by(*(this->B0), -1.0);

  // Since we need to iterate, define a double buffer to switch quickly between
  // operand and result.
  this->m_bnew->copy_from(*(this->m_tmp_b1));
  // m_tmp_b1->add_by(*(this->B0), -1.0);
  auto buffer = make_double_buffer(*(this->m_tmp_b1), *(this->m_tmp_b2));
  for (int i = 0; i < 6; i++) {
    compute_double_circ(buffer.alt(), buffer.main(), grid,
                        -beta * beta * dt * dt);

    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(buffer.alt());
    axis_boundary_b(buffer.alt(), grid);
    this->m_bnew->add_by(buffer.alt());

    buffer.swap();
  }
  // Communicate B new
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->m_bnew));

  // m_bnew now holds B^{n+1}
  // add_alpha_beta_cu(buffer.main(), *(this->B), *(this->m_bnew), alpha, beta);
  select_dev(buffer.main()[0]) =
      alpha * this->B->at(0) + beta * this->m_bnew->at(0);
  select_dev(buffer.main()[1]) =
      alpha * this->B->at(1) + beta * this->m_bnew->at(1);
  select_dev(buffer.main()[2]) =
      alpha * this->B->at(2) + beta * this->m_bnew->at(2);

  // buffer.main() now holds alpha*B^n + beta*B^{n+1}. Compute E explicitly from
  // this
  compute_e_update_explicit(*(this->E), buffer.main(), *(this->J), grid, dt);
  axis_boundary_e(*(this->E), grid);

  // Communicate E
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));

  this->B->copy_from(*(this->m_bnew));

  // apply coordinate boundary condition
  if (this->m_comm == nullptr || this->m_comm->domain_info().is_boundary[1]) {
    damping_boundary(*(this->E), *(this->B), m_damping_length, m_damping_coef);
  }

  if (this->m_comm != nullptr) {
    compute_divs_cu(*(this->divE), *(this->divB), *(this->E), *(this->B), grid,
                    this->m_comm->domain_info().is_boundary);
  } else {
    bool is_boundary[4] = {true, true, true, true};
    compute_divs_cu(*(this->divE), *(this->divB), *(this->E), *(this->B), grid,
                    is_boundary);
  }
}

template class field_solver_sph_cu<Config<2>>;

}  // namespace Aperture
