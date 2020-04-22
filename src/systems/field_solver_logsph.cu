#include "field_solver_logsph.h"
#include "framework/config.h"
#include "framework/environment.hpp"
#include "utils/double_buffer.h"
#include "utils/kernel_helper.hpp"
#include "utils/timer.h"

namespace Aperture {

// template <typename PtrType, typename Idx_t>
template <typename VecType, typename Idx_t>
HD_INLINE Scalar
circ0(const VecType& f, const VecType& dl, const Idx_t& idx,
      const Idx_t& idx_py) {
  return f[2][idx] * dl[2][idx] - f[2][idx_py] * dl[2][idx_py];
}

template <typename VecType, typename Idx_t>
HD_INLINE Scalar
circ1(const VecType& f, const VecType& dl, const Idx_t& idx,
      const Idx_t& idx_px) {
  return f[2][idx_px] * dl[2][idx_px] - f[2][idx] * dl[2][idx];
}

template <typename VecType, typename Idx_t>
HD_INLINE Scalar
circ2(const VecType& f, const VecType& dl, const Idx_t& idx_mx,
      const Idx_t& idx_my, const Idx_t& idx_px, const Idx_t& idx_py) {
  return f[1][idx_mx] * dl[1][idx_mx] - f[1][idx_px] * dl[1][idx_px] +
         f[0][idx_py] * dl[0][idx_py] - f[0][idx_my] * dl[0][idx_my];
}

template <typename Conf>
void
add_alpha_beta(vector_field<Conf>& result, const vector_field<Conf>& b1,
               const vector_field<Conf>& b2, typename Conf::value_t alpha,
               typename Conf::value_t beta) {
  auto ext = result.grid().extent();
  kernel_launch([alpha, beta, ext] __device__(auto result, auto b1, auto b2) {
      for (auto n : grid_stride_range(0, ext.size())) {
        auto idx = typename Conf::idx_t(n, ext);
        result[0][idx] = alpha * b1[0][idx] + beta * b2[0][idx];
        result[1][idx] = alpha * b1[1][idx] + beta * b2[1][idx];
        result[2][idx] = alpha * b1[2][idx] + beta * b2[2][idx];
      }
    } , result.get_ptrs(), b1.get_ptrs(), b2.get_ptrs());
}

template <typename Conf>
void
compute_double_circ(vector_field<Conf>& result, const vector_field<Conf>& b,
                    const vector_field<Conf>& b0,
                    const grid_logsph_t<Conf>& grid,
                    typename Conf::value_t coef) {
  auto ext = grid.extent();
  kernel_launch(
      [coef, ext] __device__(auto result, auto b, auto b0, auto gp) {
        auto& grid = dev_grid<Conf::dim>();
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = typename Conf::idx_t(n, ext);
          auto pos = idx.get_pos();
          if (grid.is_in_bound(pos)) {
            auto idx_mx = idx.dec_x();
            auto idx_my = idx.dec_y();
            auto idx_px = idx.inc_x();
            auto idx_py = idx.inc_y();
            auto idx_pymx = idx.inc_y().dec_x();
            auto idx_pxmy = idx.inc_x().dec_y();
            auto idx_pxpy = idx.inc_x().inc_y();
            result[0][idx] = coef * (gp.le[2][idx_py] *
                                     (circ2(b, gp.lb, idx_pymx, idx, idx_py, idx_py) -
                                      circ2(b0, gp.lb, idx_pymx, idx, idx_py, idx_py)) /
                                     gp.Ae[2][idx_py] -
                                     gp.le[2][idx] *
                                     (circ2(b, gp.lb, idx_mx, idx_my, idx, idx) -
                                      circ2(b0, gp.lb, idx_mx, idx_my, idx, idx)) /
                                     gp.Ae[2][idx]) / gp.Ae[0][idx];

            result[1][idx] = coef * (gp.le[2][idx] *
                                     (circ2(b, gp.lb, idx_mx, idx_my, idx, idx) -
                                      circ2(b0, gp.lb, idx_mx, idx_my, idx, idx)) /
                                     gp.Ae[2][idx] -
                                     gp.le[2][idx_px] *
                                     (circ2(b, gp.lb, idx, idx_pxmy, idx_px, idx_px) -
                                      circ2(b0, gp.lb, idx, idx_pxmy, idx_px, idx_px)) /
                                     gp.Ae[2][idx_px]) / gp.Ae[0][idx];

            result[2][idx] = coef * (gp.le[0][idx] *
                                     (circ0(b, gp.lb, idx_my, idx) -
                                      circ0(b0, gp.lb, idx_my, idx)) / gp.Ae[0][idx] -
                                     gp.le[0][idx_py] *
                                     (circ0(b, gp.lb, idx, idx_py) -
                                      circ0(b0, gp.lb, idx, idx_py)) / gp.Ae[0][idx_py] +
                                     gp.le[1][idx_px] *
                                     (circ0(b, gp.lb, idx_px, idx_pxpy) -
                                      circ0(b0, gp.lb, idx_px, idx_pxpy)) / gp.Ae[1][idx_px] -
                                     gp.le[1][idx] *
                                     (circ0(b, gp.lb, idx, idx_py) -
                                      circ0(b0, gp.lb, idx, idx_py)) / gp.Ae[1][idx]);
          }
        }
      },
      result.get_ptrs(), b.get_ptrs(), b0.get_ptrs(), grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Conf>
void
compute_implicit_rhs(vector_field<Conf>& result, const vector_field<Conf>& e,
                     const vector_field<Conf>& e0, const vector_field<Conf>& j,
                     const grid_logsph_t<Conf>& grid,
                     typename Conf::value_t alpha, typename Conf::value_t beta,
                     typename Conf::value_t dt) {
  auto ext = grid.extent();
  kernel_launch(
      [alpha, beta, dt, ext] __device__(auto result, auto e, auto e0, auto j,
                                        auto gp) {
        // gp is short for grid_ptrs
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = result[0].idx_at(n, ext);
          auto pos = idx.get_pos();
          if (dev_grid<Conf::dim>().is_in_bound(pos)) {
            auto idx_py = idx.inc_y();
            result[0][idx] +=
                -dt *
                ((alpha + beta) * (circ0(e, gp.le, idx, idx_py) -
                                   circ0(e0, gp.le, idx, idx_py)) -
                 dt * beta * circ0(j, gp.le, idx, idx_py)) /
                gp.Ab[0][idx];

            auto idx_px = idx.inc_x();
            result[1][idx] +=
                -dt *
                ((alpha + beta) * (circ1(e, gp.le, idx, idx_px) -
                                   circ1(e0, gp.le, idx, idx_px)) -
                 dt * beta * circ1(j, gp.le, idx, idx_px)) /
                gp.Ab[1][idx];

            result[2][idx] +=
                -dt *
                ((alpha + beta) * (circ2(e, gp.le, idx, idx, idx_px, idx_py) -
                                   circ2(e0, gp.le, idx, idx, idx_px, idx_py)) -
                 dt * beta * circ2(j, gp.le, idx, idx, idx_px, idx_py)) /
                gp.Ab[2][idx];
          }
        }
      },
      result.get_ptrs(), e.get_ptrs(), e0.get_ptrs(), j.get_ptrs(),
      grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Conf>
void
compute_e_update_explicit(vector_field<Conf>& result,
                          const vector_field<Conf>& b,
                          const vector_field<Conf>& b0,
                          const vector_field<Conf>& j,
                          const grid_logsph_t<Conf>& grid,
                          typename Conf::value_t dt) {
  auto ext = grid.extent();
  kernel_launch(
      [dt, ext] __device__(auto result, auto b, auto b0, auto j, auto gp) {
        auto& grid = dev_grid<Conf::dim>();
        // gp is short for grid_ptrs
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = result[0].idx_at(n, ext);
          auto pos = idx.get_pos();
          if (grid.is_in_bound(pos)) {
            auto idx_my = idx.dec_y();
            result[0][idx] += dt * ((circ0(b, gp.lb, idx_my, idx) -
                                     circ0(b0, gp.lb, idx_my, idx)) /
                                        gp.Ae[0][idx] -
                                    j[0][idx]);

            auto idx_mx = idx.dec_x();
            result[1][idx] += dt * ((circ1(b, gp.lb, idx_mx, idx) -
                                     circ1(b0, gp.lb, idx_mx, idx)) /
                                        gp.Ae[1][idx] -
                                    j[1][idx]);

            result[2][idx] +=
                dt * ((circ2(b, gp.lb, idx_mx, idx_my, idx, idx) -
                       circ2(b0, gp.lb, idx_mx, idx_my, idx, idx)) /
                          gp.Ae[2][idx] -
                      j[2][idx]);
          }
          // extra work for the theta = pi axis
          auto theta = grid.template pos<1>(pos[1], true);
          if (std::abs(theta - M_PI) < 0.1f * grid.delta[1]) {
            auto idx_my = idx.dec_y();
            result[0][idx] += dt * ((circ0(b, gp.lb, idx_my, idx) -
                                     circ0(b0, gp.lb, idx_my, idx)) /
                                        gp.Ae[0][idx] -
                                    j[0][idx]);
          }
        }
      },
      result.get_ptrs(), b.get_ptrs(), b0.get_ptrs(), j.get_ptrs(),
      grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Conf>
void
compute_b_update_explicit(vector_field<Conf>& result,
                          const vector_field<Conf>& e,
                          const vector_field<Conf>& e0,
                          const grid_logsph_t<Conf>& grid,
                          typename Conf::value_t dt) {
  auto ext = grid.extent();
  kernel_launch(
      [dt, ext] __device__(auto result, auto e, auto e0, auto gp) {
        // gp is short for grid_ptrs
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = typename Conf::idx_t(n, ext);
          auto pos = idx.get_pos();
          if (dev_grid<Conf::dim>().is_in_bound(pos)) {
            auto idx_py = idx.inc_y();
            result[0][idx] -=
                dt *
                (circ0(e, gp.le, idx, idx_py) - circ0(e0, gp.le, idx, idx_py)) /
                gp.Ab[0][idx];

            auto idx_px = idx.inc_x();
            result[1][idx] -=
                dt *
                (circ1(e, gp.le, idx, idx_px) - circ1(e0, gp.le, idx, idx_px)) /
                gp.Ab[1][idx];

            result[2][idx] -= dt *
                              (circ2(e, gp.le, idx, idx, idx_px, idx_py) -
                               circ2(e0, gp.le, idx, idx, idx_px, idx_py)) /
                              gp.Ab[2][idx];
          }
        }
      },
      result.get_ptrs(), e.get_ptrs(), e0.get_ptrs(), grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Conf>
void
axis_boundary(vector_field<Conf>& e, vector_field<Conf>& b,
              const vector_field<Conf>& e0, const vector_field<Conf>& b0,
              const grid_logsph_t<Conf>& grid) {
  auto ext = grid.extent();
  typedef typename Conf::idx_t idx_t;
  kernel_launch(
      [ext] __device__(auto e, auto b, auto e0, auto b0) {
        auto& grid = dev_grid<Conf::dim>();
        for (auto n0 : grid_stride_range(0, grid.dims[0])) {
          auto n1_0 = grid.guard[1];
          auto n1_pi = grid.dims[1] - grid.guard[1];
          if (abs(grid.template pos<1>(n1_0, true)) < TINY) {
            // At the theta = 0 axis

            // Set E_phi and B_theta to zero
            auto idx = idx_t(index_t<2>(n0, n1_0), ext);
            e[2][idx] = 0.0;
            b[1][idx] = 0.0;
            b[2][idx.dec_y()] = -b[2][idx];
          }
          // printf("boundary pi at %f\n", grid.template pos<1>(n1_pi, true));
          if (abs(grid.template pos<1>(n1_pi, true) - M_PI) <
              0.1f * grid.delta[1]) {
            // At the theta = pi axis
            auto idx = idx_t(index_t<2>(n0, n1_pi), ext);
            e[2][idx] = 0.0;
            b[1][idx] = 0.0;
            b[2][idx] = -b[2][idx.dec_y()];
          }
        }
      },
      e.get_ptrs(), b.get_ptrs(), e0.get_ptrs(), b0.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Conf>
void
field_solver_logsph<Conf>::init() {
  this->m_env.params().get_value("implicit_alpha", m_alpha);
  this->m_env.params().get_value("implicit_beta", m_beta);

  m_tmp_b1 =
      std::make_unique<vector_field<Conf>>(this->m_grid, MemType::device_only);
  m_tmp_b2 =
      std::make_unique<vector_field<Conf>>(this->m_grid, MemType::device_only);
  m_bnew =
      std::make_unique<vector_field<Conf>>(this->m_grid, MemType::device_only);
}

template <typename Conf>
void
field_solver_logsph<Conf>::update(double dt, uint32_t step) {
  double time = this->m_env.get_time();
  // update_semi_impl(dt, m_alpha, m_beta, time);
  update_explicit(dt, time);
}

template <typename Conf>
void
field_solver_logsph<Conf>::update_explicit(double dt, double time) {
  auto& grid = dynamic_cast<const grid_logsph_t<Conf>&>(this->m_grid);
  compute_b_update_explicit(*(this->B), *(this->E), *(this->E0), grid, dt);

  // Communicate B guard cells
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->B));

  compute_e_update_explicit(*(this->E), *(this->B), *(this->B0), *(this->J),
                            grid, dt);

  // Communicate E guard cells
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));

  // TODO: Compute divE and divB

  // apply coordinate boundary condition
  axis_boundary(*(this->E), *(this->B), *(this->E0), *(this->B0), grid);
}

template <typename Conf>
void
field_solver_logsph<Conf>::update_semi_impl(double dt, double alpha,
                                            double beta, double time) {
  // set m_tmp_b1 to B - B0
  m_tmp_b1->copy_from(*(this->B));
  m_tmp_b1->add_by(*(this->B0), -1.0);

  auto& grid = dynamic_cast<const grid_logsph_t<Conf>&>(this->m_grid);
  compute_double_circ(*m_tmp_b2, *m_tmp_b1, *(this->B0), grid, -alpha * beta * dt * dt);
  m_tmp_b1->add_by(*m_tmp_b2);

  // Send guard cells for m_tmp_b1
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*m_tmp_b1);

  compute_implicit_rhs(*m_tmp_b1, *(this->E), *(this->E0), *(this->J), grid,
                       alpha, beta, dt);

  // Since we need to iterate, define a double buffer to switch quickly between
  // operand and result.
  m_bnew->copy_from(*m_tmp_b1);
  auto buffer = make_double_buffer(*m_tmp_b1, *m_tmp_b2);
  for (int i = 0; i < 5; i++) {
    compute_double_circ(buffer.alt(), buffer.main(), *(this->B0), grid,
                        -beta * beta * dt * dt);
    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(buffer.alt());
    m_bnew->add_by(buffer.alt());

    buffer.swap();
  }
  // m_bnew now holds B^{n+1}
  add_alpha_beta(buffer.main(), *(this->B), *m_bnew, alpha, beta);

  // buffer.main() now holds alpha*B^n + beta*B^{n+1}. Compute E explicitly from this
  compute_e_update_explicit(*(this->E), buffer.main(), *(this->B0), *(this->J), grid, dt);

  // Communicate E
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));

  this->B->copy_from(*m_bnew);
}

template <typename Conf>
void
field_solver_logsph<Conf>::update_b(double dt, double alpha, double beta) {}

template <typename Conf>
void
field_solver_logsph<Conf>::update_e(double dt, double alpha, double beta) {}

template class field_solver_logsph<Config<2>>;

}  // namespace Aperture
