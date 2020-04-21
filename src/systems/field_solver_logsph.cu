#include "field_solver_logsph.h"
#include "framework/config.h"
#include "framework/environment.hpp"
#include "utils/kernel_helper.hpp"

namespace Aperture {

template <typename ValueT, typename PtrType, typename Idx_t>
HD_INLINE ValueT
circ0(const vec_t<PtrType, 3>& f, const vec_t<PtrType, 3>& dl, const Idx_t& idx,
      const Idx_t& idx_py) {
  return f[2][idx_py] * dl[2][idx_py] - f[2][idx] * dl[2][idx];
}

template <typename ValueT, typename PtrType, typename Idx_t>
HD_INLINE ValueT
circ1(const vec_t<PtrType, 3>& f, const vec_t<PtrType, 3>& dl, const Idx_t& idx,
      const Idx_t& idx_px) {
  return f[2][idx] * dl[2][idx] - f[2][idx_px] * dl[2][idx_px];
}

template <typename ValueT, typename PtrType, typename Idx_t>
HD_INLINE ValueT
circ2(const vec_t<PtrType, 3>& f, const vec_t<PtrType, 3>& dl,
      const Idx_t& idx_mx, const Idx_t& idx_my, const Idx_t& idx_px,
      const Idx_t& idx_py) {
  return f[1][idx_px] * dl[1][idx_px] - f[1][idx_mx] * dl[1][idx_mx] +
         f[0][idx_my] * dl[0][idx_my] - f[0][idx_py] * dl[0][idx_py];
}

template <typename Conf>
void
compute_double_circ(vector_field<Conf>& result, const vector_field<Conf>& b,
                    const grid_logsph_t<Conf>& grid,
                    typename Conf::value_t coef) {
  auto ext = grid.extent();
  kernel_launch(
      [coef, ext] __device__(auto result, auto b, auto grid_ptrs) {
        for (auto n : grid_stride_range(0, ext.size())) {
        }
      },
      result.get_ptrs(), b.get_ptrs(), grid.get_grid_ptrs());
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
          auto idx = result.idx_at(n, ext);
          auto pos = idx.get_pos();
          if (dev_grid<Conf>().is_in_bound(pos)) {
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
        // gp is short for grid_ptrs
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = result.idx_at(n, ext);
          auto pos = idx.get_pos();
          if (dev_grid<Conf>().is_in_bound(pos)) {
            auto idx_mx = idx.dec_x();
            result[0][idx] += dt * ((circ0(b, gp.lb, idx_mx, idx) -
                                     circ0(b0, gp.lb, idx_mx, idx)) /
                                        gp.Ae[0][idx] -
                                    j[0][idx]);

            auto idx_my = idx.dec_y();
            result[1][idx] += dt * ((circ1(b, gp.lb, idx_my, idx) -
                                     circ1(b0, gp.lb, idx_my, idx)) /
                                        gp.Ae[1][idx] -
                                    j[1][idx]);

            result[2][idx] +=
                dt * ((circ2(b, gp.lb, idx_mx, idx_my, idx, idx) -
                       circ2(b0, gp.lb, idx_mx, idx_my, idx, idx)) /
                          gp.Ae[2][idx] -
                      j[2][idx]);
          }
        }
      },
      result.get_ptrs(), b.get_ptrs(), b0.get_ptrs(), j.get_ptrs(),
      grid.get_grid_ptrs());
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
          auto idx = result.idx_at(n, ext);
          auto pos = idx.get_pos();
          if (dev_grid<Conf>().is_in_bound(pos)) {
            auto idx_px = idx.inc_x();
            result[0][idx] += dt * (circ0(e, gp.le, idx, idx_px) -
                                    circ0(e0, gp.le, idx, idx_px)) /
                gp.Ab[0][idx];

            auto idx_py = idx.inc_y();
            result[1][idx] += dt * (circ1(e, gp.le, idx, idx_py) -
                                    circ1(e0, gp.le, idx, idx_py)) /
                gp.Ab[1][idx];

            result[2][idx] +=
                dt * (circ2(e, gp.le, idx, idx, idx_px, idx_py) -
                      circ2(e0, gp.le, idx, idx, idx_px, idx_py)) /
                gp.Ab[2][idx];
          }
        }
      },
      result.get_ptrs(), e.get_ptrs(), e0.get_ptrs(),
      grid.get_grid_ptrs());
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
}

template <typename Conf>
void
field_solver_logsph<Conf>::update(double dt, uint32_t step) {
  double time = this->m_env.get_time();
  update_semi_impl(dt, m_alpha, m_beta, time);
}

template <typename Conf>
void
field_solver_logsph<Conf>::update_semi_impl(double dt, double alpha,
                                            double beta, double time) {
  // set m_tmp_b1 to B - B0
  m_tmp_b1->copy_from(*(this->B));
  m_tmp_b1->add_by(*(this->B0), -1.0);

  compute_double_circ(*m_tmp_b2, *m_tmp_b1,
                      dynamic_cast<const grid_logsph_t<Conf>&>(this->m_grid),
                      -alpha * beta * dt * dt);
  m_tmp_b1->add_by(*m_tmp_b2);

  // Send guard cells for m_tmp_b1
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*m_tmp_b1);
}

template <typename Conf>
void
field_solver_logsph<Conf>::update_b(double dt, double alpha, double beta) {}

template <typename Conf>
void
field_solver_logsph<Conf>::update_e(double dt, double alpha, double beta) {}

template class field_solver_logsph<Config<2>>;

}  // namespace Aperture
