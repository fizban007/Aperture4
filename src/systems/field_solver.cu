#include "core/cuda_control.h"
#include "field_solver.h"
#include "framework/config.h"
#include "systems/helpers/finite_diff_helper.hpp"
#include "systems/helpers/field_solver_helper_cu.hpp"
#include "utils/double_buffer.h"
#include "utils/kernel_helper.hpp"
#include "utils/timer.h"

namespace Aperture {

template <typename Conf>
using fd = finite_diff<Conf::dim, 2>;

template <typename Conf>
void
compute_e_update_explicit_cu(vector_field<Conf>& result,
                             const vector_field<Conf>& b,
                             const vector_field<Conf>& j,
                             typename Conf::value_t dt) {
  kernel_launch(
      [dt] __device__(auto result, auto b, auto stagger, auto j) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = idx.get_pos();
          if (grid.is_in_bound(pos)) {
            result[0][idx] +=
                dt *
                (1.025f * fd<Conf>::curl0(b, idx, stagger, grid) - j[0][idx]);

            result[1][idx] +=
                dt *
                (1.025f * fd<Conf>::curl1(b, idx, stagger, grid) - j[1][idx]);

            result[2][idx] +=
                dt *
                (1.025f * fd<Conf>::curl2(b, idx, stagger, grid) - j[2][idx]);
          }
        }
      },
      result.get_ptrs(), b.get_ptrs(), b.stagger_vec(), j.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_b_update_explicit_cu(vector_field<Conf>& result,
                             const vector_field<Conf>& e,
                             typename Conf::value_t dt) {
  kernel_launch(
      [dt] __device__(auto result, auto e, auto stagger) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = idx.get_pos();
          if (grid.is_in_bound(pos)) {
            result[0][idx] +=
                -dt * 1.025f * fd<Conf>::curl0(e, idx, stagger, grid);

            result[1][idx] +=
                -dt * 1.025f * fd<Conf>::curl1(e, idx, stagger, grid);

            result[2][idx] +=
                -dt * 1.025f * fd<Conf>::curl2(e, idx, stagger, grid);
          }
        }
      },
      result.get_ptrs(), e.get_ptrs(), e.stagger_vec());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_double_curl(vector_field<Conf>& result, const vector_field<Conf>& b,
                    typename Conf::value_t coef) {
  kernel_launch(
      [coef] __device__(auto result, auto b, auto stagger) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();

        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = idx.get_pos();
          if (grid.is_in_bound(pos)) {
            result[0][idx] = -coef * (fd<Conf>::laplacian(b[0], idx, grid));
            result[1][idx] = -coef * (fd<Conf>::laplacian(b[1], idx, grid));
            result[2][idx] = -coef * (fd<Conf>::laplacian(b[2], idx, grid));
          }
        }
      },
      result.get_ptrs(), b.get_ptrs(), b.stagger_vec());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_implicit_rhs(vector_field<Conf>& result, const vector_field<Conf>& e,
                     const vector_field<Conf>& j, typename Conf::value_t alpha,
                     typename Conf::value_t beta, typename Conf::value_t dt) {
  kernel_launch(
      [alpha, beta, dt] __device__(auto result, auto e, auto j, auto stagger) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();

        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = idx.get_pos();
          if (grid.is_in_bound(pos)) {
            result[0][idx] +=
                -dt * (fd<Conf>::curl0(e, idx, stagger, grid) -
                       dt * beta * fd<Conf>::curl0(j, idx, stagger, grid));
            result[1][idx] +=
                -dt * (fd<Conf>::curl1(e, idx, stagger, grid) -
                       dt * beta * fd<Conf>::curl1(j, idx, stagger, grid));
            result[2][idx] +=
                -dt * (fd<Conf>::curl2(e, idx, stagger, grid) -
                       dt * beta * fd<Conf>::curl2(j, idx, stagger, grid));
          }
        }
      },
      result.get_ptrs(), e.get_ptrs(), j.get_ptrs(), e.stagger_vec());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_divs_cu(scalar_field<Conf>& divE, scalar_field<Conf>& divB,
                const vector_field<Conf>& e, const vector_field<Conf>& b,
                const bool is_boundary[Conf::dim * 2]) {
  vec_t<bool, Conf::dim * 2> boundary(is_boundary);
  kernel_launch(
      [] __device__(auto divE, auto divB, auto e, auto b, auto st_e, auto st_b,
                    auto is_boundary) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = idx.get_pos();
          if (grid.is_in_bound(pos)) {
            divE[idx] = fd<Conf>::div(e, idx, st_e, grid);
            divB[idx] = fd<Conf>::div(b, idx, st_b, grid);

            // Check boundary
            // if (is_boundary[0] && pos[0] == grid.skirt[0])
            //   divE[idx] = divB[idx] = 0.0f;
            // if (is_boundary[1] && pos[0] == grid.dims[0] - grid.skirt[0] - 1)
            //   divE[idx] = divB[idx] = 0.0f;
            // if (is_boundary[2] && pos[1] == grid.skirt[1])
            //   divE[idx] = divB[idx] = 0.0f;
            // if (is_boundary[3] && pos[1] == grid.dims[1] - grid.skirt[1] - 1)
            //   divE[idx] = divB[idx] = 0.0f;
          }
        }
      },
      divE.get_ptr(), divB.get_ptr(), e.get_ptrs(), b.get_ptrs(),
      e.stagger_vec(), b.stagger_vec(), boundary);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
field_solver_cu<Conf>::init_impl_tmp_fields() {
  this->m_tmp_b1 =
      std::make_unique<vector_field<Conf>>(this->m_grid, MemType::device_only);
  this->m_tmp_b2 =
      std::make_unique<vector_field<Conf>>(this->m_grid, MemType::device_only);
  this->m_bnew =
      std::make_unique<vector_field<Conf>>(this->m_grid, MemType::device_only);
}

template <typename Conf>
void
field_solver_cu<Conf>::register_data_components() {
  this->register_data_impl(MemType::host_device);
}

template <typename Conf>
void
field_solver_cu<Conf>::update_explicit(double dt, double time) {
  // dt *= 1.025;
  if (time < TINY) {
    compute_e_update_explicit_cu(*(this->E), *(this->B), *(this->J), 0.5f * dt);
    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));
  }

  compute_b_update_explicit_cu(*(this->B), *(this->E), dt);

  // Communicate the new B values to guard cells
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->B));

  compute_e_update_explicit_cu(*(this->E), *(this->B), *(this->J), dt);
  // Communicate the new E values to guard cells
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));

  if (this->m_comm != nullptr) {
    compute_divs_cu(*(this->divE), *(this->divB), *(this->E), *(this->B),
                    this->m_comm->domain_info().is_boundary);
  } else {
    bool is_boundary[4] = {true, true, true, true};
    compute_divs_cu(*(this->divE), *(this->divB), *(this->E), *(this->B),
                    is_boundary);
  }

  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Conf>
void
field_solver_cu<Conf>::update_semi_implicit(double dt, double alpha,
                                            double beta, double time) {
  // set m_tmp_b1 to B
  this->m_tmp_b1->copy_from(*(this->B));

  // Assemble the RHS
  compute_double_curl(*(this->m_tmp_b2), *(this->m_tmp_b1),
                      -alpha * beta * dt * dt);
  this->m_tmp_b1->add_by(*(this->m_tmp_b2));

  // Send guard cells for m_tmp_b1
  if (this->m_comm != nullptr)
    this->m_comm->send_guard_cells(*(this->m_tmp_b1));

  compute_implicit_rhs(*(this->m_tmp_b1), *(this->E), *(this->J), alpha,
                       beta, dt);

  // Since we need to iterate, define a double buffer to switch quickly between
  // operand and result.
  this->m_bnew->copy_from(*(this->m_tmp_b1));

  auto buffer = make_double_buffer(*(this->m_tmp_b1), *(this->m_tmp_b2));
  for (int i = 0; i < 6; i++) {
    compute_double_curl(buffer.alt(), buffer.main(),
                        -beta * beta * dt * dt);

    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(buffer.alt());
    this->m_bnew->add_by(buffer.alt());

    buffer.swap();
  }
  // m_bnew now holds B^{n+1}
  add_alpha_beta_cu(buffer.main(), *(this->B), *(this->m_bnew), alpha, beta);

  // buffer.main() now holds alpha*B^n + beta*B^{n+1}. Compute E explicitly from
  // this
  compute_e_update_explicit_cu(*(this->E), buffer.main(), *(this->J), dt);

  // Communicate E
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));

  this->B->copy_from(*(this->m_bnew));

  if (this->m_comm != nullptr) {
    compute_divs_cu(*(this->divE), *(this->divB), *(this->E), *(this->B),
                    this->m_comm->domain_info().is_boundary);
  } else {
    bool is_boundary[4] = {true, true, true, true};
    compute_divs_cu(*(this->divE), *(this->divB), *(this->E), *(this->B),
                    is_boundary);
  }
}

INSTANTIATE_WITH_CONFIG(field_solver_cu);

}  // namespace Aperture
