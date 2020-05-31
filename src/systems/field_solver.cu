#include "core/cuda_control.h"
#include "field_solver.h"
#include "framework/config.h"
#include "systems/helpers/finite_diff_helper.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/timer.h"

namespace Aperture {

template <typename Conf>
void
compute_e_update_explicit(vector_field<Conf>& result,
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
            result[0][idx] += dt * (finite_diff<Conf::dim>::curl0(b, idx, stagger)
                                    - j[0][idx]);

            result[1][idx] += dt * (finite_diff<Conf::dim>::curl1(b, idx, stagger)
                                    - j[1][idx]);

            result[2][idx] += dt * (finite_diff<Conf::dim>::curl2(b, idx, stagger)
                                    - j[2][idx]);
          }
        }
      },
      result.get_ptrs(), b.get_ptrs(), b.stagger_vec(), j.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_b_update_explicit(vector_field<Conf>& result,
                          const vector_field<Conf>& e,
                          typename Conf::value_t dt) {
  kernel_launch(
      [dt] __device__(auto result, auto e, auto stagger) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = idx.get_pos();
          if (grid.is_in_bound(pos)) {
            result[0][idx] += dt * finite_diff<Conf::dim>::curl0(e, idx, stagger);

            result[1][idx] += dt * finite_diff<Conf::dim>::curl1(e, idx, stagger);

            result[2][idx] += dt * finite_diff<Conf::dim>::curl2(e, idx, stagger);
          }
        }
      },
      result.get_ptrs(), e.get_ptrs(), e.stagger_vec());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
field_solver_cu<Conf>::init_impl_tmp_fields() {
  if (this->m_use_implicit) {
    this->m_tmp_b1 = std::make_unique<vector_field<Conf>>(this->m_grid,
                                                          MemType::device_only);
    this->m_tmp_b2 = std::make_unique<vector_field<Conf>>(this->m_grid,
                                                          MemType::device_only);
    this->m_bnew = std::make_unique<vector_field<Conf>>(this->m_grid,
                                                        MemType::device_only);
  }
}

template <typename Conf>
void
field_solver_cu<Conf>::register_data_components() {
  this->register_data_impl(MemType::host_device);
}

template <typename Conf>
void
field_solver_cu<Conf>::update_explicit(double dt, double time) {
  timer::stamp("field_update");
  if (time < TINY) {
    compute_e_update_explicit(*(this->E), *(this->B), *(this->J), 0.5f * dt);
    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));
  }

  compute_b_update_explicit(*(this->B), *(this->E), dt);

  // Communicate the new B values to guard cells
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->B));

  compute_e_update_explicit(*(this->E), *(this->B), *(this->J), dt);
  // Communicate the new E values to guard cells
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));

  // compute_divs();

  CudaSafeCall(cudaDeviceSynchronize());
  timer::show_duration_since_stamp("Field update", "ms", "field_update");
}

template <typename Conf>
void
field_solver_cu<Conf>::update_semi_implicit(double dt, double alpha,
                                            double beta, double time) {
  // FIXME: implement semi implicit update!!!
}

INSTANTIATE_WITH_CONFIG(field_solver_cu);


}  // namespace Aperture
