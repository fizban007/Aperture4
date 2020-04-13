#include "grid.hpp"
#include "core/constant_mem_func.h"
#include "core/domain_info.h"
#include "framework/config.h"
#include "framework/environment.hpp"
#include <exception>

namespace Aperture {

template <typename Conf>
void
grid_t<Conf>::init() {
  // Obtain grid parameters from the params store
  auto vec_N = m_env->params().get<std::vector<int64_t>>("N");
  if (vec_N.size() < Conf::dim)
    throw std::out_of_range(
        "Need larger array for N in grid parameters!");

  auto vec_guard = m_env->params().get<std::vector<int64_t>>("guard");
  if (vec_guard.size() < Conf::dim)
    throw std::out_of_range(
        "Need larger array for guard in grid parameters!");

  auto vec_size = m_env->params().get<std::vector<double>>("size");
  if (vec_size.size() < Conf::dim)
    throw std::out_of_range(
        "Need larger array for size in grid parameters!");

  auto vec_lower = m_env->params().get<std::vector<double>>("lower");
  if (vec_lower.size() < Conf::dim)
    throw std::out_of_range(
        "Need larger array for lower in grid parameters!");

  // Initialize the grid parameters
  for (int i = 0; i < Conf::dim; i++) {
    this->guard[i] = vec_guard[i];
    this->sizes[i] = vec_size[i];
    this->lower[i] = vec_lower[i];
    this->delta[i] = vec_size[i] / vec_N[i];
    this->inv_delta[i] = 1.0 / this->delta[i];
    // TODO: the z-order case is very weird. Is there a better way?
    if (Conf::is_zorder) {
      this->skirt[i] = 8;
      this->dims[i] = vec_N[i];
    } else {
      this->skirt[i] = vec_guard[i];
      this->dims[i] = vec_N[i] + 2 * vec_guard[i];
    }
  }

  // Adjust the grid according to domain decomposition
  auto domain_info =
      m_env->shared_data().get<domain_info_t<Conf::dim>>("domain_info");
  if (domain_info != nullptr) {
    for (int d = 0; d < Conf::dim; d++) {
      this->dims[d] = this->reduced_dim(d) / domain_info->mpi_dims[d] +
                      2 * this->guard[d];
      this->sizes[d] /= domain_info->mpi_dims[d];
      this->lower[d] += domain_info->mpi_coord[d] * this->sizes[d];
      // TODO: In a non-uniform domain decomposition, the offset could
      // change, need a more robust way to count this
      this->offset[d] =
          domain_info->mpi_coord[d] * this->reduced_dim(d);
    }
  }

  // Save a pointer to the grid struct in shared data
  m_env->shared_data().save("grid", *(Grid<Conf::dim> *)this);

  // Copy the grid parameters to gpu
#ifdef CUDA_ENABLED
  init_dev_grid<Conf::dim>(*this);
#endif
}

template <typename Conf>
void
grid_t<Conf>::register_dependencies(sim_environment &env) {
  // If we are initializing the communicator system, it should be done
  // before initializing this one
  depends_on("communicator");
}

template class grid_t<Config<1>>;
template class grid_t<Config<2>>;
template class grid_t<Config<3>>;

}  // namespace Aperture
