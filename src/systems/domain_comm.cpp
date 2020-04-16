#include "domain_comm.hpp"
#include "framework/config.h"
#include "framework/environment.hpp"
#include "utils/logger.h"
#include "utils/mpi_helper.h"

namespace Aperture {

template <typename Conf>
domain_comm<Conf>::domain_comm(sim_environment &env) : system_t(env) {
  setup_domain();
}

template <typename Conf>
void
domain_comm<Conf>::setup_domain() {
  m_world = MPI_COMM_WORLD;
  MPI_Comm_rank(m_world, &m_rank);
  MPI_Comm_size(m_world, &m_size);

  m_scalar_type = MPI_Helper::get_mpi_datatype(typename Conf::value_t{});

  // This is the first place where rank is defined. Tell logger about
  // this
  Logger::init(m_rank, (LogLevel)m_env.params().get<int64_t>(
                           "log_level", (int64_t)LogLevel::info));

  auto dims = m_env.params().get<std::vector<int64_t>>("nodes");
  if (dims.size() < Conf::dim) dims.resize(Conf::dim, 1);

  int64_t total_dim = 1;
  for (int i = 0; i < Conf::dim; i++) {
    total_dim *= dims[i];
  }

  if (total_dim != m_size) {
    // Given node configuration is not correct, create one on our own
    Logger::print_err(
        "Domain decomp in config file does not make sense, generating "
        "our own.");
    for (int i = 0; i < Conf::dim; i++) dims[i] = 0;

    MPI_Dims_create(m_size, Conf::dim, m_domain_info.mpi_dims);
    Logger::err("Created domain decomp as");
    for (int i = 0; i < Conf::dim; i++) {
      Logger::err("{}", m_domain_info.mpi_dims[i]);
      if (i != Conf::dim - 1) Logger::err(" x ");
    }
    Logger::err("\n");
  } else {
    for (int i = 0; i < Conf::dim; i++) m_domain_info.mpi_dims[i] = dims[i];
  }

  auto periodic = m_env.params().get<std::vector<bool>>("periodic_boundary");
  for (int i = 0; i < std::min(Conf::dim, (int)periodic.size()); i++)
    m_domain_info.is_periodic[i] = periodic[i];

  // Create a cartesian MPI group for communication
  MPI_Cart_create(m_world, Conf::dim, m_domain_info.mpi_dims,
                  m_domain_info.is_periodic, true, &m_cart);

  // Obtain the mpi coordinate of the current rank
  MPI_Cart_coords(m_cart, m_rank, Conf::dim, m_domain_info.mpi_coord);

  // Figure out if the current rank is at any boundary
  int left = 0, right = 0;
  int rank = 0;
  for (int n = 0; n < Conf::dim; n++) {
    MPI_Cart_shift(m_cart, n, -1, &rank, &left);
    MPI_Cart_shift(m_cart, n, 1, &rank, &right);
    m_domain_info.neighbor_left[n] = left;
    m_domain_info.neighbor_right[n] = right;
    if (left < 0) m_domain_info.is_boundary[2 * n] = true;
    if (right < 0) m_domain_info.is_boundary[2 * n + 1] = true;
  }

#ifdef CUDA_ENABLED
  // Poll the system to detect how many GPUs are on the node, set the
  // GPU corresponding to the rank
  int n_devices;
  cudaGetDeviceCount(&n_devices);
  if (n_devices <= 0) {
    std::cerr << "No usable Cuda device found!!" << std::endl;
    exit(1);
  }
  // TODO: This way of finding device id may not be reliable
  int dev_id = m_rank % n_devices;
  cudaSetDevice(dev_id);
#endif
}

template <typename Conf>
void
domain_comm<Conf>::resize_buffers(const Grid<Conf::dim> &grid) {
  for (int i = 0; i < Conf::dim; i++) {
    auto ext = extent_t<Conf::dim>{};
    for (int j = 0; j < Conf::dim; j++) {
      if (j == i)
        ext[j] = grid.guard[j];
      else
        ext[j] = grid.dims[j];
    }
    m_send_buffers.emplace_back(ext);
    m_recv_buffers.emplace_back(ext);
  }

  uint32_t ptc_buffer_size =
      m_env.params().get<int64_t>("ptc_buffer_size", 100000l);
  uint32_t ph_buffer_size =
      m_env.params().get<int64_t>("ph_buffer_size", 100000l);
  int num_ptc_buffers = std::pow(3, Conf::dim);
  for (int i = 0; i < num_ptc_buffers; i++) {
    m_ptc_buffers.emplace_back(ptc_buffer_size);
    m_ph_buffers.emplace_back(ph_buffer_size);
  }
}

template <typename Conf>
void
domain_comm<Conf>::send_guard_cells(vector_field<Conf> &field) const {}

template <typename Conf>
void
domain_comm<Conf>::send_guard_cells(scalar_field<Conf> &field) const {}

template <typename Conf>
void
domain_comm<Conf>::send_add_guard_cells(vector_field<Conf> &field) const {}

template <typename Conf>
void
domain_comm<Conf>::send_add_guard_cells(scalar_field<Conf> &field) const {}

// Explicitly instantiate some of the configurations that may occur
template class domain_comm<Config<1>>;
template class domain_comm<Config<2>>;
template class domain_comm<Config<3>>;

}  // namespace Aperture
