/*
 * Copyright (c) 2022 Alex Chen.
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

#include "domain_comm_impl.hpp"
#include "framework/config.h"
#include "systems/policies/exec_policy_gpu.hpp"

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
void
domain_comm<Conf, ExecPolicy>::setup_devices() {
  // Poll the system to detect how many GPUs are on the node, set the
  // GPU corresponding to the rank
  int n_devices;
  GpuSafeCall(gpuGetDeviceCount(&n_devices));
  if (n_devices <= 0) {
    std::cerr << "No usable Cuda device found!!" << std::endl;
    exit(1);
  } else {
    Logger::print_info("Found {} Cuda devices!", n_devices);
  }

  // Use MPI_Comm_split_type to get a node-local communicator, then derive
  // the local rank for NUMA-aware GPU binding. This is more robust than
  // m_rank % n_devices, which assumes a particular rank ordering.
  MPI_Comm local_comm;
  MPI_Comm_split_type(m_world, MPI_COMM_TYPE_SHARED, m_rank,
                      MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_free(&local_comm);

  int dev_id = local_rank % n_devices;
  Logger::print_info("Rank {} binding to GPU {} (local rank {})", m_rank,
                     dev_id, local_rank);
  GpuSafeCall(gpuSetDevice(dev_id));
  init_dev_rank(m_rank);
}

template class domain_comm<Config<1, Scalar>, exec_policy_gpu>;
template class domain_comm<Config<2, Scalar>, exec_policy_gpu>;
template class domain_comm<Config<3, Scalar>, exec_policy_gpu>;

}
