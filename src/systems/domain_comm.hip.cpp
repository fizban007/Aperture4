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
  // TODO: This way of finding device id may not be reliable
  int dev_id = m_rank % n_devices;
  // std::cout << "Rank " << m_rank << " is on device #" << dev_id <<
  // std::endl;
  GpuSafeCall(gpuSetDevice(dev_id));
  init_dev_rank(m_rank);
}

template class domain_comm<Config<1, Scalar>, exec_policy_gpu>;
template class domain_comm<Config<2, Scalar>, exec_policy_gpu>;
template class domain_comm<Config<3, Scalar>, exec_policy_gpu>;

}
