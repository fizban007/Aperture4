#include "curand_states.h"
#include "framework/environment.hpp"
#include "utils/kernel_helper.hpp"

namespace Aperture {

curand_states_t::curand_states_t(size_t size, int seed) {
  m_states.set_memtype(MemType::host_device);
  m_states.resize(std::max(size, size_t(512 * 1024)));
  Logger::print_info("Resized the random state to {}", m_states.size());
  m_init_seed = seed;
}

void
curand_states_t::init() {
  kernel_launch(
      {512, 1024},
      [] __device__(auto states, int seed) {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(seed, id, 0, &states[id]);
      },
      m_states.dev_ptr(), m_init_seed);
  CudaSafeCall(cudaDeviceSynchronize());
}

}  // namespace Aperture
