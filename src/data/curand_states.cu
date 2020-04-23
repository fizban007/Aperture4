#include "curand_states.h"
#include "framework/environment.hpp"
#include "utils/kernel_helper.hpp"

namespace Aperture {

curand_states_t::curand_states_t(sim_environment& env, size_t size) :
m_env(env) {
  m_states.set_memtype(MemType::host_device);
  m_states.resize(size);
}

void
curand_states_t::init() {
  m_env.params().get_value("random_seed", m_init_seed);

  kernel_launch([]__device__(auto states, int seed) {
      int id = threadIdx.x + blockIdx.x * blockDim.x;
      curand_init(seed, id, 0, &states[id]);
    }, m_states.dev_ptr(), m_init_seed);
}

}
