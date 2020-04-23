#ifndef _CURAND_STATES_H_
#define _CURAND_STATES_H_

#ifdef CUDA_ENABLED
#include "framework/data.h"
#include "utils/buffer.h"
#include "core/cuda_control.h"

#include <curand_kernel.h>

namespace Aperture {

class sim_environment;

// Helper struct to plug into kernels
struct cuda_rng_t {
  HOST_DEVICE cuda_rng_t(curandState* state) : m_state(state) {
    m_local_state = *state;
  }
  HOST_DEVICE ~cuda_rng_t() {
    *m_state = m_local_state;
  }

  // Generates a device random number between 0.0 and 1.0
  __device__ __forceinline__ float operator()() {
    return curand_uniform(&m_local_state);
  }

  curandState* m_state;
  curandState m_local_state;
};

class curand_states_t : public data_t {
 private:
  buffer_t<curandState> m_states;
  sim_environment& m_env;
  int m_init_seed = 1234;
  int m_rand_state_size = sizeof(curandState);

 public:
  curand_states_t(sim_environment& env, size_t size);
  void init() override;

  inline curandState* states() { return m_states.dev_ptr(); }
};

}

#endif

#endif  // _CURAND_STATES_H_
