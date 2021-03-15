#include "core/random.h"
#include "data/rng_states.h"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"

using namespace Aperture;

int main() {
  rng_states_t states;
  states.init();

  kernel_launch([] __device__ (auto states) {
      rng_t rng(states);
      for(int i : grid_stride_range(0, 100)) {
        auto u = rng.maxwell_juttner_drifting(0.1, 0.5);
        printf("(%f, %f, %f)\n", u[0], u[1], u[2]);
      }
    }, states.states().dev_ptr());
  cudaDeviceSynchronize();

  return 0;
}
