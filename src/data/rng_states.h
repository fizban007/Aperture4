/*
 * Copyright (c) 2021 Alex Chen.
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

#ifndef __RNG_STATES_H_
#define __RNG_STATES_H_

#include "core/buffer.hpp"
#include "core/cuda_control.h"
#include "core/data_adapter.h"
#include "core/typedefs_and_constants.h"
#include "framework/data.h"

#ifdef CUDA_ENABLED
#include <curand_kernel.h>
#endif

namespace Aperture {

#ifdef CUDA_ENABLED

typedef curandState rand_state;

#else

struct rand_state {
  uint64_t s[4] = {};

  rand_state() {}
  rand_state(uint64_t seed[4]) {
    for (int i = 0; i < 4; i++)
      s[i] = seed[i];
  }
};

#endif

class rng_states_t : public data_t {
 public:
  rng_states_t(uint64_t seed = default_random_seed);

  void init() override;

  buffer<rand_state>& states() { return m_states; }
  const buffer<rand_state>& states() const { return m_states; }

 private:
  uint64_t m_initial_seed;
  buffer<rand_state> m_states;
};

#ifdef CUDA_ENABLED

template <>
class cuda_adapter<rng_states_t> {
  typedef rand_state* type;

  static inline type apply(rng_states_t& s) {
    return s.states().dev_ptr();
  }
};

#endif

}  // namespace Aperture

#endif  // __RNG_STATES_H_
