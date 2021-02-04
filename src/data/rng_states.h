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
#include "core/random.h"
#include "framework/data.h"

namespace Aperture {

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

template<>
struct host_adapter<rng_states_t> {
  typedef rand_state* type;

  static inline type apply(rng_states_t& s) {
    return s.states().host_ptr();
  }
};

#ifdef CUDA_ENABLED

template <>
struct cuda_adapter<rng_states_t> {
  typedef rand_state* type;

  static inline type apply(rng_states_t& s) {
    return s.states().dev_ptr();
  }
};

#endif

}  // namespace Aperture

#endif  // __RNG_STATES_H_
