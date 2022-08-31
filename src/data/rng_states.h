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
#include "core/gpu_translation_layer.h"
#include "core/gpu_error_check.h"
#include "core/data_adapter.h"
#include "core/random.h"
#include "framework/data.h"

namespace Aperture {

class rng_states_t : public data_t {
 public:
  rng_states_t(uint64_t seed = default_random_seed);
  ~rng_states_t();

  void init() override;

  rand_state* states() { return m_states; }
  const rand_state* states() const { return m_states; }
  char* host_states() { return m_states_host; }
  const char* host_states() const { return m_states_host; }

  void copy_to_device();
  void copy_to_host();

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  static constexpr int block_num = 512;
  static constexpr int thread_num = 1024;
#endif

 private:
  uint64_t m_initial_seed;
  size_t m_size;
  rand_state* m_states;
  char* m_states_host;
};

template<>
struct host_adapter<rng_states_t> {
  typedef rand_state* type;

  static inline type apply(rng_states_t& s) {
    return s.states();
  }
};

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)

template <>
struct gpu_adapter<rng_states_t> {
  typedef rand_state* type;

  static inline type apply(rng_states_t& s) {
    return s.states();
  }
};

#endif

}  // namespace Aperture

#endif  // __RNG_STATES_H_
