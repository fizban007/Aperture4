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

#pragma once

#include "core/exec_tags.h"
#include "core/gpu_translation_layer.h"
#include "core/math.hpp"
#include "core/typedefs_and_constants.h"
#include "framework/environment.h"
#include "utils/type_traits.hpp"
#include "utils/vec.hpp"
#include <limits>

// #ifdef CUDA_ENABLED
// #include <curand_kernel.h>
// #elif HIP_ENABLED
// #include <rocrand/rocrand_kernel.h>
// #endif

namespace Aperture {

struct rand_state;

namespace detail {

constexpr HD_INLINE uint64_t
rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

constexpr HD_INLINE uint64_t
split_mix_64(uint64_t x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

}  // namespace detail

struct rand_state {
  uint64_t s[4] = {};

  HOST_DEVICE rand_state() {}

  HOST_DEVICE rand_state(uint64_t seed[4]) {
    for (int i = 0; i < 4; i++) s[i] = seed[i];
  }

  HOST_DEVICE void init(uint64_t seed = default_random_seed) {
    s[0] = detail::split_mix_64(seed);
    s[1] = detail::split_mix_64(s[0]);
    s[2] = detail::split_mix_64(s[1]);
    s[3] = detail::split_mix_64(s[2]);
  }

  /*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

  To the extent possible under law, the author has dedicated all copyright
  and related and neighboring rights to this software to the public domain
  worldwide. This software is distributed without any warranty.

  See <http://creativecommons.org/publicdomain/zero/1.0/>. */
  HOST_DEVICE uint64_t next() {
    const uint64_t result = s[0] + s[3];

    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = detail::rotl(s[3], 45);

    return result;
  }

  /* This is the jump function for the generator. It is equivalent
     to 2^128 calls to next(); it can be used to generate 2^128
     non-overlapping subsequences for parallel computations. */
  HOST_DEVICE void jump(void) {
    static const uint64_t JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
                                    0xa9582618e03fc9aa, 0x39abdc4529b1661c};

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;
    for (int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
      for (int b = 0; b < 64; b++) {
        if (JUMP[i] & UINT64_C(1) << b) {
          s0 ^= s[0];
          s1 ^= s[1];
          s2 ^= s[2];
          s3 ^= s[3];
        }
        next();
      }

    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
  }

  /* This is the long-jump function for the generator. It is equivalent to
     2^192 calls to next(); it can be used to generate 2^64 starting points,
     from each of which jump() will generate 2^64 non-overlapping
     subsequences for parallel distributed computations. */

  void long_jump(void) {
    static const uint64_t LONG_JUMP[] = {0x76e15d3efefdcbbf, 0xc5004e441c522fb3,
                                         0x77710069854ee241,
                                         0x39109bb02acbe635};

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;
    for (int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
      for (int b = 0; b < 64; b++) {
        if (LONG_JUMP[i] & UINT64_C(1) << b) {
          s0 ^= s[0];
          s1 ^= s[1];
          s2 ^= s[2];
          s3 ^= s[3];
        }
        next();
      }

    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
  }
};

template <typename Float = Scalar>
HD_INLINE Float
rng_uniform(rand_state& local_state) {
  uint64_t n = local_state.next();
  // return n / 18446744073709551616.0;
  return n / static_cast<double>(std::numeric_limits<uint64_t>::max());
}

template <typename Float>
HD_INLINE Float
rng_gaussian(rand_state& local_state, Float sigma) {
  auto u1 = rng_uniform<Float>(local_state);
  auto u2 = rng_uniform<Float>(local_state);
  return math::sqrt(-2.0f * math::log(u1)) * math::cos(2.0f * M_PI * u2) *
         sigma;
}

template <typename Float>
HD_INLINE int
rng_poisson(rand_state& local_state, Float lambda) {
  Float L = math::exp(-lambda);
  Float p = 1.0;
  int k = 0;
  do {
    k += 1;
    p *= rng_uniform<Float>(local_state);
  } while (p > L);
  return k - 1;
}

template <typename Float>
HD_INLINE Float
rng_maxwell_juttner(rand_state& local_state, Float theta) {
  // This is the Sobol algorithm described in Zenitani 2015
  Float u = 0.0f;
  if (theta > 0.1) {
    while (true) {
      auto x1 = rng_uniform<Float>(local_state);
      auto x2 = rng_uniform<Float>(local_state);
      auto x3 = rng_uniform<Float>(local_state);
      auto x4 = rng_uniform<Float>(local_state);
      u = -theta * math::log(x1 * x2 * x3);
      auto eta = -theta * math::log(x1 * x2 * x3 * x4);
      if (eta * eta - u * u > 1.0) {
        break;
      }
    }
  } else {
    u = rng_gaussian<Float>(local_state, math::sqrt(theta));
  }
  return u;
}

template <typename Float>
HD_INLINE vec_t<Float, 3>
rng_maxwell_juttner_3d(rand_state& local_state, Float theta) {
  vec_t<Float, 3> result;

  auto u = rng_maxwell_juttner(local_state, theta);
  auto x1 = rng_uniform<Float>(local_state);
  auto x2 = rng_uniform<Float>(local_state);

  result[0] = u * (2.0f * x1 - 1.0f);
  result[1] =
      2.0f * u * math::sqrt(x1 * (1.0f - x1)) * math::cos(2.0f * M_PI * x2);
  result[2] =
      2.0f * u * math::sqrt(x1 * (1.0f - x1)) * math::sin(2.0f * M_PI * x2);
  return result;
}

template <typename Float>
HD_INLINE vec_t<Float, 3>
rng_maxwell_juttner_drifting(rand_state& local_state, Float theta,
                             type_identity_t<Float> beta) {
  vec_t<Float, 3> u = rng_maxwell_juttner_3d(local_state, theta);
  auto G = 1.0f / math::sqrt(1.0f - beta * beta);
  auto u0 = math::sqrt(1.0f + u.dot(u));

  auto x1 = rng_uniform<Float>(local_state);
  if (-beta * u[0] / u0 > x1) u[0] = -u[0];

  u[0] = G * (u[0] + beta * u0);

  return u;
}

// A local rng to capture the thread-local rand_state
template <typename ExecTag>
struct rng_t;

#if (defined(CUDA_ENABLED) && defined(__CUDACC__)) || \
    (defined(HIP_ENABLED) && defined(__HIPCC__))

template <>
struct rng_t<exec_tags::device> {
  __device__ rng_t(rand_state* state) {
    id = threadIdx.x + blockIdx.x * blockDim.x;
    m_state = state;
    m_local_state = m_state[id];
  }
  __device__ ~rng_t() { m_state[id] = m_local_state; }

  // Generates a device random number between 0.0 and 1.0
  template <typename Float>
  __device__ __forceinline__ Float uniform() {
    return rng_uniform<Float>(m_local_state);
  }

  template <typename Float>
  __device__ __forceinline__ Float gaussian(Float sigma) {
    return rng_gaussian(m_local_state, sigma);
  }

  template <typename Float>
  __device__ __forceinline__ int poisson(Float lambda) {
    return rng_poisson(m_local_state, lambda);
  }

  template <typename Float>
  __device__ Float maxwell_juttner(Float theta) {
    return rng_maxwell_juttner(m_local_state, theta);
  }

  template <typename Float>
  __device__ vec_t<Float, 3> maxwell_juttner_3d(Float theta) {
    return rng_maxwell_juttner_3d(m_local_state, theta);
  }

  template <typename Float>
  __device__ vec_t<Float, 3> maxwell_juttner_drifting(
      Float theta, type_identity_t<Float> beta) {
    return rng_maxwell_juttner_drifting(m_local_state, theta, beta);
  }

  int id;
  rand_state* m_state;
  rand_state m_local_state;
};

#endif

template <>
struct rng_t<exec_tags::host> {
  rand_state& m_local_state;

  rng_t(rand_state* state) : m_local_state(*state) {}

  template <typename Float>
  inline Float uniform() {
    return rng_uniform<Float>(m_local_state);
  }

  template <typename Float>
  inline Float gaussian(Float sigma) {
    return rng_gaussian(m_local_state, sigma);
  }

  template <typename Float>
  inline int poisson(Float lambda) {
    return rng_poisson(m_local_state, lambda);
  }

  template <typename Float>
  inline Float maxwell_juttner(Float theta) {
    return rng_maxwell_juttner(m_local_state, theta);
  }

  template <typename Float>
  inline vec_t<Float, 3> maxwell_juttner_3d(Float theta) {
    return rng_maxwell_juttner_3d(m_local_state, theta);
  }

  template <typename Float>
  inline vec_t<Float, 3> maxwell_juttner_drifting(Float theta,
                                                  type_identity_t<Float> beta) {
    return rng_maxwell_juttner_drifting(m_local_state, theta, beta);
  }
};

}  // namespace Aperture
