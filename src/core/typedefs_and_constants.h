/*
 * Copyright (c) 2020 Alex Chen.
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

#include "enum_types.h"
#include <cstdint>
#include <limits>

namespace Aperture {

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
constexpr MemType default_mem_type = MemType::host_device;
#else
constexpr MemType default_mem_type = MemType::host_only;
#endif
constexpr uint32_t empty_cell = std::numeric_limits<uint32_t>::max();
constexpr float eps_float = 1.0e-8f;
constexpr double eps_double = 1.0e-14;
constexpr int default_interp_order = 3;

#ifndef USE_DOUBLE
typedef float Scalar;
// typedef float Mom_t;
// typedef float Pos_t;
// constexpr float TINY = eps_float;
#define TINY 1.0e-8f
#else
typedef double Scalar;
// typedef double Mom_t;
// typedef double Pos_t;
// constexpr double TINY = eps_double;
#define TINY 1.0e-14
#endif

constexpr uint64_t default_random_seed = 0x3141592653589793;

}  // namespace Aperture
