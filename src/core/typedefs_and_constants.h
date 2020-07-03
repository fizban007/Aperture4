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

#ifndef _TYPEDEFS_N_CONSTANTS_H_
#define _TYPEDEFS_N_CONSTANTS_H_

#include <limits>
#include <cstdint>
#include "enum_types.h"

namespace Aperture {

#ifdef CUDA_ENABLED
constexpr MemType default_mem_type = MemType::host_device;
#else
constexpr MemType default_mem_type = MemType::host_only;
#endif
constexpr uint32_t empty_cell = std::numeric_limits<uint32_t>::max();
constexpr float eps_float = 1.0e-8f;
constexpr double eps_double = 1.0e-14;
constexpr int default_interp_order = 1;

#ifndef USE_DOUBLE
typedef float Scalar;
typedef float Mom_t;
typedef float Pos_t;
constexpr float TINY = eps_float;
#else
typedef double Scalar;
typedef double Mom_t;
typedef double Pos_t;
constexpr double TINY = eps_double;
#endif

}  // namespace Aperture

#endif  // _TYPEDEFS_N_CONSTANTS_H_
