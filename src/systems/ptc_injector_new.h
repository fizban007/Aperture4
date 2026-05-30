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

// #ifndef _PTC_INJECTOR_NEW_H_
// #define _PTC_INJECTOR_NEW_H_
#pragma once

#include "core/cuda_control.h"
#include "core/gpu_translation_layer.h"

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
class ptc_injector;

// Default position validator for ptc_injector::inject_pairs. Returns true
// for every (pos, x_global), i.e. accept every candidate placement. Coord
// policies that need to exclude regions (e.g. the GR KS half-cell wedge
// at the theta boundaries) supply their own validator instead.
struct always_valid_pos {
  template <typename Pos, typename X>
  HD_INLINE bool operator()(const Pos&, const X&) const {
    return true;
  }
};

}  // namespace Aperture

#include "systems/ptc_injector_host.hpp"
#include "systems/ptc_injector_gpu.hpp"

namespace Aperture {

#ifdef GPU_ENABLED
template <typename Conf>
using ptc_injector_dynamic = ptc_injector<Conf, exec_policy_gpu>;
#else
template <typename Conf>
using ptc_injector_dynamic = ptc_injector<Conf, exec_policy_host>;
#endif

}

// #endif  // _PTC_INJECTOR_NEW_H_
