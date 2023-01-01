/*
 * Copyright (c) 2022 Alex Chen.
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

#include "core/gpu_translation_layer.h"

#ifdef GPU_ENABLED
#include "systems/policies/exec_policy_gpu.hpp"
namespace Aperture {
template <typename Conf>
using exec_policy_dynamic = exec_policy_gpu<Conf>;
}
#else
#include "systems/policies/exec_policy_host.hpp"
namespace Aperture {
template <typename Conf>
using exec_policy_dynamic = exec_policy_host<Conf>;
}
#endif
