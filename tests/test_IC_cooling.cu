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

#include "core/math.hpp"
#include "cxxopts.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/exec_policy_cuda.hpp"
#include "systems/policies/phys_policy_IC_cooling.hpp"
#include "systems/ptc_updater_base_impl.hpp"
#include "utils/hdf_wrapper.h"
#include "utils/logger.h"
#include "utils/vec.hpp"
#include <fstream>
#include <memory>
#include <vector>

namespace Aperture {

template class ptc_updater<Config<2>, exec_policy_cuda,
                               coord_policy_cartesian, phys_policy_IC_cooling>;

}  // namespace Aperture

using namespace Aperture;

int
main(int argc, char* argv[]) {
  return 0;
}
