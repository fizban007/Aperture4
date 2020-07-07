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

#ifndef _PH_FREEPATH_DEV_H_
#define _PH_FREEPATH_DEV_H_

#include "systems/radiative_transfer.h"

namespace Aperture {

template <typename Conf>
struct ph_freepath_dev_t;

template <typename Conf>
using ph_freepath_dev = radiative_transfer_cu<Conf, ph_freepath_dev_t<Conf>>;

}

#endif  // _PH_FREEPATH_DEV_H_
