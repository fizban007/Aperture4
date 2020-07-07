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

#ifndef _RT_MAGNETAR_H_
#define _RT_MAGNETAR_H_

#include "systems/radiative_transfer.h"

namespace Aperture {

template <typename Conf>
struct rt_magnetar_impl_t;

template <typename Conf>
using rt_magnetar = radiative_transfer_cu<Conf, rt_magnetar_impl_t<Conf>>;

}



#endif  // _RT_MAGNETAR_H_
