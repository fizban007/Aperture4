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

#ifndef __DEFAULT_RADIATION_SCHEME_H_
#define __DEFAULT_RADIATION_SCHEME_H_

#include "fixed_photon_path.hpp"
#include "photon_pair_creation.hpp"
#include "threshold_emission.hpp"

namespace Aperture {

template <typename Conf>
struct default_radiation_scheme : public fixed_photon_path<Conf>,
                                  public photon_pair_creation<Conf>,
                                  public threshold_emission {
  void init() {
    fixed_photon_path<Conf>::init();
    photon_pair_creation<Conf>::init();
    threshold_emission::init();
  }
};

}


#endif // __DEFAULT_RADIATION_SCHEME_H_
