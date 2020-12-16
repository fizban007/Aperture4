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

#ifndef _PTC_UPDATER_PULSAR_H_
#define _PTC_UPDATER_PULSAR_H_

#include "data/multi_array_data.hpp"
#include "systems/ptc_updater_sph.h"
#include <memory>

namespace Aperture {

template <typename Pusher>
struct pusher_impl_pulsar;

template <typename Conf>
class ptc_updater_pulsar : public ptc_updater_sph_cu<Conf> {
 public:
  typedef typename Conf::value_t value_t;
  static std::string name() { return "ptc_updater"; }

  using ptc_updater_sph_cu<Conf>::ptc_updater_sph_cu;
  ~ptc_updater_pulsar();

  void init() override;
  void push_default(value_t dt) override;

 protected:
  pusher_impl_pulsar<boris_pusher>* m_impl_boris = nullptr;
  pusher_impl_pulsar<vay_pusher>* m_impl_vay = nullptr;
  pusher_impl_pulsar<higuera_pusher>* m_impl_higuera = nullptr;
};

}



#endif  // _PTC_UPDATER_PULSAR_H_
