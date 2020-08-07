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

#ifndef _PTC_INJECTOR_PULSAR_H_
#define _PTC_INJECTOR_PULSAR_H_


#include "systems/ptc_injector.h"

namespace Aperture {

template <typename Conf>
class ptc_injector_pulsar : public ptc_injector_cu<Conf> {
 public:
  typedef typename Conf::value_t value_t;
  typedef nvstd::function<value_t(value_t, value_t, value_t)> weight_func_t;
  static std::string name() { return "ptc_injector_pulsar"; }

  using ptc_injector_cu<Conf>::ptc_injector_cu;

  void update(double dt, uint32_t step) override;
  void init() override;
  // template <typename WeightFunc>
  // void add_injector(const vec_t<value_t, Conf::dim>& lower,
  //                   const vec_t<value_t, Conf::dim>& size, value_t inj_rate,
  //                   value_t p0, value_t inj_weight, WeightFunc f) {
  //   ptc_injector_cu<Conf>::add_injector(lower, size, inj_rate, inj_weight, f);

  // }

 protected:
  vector_field<Conf> *B;
};


}


#endif  // _PTC_INJECTOR_PULSAR_H_
