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

#include "framework/config.h"
#include "ptc_updater_pulsar.h"
#include "systems/grid_sph.h"
#include "systems/helpers/ptc_update_helper.hpp"
#include "systems/physics/gravity.hpp"
#include "systems/physics/sync_cooling.hpp"
#include "utils/kernel_helper.hpp"

namespace Aperture {

template <typename Pusher>
struct pusher_impl_pulsar {
  Pusher pusher;
  double cooling_coef = 0.0, B0 = 1.0;

  HOST_DEVICE pusher_impl_pulsar() {}
  pusher_impl_pulsar(sim_environment& env) {
    env.params().get_value("sync_cooling_coef", cooling_coef);
    env.params().get_value("Bp", B0);
  }

  HOST_DEVICE pusher_impl_pulsar(const pusher_impl_pulsar<Pusher>& other) = default;

  template <typename value_t>
  __device__ void operator()(ptc_ptrs& ptc, uint32_t n, EB_t<value_t>& EB,
                             value_t qdt_over_2m, value_t dt) {
    auto &p1 = ptc.p1[n];
    auto &p2 = ptc.p2[n];
    auto &p3 = ptc.p3[n];
    auto &gamma = ptc.E[n];
    // printf("before push, p1 is %f, p2 is %f, p3 is %f, gamma is %f\n", p1, p2, p3, gamma);
    pusher(p1, p2, p3, gamma, EB.E1, EB.E2, EB.E3, EB.B1, EB.B2, EB.B3,
           qdt_over_2m, dt);

    // sync_kill_perp(p1, p2, p3, gamma, EB.E1, EB.E2, EB.E3, EB.B1, EB.B2, EB.B3,
    //                qdt_over_2m * 2.0f / dt, (Scalar)cooling_coef, (Scalar)B0);
    // printf("after push, p1 is %f, p2 is %f, p3 is %f, gamma is %f\n", p1, p2, p3, gamma);
    sync_kill_gyration(p1, p2, p3, gamma, EB.E1, EB.E2, EB.E3, EB.B1, EB.B2, EB.B3,
                       qdt_over_2m * 2.0f / dt, (value_t)cooling_coef, (value_t)B0);
    // printf("gamma is %f\n", gamma);
  }
};

template <typename Conf>
ptc_updater_pulsar<Conf>::~ptc_updater_pulsar() {
  if (m_impl_boris != nullptr) delete m_impl_boris;
  if (m_impl_vay != nullptr) delete m_impl_vay;
  if (m_impl_higuera != nullptr) delete m_impl_higuera;
}

template <typename Conf>
void
ptc_updater_pulsar<Conf>::init() {
  ptc_updater_old_sph_cu<Conf>::init();

  if (this->m_pusher == Pusher::boris) {
    m_impl_boris = new pusher_impl_pulsar<boris_pusher>(this->m_env);
  } else if (this->m_pusher == Pusher::vay) {
    m_impl_vay = new pusher_impl_pulsar<vay_pusher>(this->m_env);
  } else if (this->m_pusher == Pusher::higuera) {
    m_impl_higuera = new pusher_impl_pulsar<higuera_pusher>(this->m_env);
  }
}

template <typename Conf>
void
ptc_updater_pulsar<Conf>::push_default(value_t dt) {
  // dispatch according to enum. This will also instantiate all the versions of
  // push
  if (this->m_pusher == Pusher::boris) {
    this->push(dt, *m_impl_boris);
  } else if (this->m_pusher == Pusher::vay) {
    this->push(dt, *m_impl_vay);
  } else if (this->m_pusher == Pusher::higuera) {
    this->push(dt, *m_impl_higuera);
  }
}

#include "systems/ptc_updater_cu_impl.hpp"

template class ptc_updater_pulsar<Config<2, double>>;
template class ptc_updater_pulsar<Config<2, float>>;

}
