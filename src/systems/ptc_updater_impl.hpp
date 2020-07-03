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

#ifndef __PTC_UPDATER_IMPL_H_
#define __PTC_UPDATER_IMPL_H_

template <typename Conf>
template <typename P>
void
ptc_updater<Conf>::push(double dt, P& pusher) {
  auto num = ptc->number();
  auto ext = m_grid.extent();
  if (num > 0) {
    for (auto n : range(0, num)) {
      uint32_t cell = ptc->cell[n];
      if (cell == empty_cell) continue;
      auto idx = E->at(0).idx_at(cell);
      // auto pos = idx.get_pos();

      auto interp = interpolator<spline_t, Conf::dim>{};
      auto flag = ptc->flag[n];
      int sp = get_ptc_type(flag);

      Scalar qdt_over_2m = dt * 0.5f * m_q_over_m[sp];

      auto x = vec_t<Pos_t, 3>(ptc->x1[n], ptc->x2[n], ptc->x3[n]);
      //  Grab E & M fields at the particle position
      EB_t<value_t> EB;
      EB.E1 = interp(E->at(0), x, idx, stagger_t(0b110));
      EB.E2 = interp(E->at(1), x, idx, stagger_t(0b101));
      EB.E3 = interp(E->at(2), x, idx, stagger_t(0b011));
      EB.B1 = interp(B->at(0), x, idx, stagger_t(0b001));
      EB.B2 = interp(B->at(1), x, idx, stagger_t(0b010));
      EB.B3 = interp(B->at(2), x, idx, stagger_t(0b100));

      // Logger::print_debug("E1 {}, E2 {}, E3 {}, B1 {}, B2 {}, B3 {}",
      //                     E1, E2, E3, B1, B2, B3);

      //  Push particles
      if (!check_flag(flag, PtcFlag::ignore_EM)) {
        pusher(ptc->get_host_ptrs(), n, EB, qdt_over_2m,
               (Scalar)dt);
      }

      // if (dev_params.rad_cooling_on && sp != (int)ParticleType::ion) {
      //   sync_kill_perp(p1, p2, p3, gamma, B1, B2, B3, E1, E2, E3,
      //                  q_over_m);
      // }
      auto gamma = ptc->E[n];
      if (gamma != gamma) {
        printf(
            "NaN detected after push! p1 is %f, p2 is %f, p3 is %f, gamma "
            "is %f\n",
            ptc->p1[n], ptc->p2[n], ptc->p3[n], gamma);
        exit(1);
      }
    }
  }
}


#endif // __PTC_UPDATER_IMPL_H_
