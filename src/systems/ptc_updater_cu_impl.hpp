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
ptc_updater_cu<Conf>::push(double delta_t, P& pusher) {
  value_t dt = delta_t;
  auto num = this->ptc->number();
  auto ext = this->m_grid.extent();

  auto pusher_kernel = [dt, num, ext] __device__(auto ptc, auto E, auto B,
                                                 auto pusher) {
    using value_t = typename Conf::value_t;
    for (auto n : grid_stride_range(0, num)) {
      uint32_t cell = ptc.cell[n];
      if (cell == empty_cell) continue;
      auto idx = E[0].idx_at(cell, ext);
      // auto pos = idx.get_pos();

      auto interp = interpolator<typename Conf::spline_t, Conf::dim>{};
      auto flag = ptc.flag[n];
      int sp = get_ptc_type(flag);

      value_t qdt_over_2m = dt * 0.5f * dev_charges[sp] / dev_masses[sp];

      auto x = vec_t<value_t, 3>(ptc.x1[n], ptc.x2[n], ptc.x3[n]);
      //  Grab E & M fields at the particle position
      EB_t<value_t> EB;
      EB.E1 = interp(E[0], x, idx, stagger_t(0b110));
      EB.E2 = interp(E[1], x, idx, stagger_t(0b101));
      EB.E3 = interp(E[2], x, idx, stagger_t(0b011));
      EB.B1 = interp(B[0], x, idx, stagger_t(0b001));
      EB.B2 = interp(B[1], x, idx, stagger_t(0b010));
      EB.B3 = interp(B[2], x, idx, stagger_t(0b100));

      //  Push particles
      // value_t p1 = ptc.p1[n], p2 = ptc.p2[n], p3 = ptc.p3[n],
      if (!check_flag(flag, PtcFlag::ignore_EM)) {
        // pusher(p1, p2, p3, gamma, E1, E2, E3, B1, B2, B3, qdt_over_2m,
        //        (value_t)dt);
        pusher(ptc, n, EB, qdt_over_2m, dt);
      }

      // if (dev_params.rad_cooling_on && sp != (int)ParticleType::ion) {
      //   sync_kill_perp(p1, p2, p3, gamma, B1, B2, B3, E1, E2, E3,
      //                  q_over_m);
      // }
      auto gamma = ptc.E[n];
      if (gamma != gamma) {
        printf(
            "NaN detected in push! p1 is %f, p2 is %f, p3 is %f, gamma "
            "is %f\n",
            ptc.p1[n], ptc.p2[n], ptc.p3[n], gamma);
        asm("trap;");
      }

    }
  };

  if (num > 0) {
    // exec_policy p;
    // configure_grid(p, pusher_kernel, this->ptc->dev_ptrs(),
    // this->E->get_ptrs(),
    //               this->B->get_ptrs(), pusher);
    // Logger::print_info(
    //     "pusher kernel: block_size: {}, grid_size: {}, shared_mem: {}",
    //     p.get_block_size(), p.get_grid_size(), p.get_shared_mem_bytes());

    kernel_launch(pusher_kernel, this->ptc->dev_ptrs(), this->E->get_ptrs(),
                  this->B->get_ptrs(), pusher);
  }
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}


#endif // __PTC_UPDATER_IMPL_H_
