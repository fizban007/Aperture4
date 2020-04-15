#include "core/constant_mem_func.h"
#include "framework/config.h"
#include "ptc_updater.h"
#include "utils/interpolation.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
void
ptc_updater<Conf>::init() {
  init_charge_mass();
  init_dev_charge_mass(m_charges, m_masses);

  Etmp = vector_field<Conf>(m_grid);
  Btmp = vector_field<Conf>(m_grid);
}

template <typename Conf>
void
ptc_updater<Conf>::update(double dt, uint32_t step) {}

template <typename Conf>
void
ptc_updater<Conf>::push(double dt) {
  // TODO: First interpolate E and B fields to vertices and store them in Etmp
  // and Btmp

  auto num = ptc->number();
  auto ext = m_grid.extent();
  if (num > 0) {
    kernel_launch(
        [dt, num, ext] __device__(auto ptrs, auto E, auto B, auto pusher) {
          for (auto n : grid_stride_range(0ul, num)) {
            uint32_t cell = ptrs.cell[n];
            if (cell == empty_cell) continue;
            auto idx = E[0].idx_at(cell, ext);
            auto pos = idx.get_pos();

            auto interp = interpolator<bspline<1>, Conf::dim>{};
            auto flag = ptrs.flag[n];
            int sp = get_ptc_type(flag);

            Scalar qdt_over_2m = dt * 0.5f * dev_charges[sp] / dev_masses[sp];

            auto x = vec_t<Pos_t, 3>(ptrs.x1[n], ptrs.x2[n], ptrs.x3[n]);
            //  Grab E & M fields at the particle position
            Scalar E1 = interp(E[0], x, idx, pos);
            Scalar E2 = interp(E[1], x, idx, pos);
            Scalar E3 = interp(E[2], x, idx, pos);
            Scalar B1 = interp(B[0], x, idx, pos);
            Scalar B2 = interp(B[1], x, idx, pos);
            Scalar B3 = interp(B[2], x, idx, pos);

            //  Push particles
            Scalar p1 = ptrs.p1[n], p2 = ptrs.p2[n], p3 = ptrs.p3[n],
                gamma = ptrs.E[n];
            if (p1 != p1 || p2 != p2 || p3 != p3) {
              printf(
                  "NaN detected in push! p1 is %f, p2 is %f, p3 is %f, gamma "
                  "is %f\n",
                  p1, p2, p3, gamma);
              asm("trap;");
            }

            if (!check_flag(flag, PtcFlag::ignore_EM)) {
              pusher(p1, p2, p3, gamma, E1, E2, E3, B1, B2, B3,
                     qdt_over_2m, (Scalar)dt);
            }

            // if (dev_params.rad_cooling_on && sp != (int)ParticleType::ion) {
            //   sync_kill_perp(p1, p2, p3, gamma, B1, B2, B3, E1, E2, E3,
            //                  q_over_m);
            // }
            ptrs.p1[n] = p1;
            ptrs.p2[n] = p2;
            ptrs.p3[n] = p3;
            ptrs.E[n] = gamma;
          }
        },
        ptc->dev_ptrs(), E->get_ptrs(), B->get_ptrs(), m_pusher);
  }
}

template class ptc_updater<Config<1>>;
template class ptc_updater<Config<2>>;
template class ptc_updater<Config<3>>;

}  // namespace Aperture
