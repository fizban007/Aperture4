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

#include "ptc_updater_simd.h"
#include "core/simd.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/helpers/filter_field.hpp"
#include "systems/helpers/ptc_update_helper.hpp"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/exec_policy_omp_simd.hpp"
#include "utils/interpolation_simd.hpp"

namespace Aperture {

template <typename Conf, template <class> class CoordPolicy,
          template <class> class PhysicsPolicy>
void
ptc_updater_simd<Conf, CoordPolicy, PhysicsPolicy>::update_particles(
    value_t dt, uint32_t step) {
  auto num = this->ptc->number();
  if (num == 0) return;
  int rho_interval = this->m_rho_interval;
  auto charges = this->m_charges;
  auto masses = this->m_masses;
  auto coord_policy = *(this->m_coord_policy);
  bool deposit_rho = (step % rho_interval == 0);

  // using namespace simd;

  // Main particle update loop
  exec_policy_openmp_simd<Conf>::launch(
      [num, dt, rho_interval, deposit_rho, charges, masses,
       coord_policy] LAMBDA(auto ptc, auto E, auto B, auto J, auto Rho) {
        auto& grid = exec_policy_openmp_simd<Conf>::grid();
        auto ext = grid.extent();
        auto interp =
            simd::interpolator<simd::bspline<Conf::interp_order>, Conf::dim>{};
        exec_policy_openmp_simd<Conf>::loop(
            0ul, num,
            [&ext, &charges, &masses, &coord_policy, dt, deposit_rho,
             interp] LAMBDA(auto n, auto& ptc, auto& E, auto& B, auto& J,
                            auto& Rho, auto& grid) {
              ptc_context<Conf::dim, simd::Vec_i_t, simd::Vec_i_t,
                          simd::Vec_f_t>
                  context;
              context.cell.load(&ptc.cell[n]);
              // if (context.cell == empty_cell) return;
              simd::Vec_ib_t empty_mask =
                  (context.cell != simd::Vec_ui_t(empty_cell));
              context.cell = select(empty_mask, context.cell, 0);

              // auto idx = Conf::idx(context.cell, ext);
              // auto pos = get_pos(context.cell, ext);
              vec_t<simd::Vec_i_t, Conf::dim> pos;

              context.x[0].load(ptc.x1 + n);
              context.x[1].load(ptc.x2 + n);
              context.x[2].load(ptc.x3 + n);
              context.p[0].load(ptc.p1 + n);
              context.p[1].load(ptc.p2 + n);
              context.p[2].load(ptc.p3 + n);
              context.gamma.load(ptc.E + n);

              context.flag.load(ptc.flag + n);
              context.sp = get_ptc_type(context.flag);

              context.weight.load(ptc.weight + n);
              simd::Vec_f_t qs =
                  lookup<simd::vec_width>(context.sp, charges.data());
              simd::Vec_f_t ms =
                  lookup<simd::vec_width>(context.sp, masses.data());
              context.weight *= qs;

              context.E[0] = interp(E[0].p, context.x, context.cell, ext,
                                    stagger_t(0b110));
              context.E[1] = interp(E[1].p, context.x, context.cell, ext,
                                    stagger_t(0b101));
              context.E[2] = interp(E[2].p, context.x, context.cell, ext,
                                    stagger_t(0b011));
              context.B[0] = interp(B[0].p, context.x, context.cell, ext,
                                    stagger_t(0b001));
              context.B[1] = interp(B[1].p, context.x, context.cell, ext,
                                    stagger_t(0b010));
              context.B[2] = interp(B[2].p, context.x, context.cell, ext,
                                    stagger_t(0b100));

              // printf("x1: %f, x2: %f, p1: %f, p2: %f, q_over_m: %f, dt:
              // %f\n",
              //        context.x[0], context.x[1], context.p[0], context.p[1],
              //        charges[context.sp] / masses[context.sp], dt);

              coord_policy.update_ptc(grid, context, pos, qs / ms, dt);

              context.p[0].store(&ptc.p1[n]);
              context.p[1].store(&ptc.p2[n]);
              context.p[2].store(&ptc.p3[n]);
              context.gamma.store(&ptc.E[n]);

              simd::deposit_t<Conf::dim, typename Conf::spline_t> deposit{};
              for (int i = 0; i < simd::vec_width; i++) {
                auto idx = Conf::idx(context.cell[i], ext);
                deposit(i, context, J, Rho, idx, dt, deposit_rho);
              }

              context.new_x[0].store(&ptc.x1[n]);
              context.new_x[1].store(&ptc.x2[n]);
              context.new_x[2].store(&ptc.x3[n]);
              context.cell += context.dc.dot(ext.strides());
              context.cell.store(&ptc.cell[n]);
            },
            ptc, E, B, J, Rho, grid);
      },
      *(this->ptc), *(this->E), *(this->B), *(this->J), this->Rho);
  size_t iterated = (num / simd::vec_width) * simd::vec_width;

  // handle leftover particles
  base_class::update_particles(dt, step, iterated, num);

  coord_policy.template process_J_Rho<exec_policy_openmp_simd<Conf>>(
      *(this->J), this->Rho, dt, deposit_rho);

  this->filter_current(this->m_filter_times, step);
}

template class ptc_updater_simd<Config<1>, coord_policy_cartesian>;
template class ptc_updater_simd<Config<2>, coord_policy_cartesian>;
template class ptc_updater_simd<Config<3>, coord_policy_cartesian>;

}  // namespace Aperture
