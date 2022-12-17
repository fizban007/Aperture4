/*
 * Copyright (c) 2022 Alex Chen.
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

#ifndef GATHER_TRACKED_PTC_IMPL_H_
#define GATHER_TRACKED_PTC_IMPL_H_

#include "framework/environment.h"
#include "systems/gather_tracked_ptc.h"

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
void
gather_tracked_ptc<Conf, ExecPolicy>::register_data_components() {}

template <typename Conf, template <class> class ExecPolicy>
void
gather_tracked_ptc<Conf, ExecPolicy>::init() {
  int64_t max_tracked_num = 100000;
  sim_env().params().get_value("max_tracked_num", max_tracked_num);

  sim_env().get_data("particles", ptc);
  if (ptc != nullptr) {
    tracked_ptc = sim_env().register_data<tracked_particles_t>("tracked_ptc",
                                                               max_tracked_num);
  }
  sim_env().get_data("photons", ph);
  if (ph != nullptr) {
    tracked_ph = sim_env().register_data<tracked_photons_t>("tracked_ph",
                                                            max_tracked_num);
  }

  sim_env().get_data("E", E);
  sim_env().get_data("B", B);

  m_ptc_output_interval = 0;
  sim_env().params().get_value("ptc_output_interval", m_ptc_output_interval);
}

template <typename Conf, template <class> class ExecPolicy>
template <typename T, typename Func>
void
gather_tracked_ptc<Conf, ExecPolicy>::gather_tracked_attr(
    buffer<T>& data, const buffer<uint32_t>& tracked_map, size_t tracked_num,
    Func data_func) {
  ExecPolicy<Conf>::launch(
      [data_func, tracked_num] LAMBDA(auto data, auto tracked_map, auto ptc,
                                      auto ph, auto E, auto B) {
        ExecPolicy<Conf>::loop(0, tracked_num, [&] LAMBDA(auto n) {
          data[n] = data_func(tracked_map[n], ptc, ph, E, B);
        });
      },
      data, tracked_map, ptc, ph, E, B);
}

template <typename Conf, template <class> class ExecPolicy>
void
gather_tracked_ptc<Conf, ExecPolicy>::update(double dt, uint32_t step) {
  if (m_ptc_output_interval == 0) return;

  if (step % m_ptc_output_interval == 0) {
    auto ext = m_grid.extent();
    if (ptc != nullptr) {
      ptc->gather_tracked_ptc_map(tracked_ptc->size());
      size_t tracked_num = ptc->tracked_num()[0];
      tracked_ptc->set_number(tracked_num);
      auto& tracked_map = ptc->tracked_map();

      // Get positions
      gather_tracked_attr(
          tracked_ptc->x1, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            auto& grid = ExecPolicy<Conf>::grid();
            auto idx = Conf::idx(ptc.cell[n], ext);
            auto pos = get_pos(idx, ext);
            return grid.pos<0>(pos, ptc.x1[n]);
          });
      tracked_ptc->x1.copy_to_host();
      gather_tracked_attr(
          tracked_ptc->x2, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            auto& grid = ExecPolicy<Conf>::grid();
            auto idx = Conf::idx(ptc.cell[n], ext);
            auto pos = get_pos(idx, ext);
            return grid.pos<1>(pos, ptc.x2[n]);
          });
      tracked_ptc->x2.copy_to_host();
      gather_tracked_attr(
          tracked_ptc->x3, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            auto& grid = ExecPolicy<Conf>::grid();
            auto idx = Conf::idx(ptc.cell[n], ext);
            auto pos = get_pos(idx, ext);
            return grid.pos<2>(pos, ptc.x3[n]);
          });
      tracked_ptc->x3.copy_to_host();
      gather_tracked_attr(
          tracked_ptc->p1, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            return ptc.p1[n];
          });
      tracked_ptc->p1.copy_to_host();
      gather_tracked_attr(
          tracked_ptc->p2, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            return ptc.p2[n];
          });
      tracked_ptc->p2.copy_to_host();
      gather_tracked_attr(
          tracked_ptc->p3, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            return ptc.p3[n];
          });
      tracked_ptc->p3.copy_to_host();
      gather_tracked_attr(
          tracked_ptc->E, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            return ptc.E[n];
          });
      tracked_ptc->E.copy_to_host();
      gather_tracked_attr(
          tracked_ptc->weight, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            return ptc.weight[n];
          });
      tracked_ptc->weight.copy_to_host();
      gather_tracked_attr(
          tracked_ptc->flag, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            return ptc.flag[n];
          });
      tracked_ptc->flag.copy_to_host();
      gather_tracked_attr(
          tracked_ptc->id, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            return ptc.id[n];
          });
      tracked_ptc->id.copy_to_host();
    }
    if (ph != nullptr) {
      ph->gather_tracked_ptc_map(tracked_ph->size());
      size_t tracked_num = ph->tracked_num()[0];
      tracked_ph->set_number(tracked_num);
      auto& tracked_map = ph->tracked_map();

      // Get positions
      gather_tracked_attr(
          tracked_ph->x1, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            auto& grid = ExecPolicy<Conf>::grid();
            auto idx = Conf::idx(ph.cell[n], ext);
            auto pos = get_pos(idx, ext);
            return grid.pos<0>(pos, ph.x1[n]);
          });
      tracked_ph->x1.copy_to_host();
      gather_tracked_attr(
          tracked_ph->x2, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            auto& grid = ExecPolicy<Conf>::grid();
            auto idx = Conf::idx(ph.cell[n], ext);
            auto pos = get_pos(idx, ext);
            return grid.pos<1>(pos, ph.x2[n]);
          });
      tracked_ph->x2.copy_to_host();
      gather_tracked_attr(
          tracked_ph->x3, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            auto& grid = ExecPolicy<Conf>::grid();
            auto idx = Conf::idx(ph.cell[n], ext);
            auto pos = get_pos(idx, ext);
            return grid.pos<2>(pos, ph.x3[n]);
          });
      tracked_ph->x3.copy_to_host();
      gather_tracked_attr(
          tracked_ph->p1, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            return ph.p1[n];
          });
      tracked_ph->p1.copy_to_host();
      gather_tracked_attr(
          tracked_ph->p2, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            return ph.p2[n];
          });
      tracked_ph->p2.copy_to_host();
      gather_tracked_attr(
          tracked_ph->p3, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            return ph.p3[n];
          });
      tracked_ph->p3.copy_to_host();
      gather_tracked_attr(
          tracked_ph->E, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            return ph.E[n];
          });
      tracked_ph->E.copy_to_host();
      gather_tracked_attr(
          tracked_ph->weight, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            return ph.weight[n];
          });
      tracked_ph->weight.copy_to_host();
      gather_tracked_attr(
          tracked_ph->flag, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            return ph.flag[n];
          });
      tracked_ph->flag.copy_to_host();
      gather_tracked_attr(
          tracked_ph->id, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto ph, auto E, auto B) {
            return ph.id[n];
          });
      tracked_ph->id.copy_to_host();
    }
  }
}

}  // namespace Aperture

#endif  // GATHER_TRACKED_PTC_IMPL_H_
