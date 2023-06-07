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

#pragma once

#include "framework/environment.h"
#include "systems/gather_tracked_ptc.h"

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
void
gather_tracked_ptc<Conf, ExecPolicy>::register_data_components() {}

template <typename Conf, template <class> class ExecPolicy>
void
gather_tracked_ptc<Conf, ExecPolicy>::init() {
  // Logger::print_err("Initializing tracker system");
  sim_env().params().get_value("max_tracked_num", m_max_tracked);
  sim_env().params().get_value("ptc_output_interval", m_ptc_output_interval);
  sim_env().get_data("E", E);
  sim_env().get_data("B", B);

  if (m_max_tracked == 0) {
    return;
  }

  m_tracked_map.set_memtype(ExecPolicy<Conf>::data_mem_type());
  m_tracked_map.resize(m_max_tracked);
  m_tracked_num.set_memtype(ExecPolicy<Conf>::data_mem_type());
  m_tracked_num.resize(1);

  sim_env().get_data_optional("particles", ptc);
  if (ptc != nullptr) {
    tracked_ptc = sim_env().register_data<tracked_particles_t>(
        "tracked_ptc", m_max_tracked, ExecPolicy<Conf>::data_mem_type());
  }
  sim_env().get_data_optional("photons", ph);
  if (ph != nullptr) {
    tracked_ph = sim_env().register_data<tracked_photons_t>(
        "tracked_ph", m_max_tracked, ExecPolicy<Conf>::data_mem_type());
  }

}

template <typename Conf, template <class> class ExecPolicy>
template <typename T, typename Func>
void
gather_tracked_ptc<Conf, ExecPolicy>::gather_tracked_ptc_attr(
    buffer<T>& data, const buffer<uint32_t>& tracked_map, size_t tracked_num,
    Func data_func) {
  if (ptc != nullptr) {
    ExecPolicy<Conf>::launch(
        [data_func, tracked_num] LAMBDA(auto data, auto tracked_map, auto ptc,
                                        auto E, auto B) {
          ExecPolicy<Conf>::loop(0, tracked_num, [&] LAMBDA(auto n) {
            // printf("n is %du, map[n] is %du\n", n, tracked_map[n]);
            data[n] = data_func(tracked_map[n], ptc, E, B);
          });
        },
        data, tracked_map, ptc, E, B);
    data.copy_to_host(0, tracked_num);
  }
}

template <typename Conf, template <class> class ExecPolicy>
template <typename T, typename Func>
void
gather_tracked_ptc<Conf, ExecPolicy>::gather_tracked_ph_attr(
    buffer<T>& data, const buffer<uint32_t>& tracked_map, size_t tracked_num,
    Func data_func) {
  if (ph != nullptr) {
    ExecPolicy<Conf>::launch(
        [data_func, tracked_num] LAMBDA(auto data, auto tracked_map, auto ph,
                                        auto E, auto B) {
          ExecPolicy<Conf>::loop(0, tracked_num, [&] LAMBDA(auto n) {
            data[n] = data_func(tracked_map[n], ph, E, B);
          });
        },
        data, tracked_map, ph, E, B);
    data.copy_to_host(0, tracked_num);
  }
}

template <typename Conf, template <class> class ExecPolicy>
template <typename BufferType>
void
gather_tracked_ptc<Conf, ExecPolicy>::gather_tracked_ptc_index(
    const particles_base<BufferType>& ptc) {
  size_t number = ptc.number();
  size_t max_tracked = m_max_tracked;
  m_tracked_num[0] = 0;
  m_tracked_num.copy_to_device();
  // Logger::print_info_all("getting tracked index");
  ExecPolicy<Conf>::launch(
      [number, max_tracked] LAMBDA(auto flags, auto cells, auto tracked_map,
                                   auto tracked_num) {
        // for (auto n : grid_stride_range(0, number)) {
        ExecPolicy<Conf>::loop(0, number, [&] LAMBDA(auto n) {
          if (check_flag(flags[n], PtcFlag::tracked) &&
              cells[n] != empty_cell) {
            uint32_t nt = atomic_add(&tracked_num[0], 1);
            if (nt < max_tracked) {
              tracked_map[nt] = n;
            }
          }
        });
      },
      ptc.flag, ptc.cell, m_tracked_map, m_tracked_num);
  ExecPolicy<Conf>::sync();

  // Logger::print_info_all("finished getting tracked index");
  m_tracked_num.copy_to_host();
  Logger::print_debug_all("There are {} tracked particles, {} particles",
                          m_tracked_num[0], ptc.number());
  if (m_tracked_num[0] > max_tracked) {
    m_tracked_num[0] = max_tracked;
    m_tracked_num.copy_to_device();
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
gather_tracked_ptc<Conf, ExecPolicy>::update(double dt, uint32_t step) {
  if (m_ptc_output_interval == 0 || m_max_tracked == 0) return;

  // Logger::print_debug_all("gathering tracked particles, {}", m_ptc_output_interval);
  // Logger::print_info("tracked_ptc size is {}", tracked_ptc->size());
  if (step % m_ptc_output_interval == 0) {
    auto ext = m_grid.extent();
    if (ptc != nullptr) {
      gather_tracked_ptc_index(*ptc);
      // Logger::print_debug_all("gathered index");
      size_t tracked_num = m_tracked_num[0];
      tracked_ptc->set_number(tracked_num);
      auto& tracked_map = m_tracked_map;

      // Get positions
      // Logger::print_debug_all("before gather x1");
      gather_tracked_ptc_attr(
          tracked_ptc->x1, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto E, auto B) {
            // printf("n is %u, cell[n] is %u\n", n, ptc.cell[n]);
            auto& grid = ExecPolicy<Conf>::grid();
            auto idx = Conf::idx(ptc.cell[n], ext);
            auto pos = get_pos(idx, ext);
            return grid.template coord<0>(pos, ptc.x1[n]);
          });
      // Logger::print_debug_all("gathered x1");
      gather_tracked_ptc_attr(
          tracked_ptc->x2, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto E, auto B) {
            auto& grid = ExecPolicy<Conf>::grid();
            auto idx = Conf::idx(ptc.cell[n], ext);
            auto pos = get_pos(idx, ext);
            return grid.template coord<1>(pos, ptc.x2[n]);
          });
      gather_tracked_ptc_attr(
          tracked_ptc->x3, tracked_map, tracked_num,
          [ext] LAMBDA(uint32_t n, auto ptc, auto E, auto B) {
            auto& grid = ExecPolicy<Conf>::grid();
            auto idx = Conf::idx(ptc.cell[n], ext);
            auto pos = get_pos(idx, ext);
            return grid.template coord<2>(pos, ptc.x3[n]);
          });
      gather_tracked_ptc_attr(tracked_ptc->p1, tracked_map, tracked_num,
                              [ext] LAMBDA(uint32_t n, auto ptc, auto E,
                                           auto B) { return ptc.p1[n]; });
      gather_tracked_ptc_attr(tracked_ptc->p2, tracked_map, tracked_num,
                              [ext] LAMBDA(uint32_t n, auto ptc, auto E,
                                           auto B) { return ptc.p2[n]; });
      gather_tracked_ptc_attr(tracked_ptc->p3, tracked_map, tracked_num,
                              [ext] LAMBDA(uint32_t n, auto ptc, auto E,
                                           auto B) { return ptc.p3[n]; });
      gather_tracked_ptc_attr(tracked_ptc->E, tracked_map, tracked_num,
                              [ext] LAMBDA(uint32_t n, auto ptc, auto E,
                                           auto B) { return ptc.E[n]; });
      gather_tracked_ptc_attr(tracked_ptc->weight, tracked_map, tracked_num,
                              [ext] LAMBDA(uint32_t n, auto ptc, auto E,
                                           auto B) { return ptc.weight[n]; });
      gather_tracked_ptc_attr(tracked_ptc->flag, tracked_map, tracked_num,
                              [ext] LAMBDA(uint32_t n, auto ptc, auto E,
                                           auto B) { return ptc.flag[n]; });
      gather_tracked_ptc_attr(tracked_ptc->id, tracked_map, tracked_num,
                              [ext] LAMBDA(uint32_t n, auto ptc, auto E,
                                           auto B) { return ptc.id[n]; });
    }
    if (ph != nullptr) {
      gather_tracked_ptc_index(*ph);
      size_t tracked_num = m_tracked_num[0];
      tracked_ph->set_number(tracked_num);
      auto& tracked_map = m_tracked_map;

      // Get positions
      gather_tracked_ph_attr(tracked_ph->x1, tracked_map, tracked_num,
                             [ext] LAMBDA(uint32_t n, auto ph, auto E, auto B) {
                               auto& grid = ExecPolicy<Conf>::grid();
                               auto idx = Conf::idx(ph.cell[n], ext);
                               auto pos = get_pos(idx, ext);
                               return grid.template coord<0>(pos, ph.x1[n]);
                             });
      gather_tracked_ph_attr(tracked_ph->x2, tracked_map, tracked_num,
                             [ext] LAMBDA(uint32_t n, auto ph, auto E, auto B) {
                               auto& grid = ExecPolicy<Conf>::grid();
                               auto idx = Conf::idx(ph.cell[n], ext);
                               auto pos = get_pos(idx, ext);
                               return grid.template coord<1>(pos, ph.x2[n]);
                             });
      gather_tracked_ph_attr(tracked_ph->x3, tracked_map, tracked_num,
                             [ext] LAMBDA(uint32_t n, auto ph, auto E, auto B) {
                               auto& grid = ExecPolicy<Conf>::grid();
                               auto idx = Conf::idx(ph.cell[n], ext);
                               auto pos = get_pos(idx, ext);
                               return grid.template coord<2>(pos, ph.x3[n]);
                             });
      gather_tracked_ph_attr(tracked_ph->p1, tracked_map, tracked_num,
                             [ext] LAMBDA(uint32_t n, auto ph, auto E, auto B) {
                               return ph.p1[n];
                             });
      gather_tracked_ph_attr(tracked_ph->p2, tracked_map, tracked_num,
                             [ext] LAMBDA(uint32_t n, auto ph, auto E, auto B) {
                               return ph.p2[n];
                             });
      gather_tracked_ph_attr(tracked_ph->p3, tracked_map, tracked_num,
                             [ext] LAMBDA(uint32_t n, auto ph, auto E, auto B) {
                               return ph.p3[n];
                             });
      gather_tracked_ph_attr(tracked_ph->E, tracked_map, tracked_num,
                             [ext] LAMBDA(uint32_t n, auto ph, auto E, auto B) {
                               return ph.E[n];
                             });
      gather_tracked_ph_attr(tracked_ph->weight, tracked_map, tracked_num,
                             [ext] LAMBDA(uint32_t n, auto ph, auto E, auto B) {
                               return ph.weight[n];
                             });
      gather_tracked_ph_attr(tracked_ph->flag, tracked_map, tracked_num,
                             [ext] LAMBDA(uint32_t n, auto ph, auto E, auto B) {
                               return ph.flag[n];
                             });
      gather_tracked_ph_attr(tracked_ph->id, tracked_map, tracked_num,
                             [ext] LAMBDA(uint32_t n, auto ph, auto E, auto B) {
                               return ph.id[n];
                             });
    }
  }
}

}  // namespace Aperture

