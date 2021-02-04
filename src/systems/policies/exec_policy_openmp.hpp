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

#ifndef __EXEC_POLICY_OPENMP_H_
#define __EXEC_POLICY_OPENMP_H_

#include "core/multi_array.hpp"
#include "data/fields.h"
#include "data/particle_data.h"
#include "framework/environment.h"
#include "utils/nonown_ptr.hpp"
#include "utils/singleton_holder.h"
#include <omp.h>

namespace Aperture {

template <typename Conf>
class exec_policy_openmp {
 public:
  static void set_grid(const grid_t<Conf>& grid) {
    m_grid = &grid;
  }

  template <typename Func, typename... Args>
  static void launch(const Func& f, Args&&... args) {
    f(adapt_host(args)...);
  }

  template <typename Func, typename Idx, typename... Args>
  static void loop(const Func& f, Idx begin, Idx end, Args&&... args) {
#pragma omp parallel for
    for (auto idx : range(begin, end)) {
      f(idx, args...);
    }
  }

  static void sync() {}

  static const Grid<Conf::dim, typename Conf::value_t>& grid() {
    return *m_grid;
  }

  static MemType data_mem_type() { return MemType::host_only; }
  static MemType tmp_mem_type() { return MemType::host_only; }

 private:
  static const Grid<Conf::dim, typename Conf::value_t>* m_grid;
};

template <typename Conf>
const Grid<Conf::dim, typename Conf::value_t>*
exec_policy_openmp<Conf>::m_grid = nullptr;

// template <typename Conf>
// using exec_policy_host = singleton_holder<exec_policy_host_impl<Conf>>;

}  // namespace Aperture

#endif  // __EXEC_POLICY_OPENMP_H_
