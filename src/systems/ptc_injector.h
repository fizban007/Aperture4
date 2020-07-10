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

#ifndef _PTC_INJECTOR_H_
#define _PTC_INJECTOR_H_

#include "core/enum_types.h"
#include "core/multi_array.hpp"
#include "data/fields.h"
#include "data/particle_data.h"
#include "framework/system.h"
#include "systems/grid.h"
#include <functional>
#include <memory>

#ifdef CUDA_ENABLED
#include "utils/kernel_helper.hpp"
#include <nvfunctional>
#endif

namespace Aperture {

class curand_states_t;

template <typename Conf>
class ptc_injector : public system_t {
 public:
  typedef typename Conf::value_t value_t;
  static std::string name() { return "ptc_injector"; }

  ptc_injector(sim_environment& env, const grid_t<Conf>& grid)
      : system_t(env), m_grid(grid) {}
  virtual ~ptc_injector() {}

  void add_injector(const vec_t<value_t, Conf::dim>& lower,
                    const vec_t<value_t, Conf::dim>& size, value_t inj_rate,
                    value_t inj_weight);

  template <typename WeightFunc>
  void add_injector(const vec_t<value_t, Conf::dim>& lower,
                    const vec_t<value_t, Conf::dim>& size, value_t inj_rate,
                    value_t inj_weight, WeightFunc f) {
    add_injector(lower, size, inj_rate, inj_weight);
    m_weight_funcs.back() = f;
  }

  void init() override;
  void update(double dt, uint32_t step) override;
  void register_data_components() override;

 protected:
  const grid_t<Conf>& m_grid;
  particle_data_t* ptc;

 // private:
  struct injector_params {
    int num;
    int interval;
    value_t weight;
    index_t<Conf::dim> begin;
    extent_t<Conf::dim> ext;
  };

  std::vector<injector_params> m_injectors;
  std::vector<std::function<value_t(value_t, value_t, value_t)>> m_weight_funcs;
  typename Conf::multi_array_t m_ptc_density;

};

#if defined(CUDA_ENABLED) && defined(__CUDACC__)

template <typename Conf>
class ptc_injector_cu : public ptc_injector<Conf> {
 public:
  typedef typename Conf::value_t value_t;
  typedef nvstd::function<value_t(value_t, value_t, value_t)> weight_func_t;
  static std::string name() { return "ptc_injector"; }

  using ptc_injector<Conf>::ptc_injector;
  virtual ~ptc_injector_cu();

  void init() override;
  void update(double dt, uint32_t step) override;
  void register_data_components() override;

  void add_injector(const vec_t<value_t, Conf::dim>& lower,
                    const vec_t<value_t, Conf::dim>& size, value_t inj_rate,
                    value_t inj_weight) {
    ptc_injector<Conf>::add_injector(lower, size, inj_rate, inj_weight);
    m_weight_funcs_dev.push_back(nullptr);
  }

  template <typename WeightFunc>
  void add_injector(const vec_t<value_t, Conf::dim>& lower,
                    const vec_t<value_t, Conf::dim>& size, value_t inj_rate,
                    value_t inj_weight, WeightFunc f) {
    add_injector(lower, size, inj_rate, inj_weight);
    Logger::print_info("Added a weight func");
    CudaSafeCall(cudaMalloc(&(m_weight_funcs_dev.back()),
                            sizeof(weight_func_t)));
    kernel_launch({1, 1}, [] __device__(auto *p, auto f) {
      *p = f;
    }, m_weight_funcs_dev.back(), f);
    CudaSafeCall(cudaDeviceSynchronize());
  }

 protected:
  curand_states_t* m_rand_states;
  multi_array<int, Conf::dim> m_num_per_cell;
  multi_array<int, Conf::dim> m_cum_num_per_cell;

  std::vector<weight_func_t*> m_weight_funcs_dev;
};

#endif

}  // namespace Aperture

#endif  // _PTC_INJECTOR_H_
