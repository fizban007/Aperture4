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
                    value_t inj_weight, const WeightFunc& f) {
    add_injector(lower, size, inj_rate, inj_weight);
    m_injectors.back().weight_func = f;
  }

  void init() override;
  void update(double dt, uint32_t step) override;
  void register_data_components() override;

 protected:
  const grid_t<Conf>& m_grid;
  particle_data_t* ptc;

 private:
  struct injector_params {
    int num;
    int interval;
    value_t weight;
    index_t<Conf::dim> begin;
    extent_t<Conf::dim> ext;
    std::function<value_t(value_t, value_t, value_t)> weight_func = nullptr;
  };

  std::vector<injector_params> m_injectors;
  // vector_field<Conf>* B;

  // value_t m_target_sigma = 100.0;
};

#ifdef CUDA_ENABLED

template <typename Conf>
class ptc_injector_cu : public ptc_injector<Conf> {
 public:
  typedef typename Conf::value_t value_t;
  static std::string name() { return "ptc_injector"; }

  using ptc_injector<Conf>::ptc_injector;
  virtual ~ptc_injector_cu() {}

  void init() override;
  void update(double dt, uint32_t step) override;
  void register_data_components() override;

  using ptc_injector<Conf>::add_injector;
  template <typename WeightFunc>
  void add_injector(const vec_t<value_t, Conf::dim>& lower,
                    const vec_t<value_t, Conf::dim>& size, value_t inj_rate,
                    value_t inj_weight, const WeightFunc& f) {
    this->add_injector(lower, size, inj_rate, inj_weight);
    m_injectors.back().weight_func = f;
  }

 protected:
  curand_states_t* m_rand_states;
  multi_array<int, Conf::dim> m_num_per_cell;
  multi_array<int, Conf::dim> m_cum_num_per_cell;

 private:
  // struct
  // scalar_field<Conf>* m_sigma;
  struct injector_params_dev {
    int num;
    int interval;
    value_t weight;
    index_t<Conf::dim> begin;
    extent_t<Conf::dim> ext;
    nvstd::function<value_t(value_t, value_t, value_t)> weight_func = nullptr;
  };

  std::vector<injector_params_dev> m_injectors;
};

#endif

}  // namespace Aperture

#endif  // _PTC_INJECTOR_H_
