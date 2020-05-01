#ifndef _PTC_INJECTOR_H_
#define _PTC_INJECTOR_H_

#include "core/enum_types.h"
#include "data/particle_data.h"
#include "data/curand_states.h"
#include "framework/system.h"
#include "systems/domain_comm.h"
#include "systems/grid.h"

namespace Aperture {

template <typename Conf>
class ptc_injector : public system_t {
 protected:
  const grid_t<Conf>& m_grid;

  particle_data_t* ptc;
  vector_field<Conf>* B;

 public:
  typedef typename Conf::value_t value_t;
  static std::string name() { return "ptc_injector"; }

  ptc_injector(sim_environment& env, const grid_t<Conf>& grid)
      : system_t(env), m_grid(grid) {}
  virtual ~ptc_injector() {}

  void init() override;
  void update(double dt, uint32_t step) override;
  void register_dependencies() override;
};

template <typename Conf>
class ptc_injector_cu : public ptc_injector<Conf> {
 protected:
  curand_states_t* m_rand_states;
  buffer<int> m_num_per_cell;
  buffer<int> m_cum_num_per_cell;
  // buffer<int> m_pos_in_array;
 
 public:
  static std::string name() { return "ptc_injector"; }

  ptc_injector_cu(sim_environment& env, const grid_t<Conf>& grid)
      : ptc_injector<Conf>(env, grid) {}
  virtual ~ptc_injector_cu() {}

  void init() override;
  void update(double dt, uint32_t step) override;
  void register_dependencies() override;
};


}

#endif  // _PTC_INJECTOR_H_
