#ifndef __BOUNDARY_CONDITION_H_
#define __BOUNDARY_CONDITION_H_

#include "data/fields.h"
#include "data/particle_data.h"
#include "data/curand_states.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/grid.h"
#include <memory>

namespace Aperture {

template <typename Conf>
class boundary_condition : public system_t {
 protected:
  const grid_t<Conf>& m_grid;
  // typename Conf::value_t m_rpert1 = 5.0, m_rpert2 = 10.0;
  typename Conf::value_t m_tp_start, m_tp_end, m_nT, m_dw0;

  vector_field<Conf> *E, *B, *E0, *B0;
  particle_data_t *ptc;
  curand_states_t *rand_states;

  buffer<float> m_surface_n;

 public:
  static std::string name() { return "boundary_condition"; }

  boundary_condition(sim_environment& env, const grid_t<Conf>& grid) :
      system_t(env), m_grid(grid) {}

  void init() override;
  void update(double dt, uint32_t step) override;
};

}

#endif // __BOUNDARY_CONDITION_H_
