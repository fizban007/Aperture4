#ifndef _COMPUTE_LORENTZ_FACTOR_H_
#define _COMPUTE_LORENTZ_FACTOR_H_

#include "data/fields.h"
#include "data/particle_data.h"
#include "framework/system.h"
#include "systems/grid.h"
#include <vector>

namespace Aperture {

template <typename Conf>
class compute_lorentz_factor : public system_t {
 public:
  static std::string name() { return "compute_lorentz_factor"; }

  compute_lorentz_factor(sim_environment& env, const grid_t<Conf>& grid)
      : system_t(env), m_grid(grid) {}
  virtual ~compute_lorentz_factor() {}

  virtual void register_data_components() override;
  virtual void init() override;
  virtual void update(double dt, uint32_t step) override;

 protected:
  const grid_t<Conf>& m_grid;
  int m_data_interval = 1;

  std::vector<scalar_field<Conf>*> gamma;
  particle_data_t* ptc;

  virtual void register_data_impl(MemType type);
};

template <typename Conf>
class compute_lorentz_factor_cu : public compute_lorentz_factor<Conf> {
 public:
  typedef typename Conf::value_t value_t;
  static std::string name() { return "compute_lorentz_factor"; }

  compute_lorentz_factor_cu(sim_environment& env, const grid_t<Conf>& grid)
      : compute_lorentz_factor<Conf>(env, grid) {}
  virtual ~compute_lorentz_factor_cu() {}

  virtual void register_data_components() override;
  virtual void init() override;
  virtual void update(double dt, uint32_t step) override;

 protected:
  buffer<typename Conf::ndptr_t> m_gamma_ptrs;
  buffer<typename Conf::ndptr_t> m_nums_ptrs;

  std::vector<std::unique_ptr<scalar_field<Conf>>> m_nums;
};

}

#endif  // _COMPUTE_LORENTZ_FACTOR_H_
