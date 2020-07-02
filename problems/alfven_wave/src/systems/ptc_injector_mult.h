#ifndef _PTC_INJECTOR_MULT_H_
#define _PTC_INJECTOR_MULT_H_

#include "systems/ptc_injector.h"

namespace Aperture {

template <typename Conf>
class ptc_injector_mult : public ptc_injector<Conf> {
 public:
  static std::string name() { return "ptc_injector"; }

  using ptc_injector<Conf>::ptc_injector;
  virtual ~ptc_injector_mult() {}

  virtual void init() override;
  virtual void update(double dt, uint32_t step) override;
  virtual void register_data_components() override;

 protected:
  typedef buffer<typename Conf::ndptr_const_t> rho_ptrs_t;

  curand_states_t* m_rand_states;
  vector_field<Conf>* J;
  std::vector<const scalar_field<Conf>*> Rho;
  rho_ptrs_t m_rho_ptrs;
  multi_array<int, Conf::dim> m_num_per_cell;
  multi_array<int, Conf::dim> m_cum_num_per_cell;
  // buffer<int> m_pos_in_array;

  const rho_ptrs_t& get_rho_ptrs() { return m_rho_ptrs; }

};



}

#endif  // _PTC_INJECTOR_MULT_H_
